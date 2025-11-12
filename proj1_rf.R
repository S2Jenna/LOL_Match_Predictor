# ---- Setup ----
library(readr)
library(dplyr)
library(tibble)
library(purrr)
library(xgboost)
library(data.table)
library(caret)
library(Matrix)
library(dplyr)
library(ranger)


# Read input data (same filenames as in the Python notebook)
data <- read_csv("train.csv", show_col_types = FALSE)
Xte  <- read_csv("Xte.csv",   show_col_types = FALSE)

# Unique game ids
gids <- unique(data$gid)

# Containers for one-row records per game
X_rows <- list()
Y_rows <- list()
j <- 0L

# ---- Build Xtr and Ytr (one row per gid) ----
for (gid_i in gids) {
  subset <- dplyr::filter(data, gid == gid_i)
  red  <- dplyr::filter(subset, side == "Red")
  blue <- dplyr::filter(subset, side == "Blue")
  
  # Must have exactly 5 players per side (as in the Python code)
  if (nrow(red) != 5L || nrow(blue) != 5L) next
  
  j <- j + 1L
  if (j %% 5000L == 0L) cat(sprintf("%d / %d\n", j, length(gids)))
  
  # ---- Feature row (X) ----
  rowx <- list(
    gid       = gid_i,
    tid_Blue  = blue$tid[1],
    tid_Red   = red$tid[1],
    date      = subset$date[1],
    lid       = subset$lid[1]
  )
  
  # Player ids
  for (i in 1:5) rowx[[paste0("pid", i, "_Red")]]  <- red$pid[i]
  for (i in 1:5) rowx[[paste0("pid", i, "_Blue")]] <- blue$pid[i]
  
  # Champions
  for (i in 1:5) rowx[[paste0("champion", i, "_Red")]]  <- red$champion[i]
  for (i in 1:5) rowx[[paste0("champion", i, "_Blue")]] <- blue$champion[i]
  
  # --- Add this inside your existing preprocessing loop (after you build `red` and `blue`) ---
  # Normalize and consistently order by role so slot 1..5 are stable across games
  # Adjust the mapping below to match your dataset's exact tokens if needed.
  
  
  normalize_positions <- function(df) {
    df %>%
      dplyr::mutate(position = tolower(position)) %>%
      dplyr::mutate(position = factor(position,
                                      levels = c("top", "jng", "mid", "bot", "sup"))) %>%
      dplyr::arrange(position)
  }
  
  red  <- normalize_positions(red)
  blue <- normalize_positions(blue)
  
  # Position fields (parallel to pid/champion fields)
  for (i in 1:5) {
    rowx[[paste0("position", i, "_Red")]]  <- as.character(red$position[i])
    rowx[[paste0("position", i, "_Blue")]] <- as.character(blue$position[i])
  }
  
  X_rows[[j]] <- tibble::as_tibble(rowx)
  
  # ---- Target row (Y) ----
  rowy <- list(
    gid        = gid_i,
    winner_Red = red$result[1]
  )
  
  for (i in 1:5) rowy[[paste0("k", i, "_Red")]]  <- red$k[i]
  for (i in 1:5) rowy[[paste0("k", i, "_Blue")]] <- blue$k[i]
    
  
  rowy[["gamelength"]] <- subset$gamelength[1]
  
  Y_rows[[j]] <- tibble::as_tibble(rowy)
}


# Bind the accumulated rows
Xtr1 <- dplyr::bind_rows(X_rows)
Ytr1 <- dplyr::bind_rows(Y_rows)

# Match training feature columns to Xte's columns (mirrors pandas DataFrame with columns=Xte.columns)
Xtr1 <- Xtr1 %>% dplyr::select(dplyr::any_of(names(Xte)))

# Optional sanity checks (similar to the notebook exploration)
# print(dim(Xtr1))
# print(any(is.na(Ytr1)))
# print(mean(Ytr1$winner_Red))

# Write outputs
write_csv(Xtr1, "Xtr.csv")
write_csv(Ytr1, "Ytr.csv")

# ---- Baseline predictions (median kills, mean gamelength) ----
preds <- tibble(
  gid         = Xte$gid,
  winner_Red  = 0.5  # constant baseline like the Python notebook
)

for (i in 1:5) {
  rcol <- paste0("k", i, "_Red")
  bcol <- paste0("k", i, "_Blue")
  preds[[rcol]] <- median(Ytr1[[rcol]], na.rm = TRUE)
  preds[[bcol]] <- median(Ytr1[[bcol]], na.rm = TRUE)
}

preds$gamelength <- mean(Ytr1$gamelength, na.rm = TRUE)

write_csv(preds, "baseline.csv")

#################################################################################
# =========================
# 0) Setup
# =========================

set.seed(2025)

# =========================
# 1) Load data
# =========================
Xtr <- read_csv("Xtr.csv", show_col_types = FALSE)
Ytr <- read_csv("Ytr.csv", show_col_types = FALSE)
Xte <- read_csv("Xte.csv", show_col_types = FALSE)

train <- Xtr %>% left_join(Ytr, by = "gid")

K <- 5
fold_map <- Xtr %>%
  dplyr::distinct(gid) %>%
  dplyr::mutate(fold = sample(rep(1:K, length.out = dplyr::n())))

# Build feat_tr, then attach fold ONCE
feat_tr <- Xtr %>%
  dplyr::select(dplyr::any_of(c(team_cols, champ_cols, pid_cols, pos_cols, drop_cols))) %>%
  dplyr::left_join(fold_map, by = "gid")



# =========================
# 2) Choose features (your spec)
#    Remove gid, lid, date; include pids + positions; keep teams/champions
# =========================
pid_cols <- c(paste0("pid", 1:5, "_Red"), paste0("pid", 1:5, "_Blue"))
pos_cols <- c(paste0("position", 1:5, "_Red"), paste0("position", 1:5, "_Blue"))
champ_cols <- c(paste0("champion", 1:5, "_Red"), paste0("champion", 1:5, "_Blue"))
team_cols <- c("tid_Red", "tid_Blue")

# We'll drop gid/lid/date from features:
drop_cols <- c("gid", "lid","date")

feat_tr <- Xtr %>% select(any_of(c(team_cols, champ_cols, pid_cols, pos_cols, drop_cols)))
feat_te <- Xte %>% select(any_of(c(team_cols, champ_cols, pid_cols, pos_cols, drop_cols)))


# =========================
# 3) Out-of-fold target encoding for PIDs (to avoid one-hot blow-up)
#    We encode PIDs by their out-of-fold mean Red win rate + frequency.
#    This captures skill impact while limiting leakage.
# =========================

champ_stats_from <- function(df, m = 20){
  
  #df must have champion, position, and result
  global_avg <- mean(df$result, na.rm = TRUE)
  
  #lane specific champion winrate
  overall <- df %>%
    group_by(champion, position) %>%
    summarize(
      wins = sum(result, na.rm = TRUE),
      games = n(),
      wr_overall = (wins + m * global_avg) / (games + m),
      .groups = "drop"
    )
  
  list(overall = overall, global = global_avg)
}

oof_target_encode <- function(train_df, test_df, cols, y, fold, m = 20) {
  # train_df/test_df : data.frames that contain columns in `cols`
  # cols             : character vector of col names to encode
  # y                : numeric target for training rows (length nrow(train_df))
  # fold             : integer vector (length nrow(train_df)) with values in {1..K}, same game -> same fold
  # m                : smoothing strength for the target mean
  
  stopifnot(length(y) == nrow(train_df), length(fold) == nrow(train_df))
  K <- max(fold, na.rm = TRUE)
  global_mean <- mean(y, na.rm = TRUE)
  
  build_map <- function(df, col) {
    df %>%
      dplyr::group_by(.data[[col]]) %>%
      dplyr::summarise(n = dplyr::n(), sumy = sum(y, na.rm = TRUE), .groups = "drop") %>%
      dplyr::mutate(enc = (sumy + m * global_mean) / (n + m)) %>%
      dplyr::select(level = .data[[col]], n, enc)
  }
  
  # Frequency maps from full train (safe; no target)
  freq_maps <- lapply(cols, function(col) {
    train_df %>%
      dplyr::group_by(.data[[col]]) %>%
      dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
      dplyr::rename(level = .data[[col]], freq = n)
  })
  names(freq_maps) <- cols
  
  tr_enc <- list(); te_enc <- list()
  tr_freq <- list(); te_freq <- list()
  
  for (col in cols) {
    enc_vec <- numeric(nrow(train_df))
    for (f in seq_len(K)) {
      trn_idx <- which(fold != f)
      val_idx <- which(fold == f)
      
      map_f <- build_map(dplyr::bind_cols(train_df[trn_idx, , drop = FALSE], y = y[trn_idx]), col)
      
      val_levels <- tibble::tibble(level = train_df[[col]][val_idx])
      joined     <- dplyr::left_join(val_levels, map_f, by = "level")
      enc_vals   <- ifelse(is.na(joined$enc), global_mean, joined$enc)
      enc_vec[val_idx] <- enc_vals
    }
    tr_enc[[paste0(col, "_te")]] <- enc_vec
    
    # Test encodings from full train
    full_map <- build_map(dplyr::bind_cols(train_df, y = y), col)
    te_levels <- tibble::tibble(level = test_df[[col]])
    joined_te <- dplyr::left_join(te_levels, full_map, by = "level")
    te_enc[[paste0(col, "_te")]] <- ifelse(is.na(joined_te$enc), global_mean, joined_te$enc)
    
    # Frequencies (train/test)
    frmap <- freq_maps[[col]]
    trf <- tibble::tibble(level = train_df[[col]]) %>% dplyr::left_join(frmap, by = "level")
    tef <- tibble::tibble(level = test_df[[col]])  %>% dplyr::left_join(frmap, by = "level")
    tr_freq[[paste0(col, "_freq")]] <- ifelse(is.na(trf$freq), 0, trf$freq)
    te_freq[[paste0(col, "_freq")]] <- ifelse(is.na(tef$freq), 0, tef$freq)
  }
  
  list(
    train = dplyr::bind_cols(tibble::as_tibble(tr_enc), tibble::as_tibble(tr_freq)),
    test  = dplyr::bind_cols(tibble::as_tibble(te_enc), tibble::as_tibble(te_freq))
  )
}

# Attach to feat_tr (must still contain gid here)
feat_tr <- feat_tr %>% dplyr::left_join(fold_map, by = "gid")

enc_pid <- oof_target_encode(
  train_df = feat_tr,
  test_df  = feat_te,
  cols = pid_cols,
  y    = train$winner_Red,
  fold = feat_tr$fold,   # << pass the shared folds
  m = 20
)

# -------- Champion OOF win-rate features (overall) --------
# Requires: champ_stats_from() defined; feat_tr has gid + fold; Xtr/Ytr loaded.

# 1) Build a row-level training table (fallback from Xtr+Ytr)
xy <- Xtr %>%
  dplyr::select(gid, dplyr::starts_with("champion"), dplyr::starts_with("position")) %>%
  dplyr::left_join(Ytr %>% dplyr::select(gid, winner_Red), by = "gid")

train_raw <- xy %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("champion"),
                      names_to = "slot", values_to = "champion") %>%
  dplyr::mutate(side = ifelse(grepl("_Red$", slot), "Red", "Blue"),
                slot_idx = as.integer(gsub("\\D", "", slot))) %>%
  dplyr::left_join(
    xy %>%
      tidyr::pivot_longer(cols = dplyr::starts_with("position"),
                          names_to = "slot_pos", values_to = "position") %>%
      dplyr::mutate(side = ifelse(grepl("_Red$", slot_pos), "Red", "Blue"),
                    slot_idx = as.integer(gsub("\\D", "", slot_pos))) %>%
      dplyr::select(gid, side, slot_idx, position),
    by = c("gid", "side", "slot_idx")
  ) %>%
  dplyr::mutate(
    position = tolower(position),
    result   = ifelse(side == "Red", winner_Red, 1 - winner_Red)
  ) %>%
  dplyr::select(gid, champion, position, result)

# 2) Long views for feat_tr / feat_te
feat_tr <- feat_tr %>% dplyr::mutate(.rowid = dplyr::row_number())

champ_cols_red  <- paste0("champion", 1:5, "_Red")
champ_cols_blue <- paste0("champion", 1:5, "_Blue")
pos_cols_red    <- paste0("position", 1:5, "_Red")
pos_cols_blue   <- paste0("position", 1:5, "_Blue")

long_champs <- function(df) {
  r <- df %>%
    dplyr::select(.rowid, gid, dplyr::all_of(champ_cols_red), dplyr::all_of(pos_cols_red)) %>%
    tidyr::pivot_longer(cols = dplyr::all_of(champ_cols_red),
                        names_to = "slot", values_to = "champion") %>%
    dplyr::mutate(side = "Red", slot_idx = as.integer(gsub("\\D", "", slot))) %>%
    dplyr::select(.rowid, gid, side, slot_idx, champion) %>%
    dplyr::left_join(
      df %>%
        dplyr::select(.rowid, dplyr::all_of(pos_cols_red)) %>%
        tidyr::pivot_longer(cols = dplyr::all_of(pos_cols_red),
                            names_to = "slot_pos", values_to = "position") %>%
        dplyr::mutate(slot_idx = as.integer(gsub("\\D", "", slot_pos))) %>%
        dplyr::select(.rowid, slot_idx, position),
      by = c(".rowid","slot_idx")
    )
  
  b <- df %>%
    dplyr::select(.rowid, gid, dplyr::all_of(champ_cols_blue), dplyr::all_of(pos_cols_blue)) %>%
    tidyr::pivot_longer(cols = dplyr::all_of(champ_cols_blue),
                        names_to = "slot", values_to = "champion") %>%
    dplyr::mutate(side = "Blue", slot_idx = as.integer(gsub("\\D", "", slot))) %>%
    dplyr::select(.rowid, gid, side, slot_idx, champion) %>%
    dplyr::left_join(
      df %>%
        dplyr::select(.rowid, dplyr::all_of(pos_cols_blue)) %>%
        tidyr::pivot_longer(cols = dplyr::all_of(pos_cols_blue),
                            names_to = "slot_pos", values_to = "position") %>%
        dplyr::mutate(slot_idx = as.integer(gsub("\\D", "", slot_pos))) %>%
        dplyr::select(.rowid, slot_idx, position),
      by = c(".rowid","slot_idx")
    )
  
  dplyr::bind_rows(r, b) %>% dplyr::mutate(position = tolower(position))
}

long_tr <- long_champs(feat_tr) %>%
  dplyr::left_join(feat_tr %>% dplyr::select(.rowid, fold), by = ".rowid")

# 3) OOF: for each fold, compute maps on other folds, join to validation
Kfold <- max(feat_tr$fold, na.rm = TRUE)
pieces <- vector("list", Kfold)

for (f in seq_len(Kfold)) {
  tr_df  <- train_raw %>%
    dplyr::left_join(Xtr %>% dplyr::distinct(gid) %>% dplyr::left_join(fold_map, by = "gid"), by = "gid") %>%
    dplyr::filter(fold != f) %>%
    dplyr::select(champion, position, result)
  
  maps <- champ_stats_from(tr_df, m = 20)
  
  val_df <- long_tr %>% dplyr::filter(fold == f) %>%
    dplyr::left_join(maps$overall %>% dplyr::select(champion, position, wr_overall), by = c("champion","position")) %>%
    dplyr::mutate(wr_overall = ifelse(is.na(wr_overall), maps$global, wr_overall)) %>%
    dplyr::mutate(
      col_overall = paste0("ch_wr_overall", slot_idx, "_", side)
    ) %>%
    dplyr::select(.rowid, col_overall, wr_overall)
  
  pieces[[f]] <- val_df
}

enc_tr_df <- dplyr::bind_rows(pieces)

enc_tr_wide <- enc_tr_df %>%
  tidyr::pivot_wider(names_from = col_overall, values_from = wr_overall)

feat_tr <- feat_tr %>% dplyr::left_join(enc_tr_wide, by = ".rowid")

# 4) Apply full-train mapping to TEST
feat_te <- feat_te %>% dplyr::mutate(.rowid = dplyr::row_number())
long_te <- long_champs(feat_te)

maps_full <- champ_stats_from(train_raw, m = 20)

enc_te <- long_te %>%
  dplyr::left_join(maps_full$overall %>% dplyr::select(champion, position, wr_overall), by = c("champion","position")) %>%
  dplyr::mutate(wr_overall = ifelse(is.na(wr_overall), maps_full$global, wr_overall)) %>%
  dplyr::mutate(col_overall = paste0("ch_wr_overall", slot_idx, "_", side)) %>%
  dplyr::select(.rowid, col_overall, wr_overall) %>%
  tidyr::pivot_wider(names_from = col_overall, values_from = wr_overall)

feat_te <- feat_te %>% dplyr::left_join(enc_te, by = ".rowid")

# 5) Clean up helper columns
feat_tr <- feat_tr %>% dplyr::select(-.rowid, -fold)  # drop fold now
feat_te <- feat_te %>% dplyr::select(-.rowid)
# -------- end champion OOF block --------

feat_tr$..rowid <- seq_len(nrow(feat_tr))
feat_te$..rowid <- seq_len(nrow(feat_te))
feat_tr <- feat_tr %>% left_join(bind_cols(..rowid = seq_len(nrow(feat_tr)), enc_pid$train), by = "..rowid")
feat_te <- feat_te %>% left_join(bind_cols(..rowid = seq_len(nrow(feat_te)),  enc_pid$test),  by = "..rowid")
feat_tr <- feat_tr %>% select(-..rowid)
feat_te <- feat_te %>% select(-..rowid)

# Drop raw pid columns (we'll use the encoded + freq versions)
feat_tr <- feat_tr %>% select(-all_of(pid_cols))
feat_te <- feat_te %>% select(-all_of(pid_cols))

# =========================
# 4) Prepare design matrices (one-hot positions/teams/champions; exclude gid/lid/date)
# =========================
prep_for_sparse <- function(df) {
  # Remove gid/lid/date if present
  df <- df %>% select(-any_of(c("gid", "lid", "date")))
  # Ensure character columns (except our numeric encodings) are factors
  char_cols <- names(df)[sapply(df, is.character)]
  for (cn in char_cols) df[[cn]] <- factor(df[[cn]])
  df
}

feat_tr2 <- prep_for_sparse(feat_tr)
feat_te2 <- prep_for_sparse(feat_te)

# Keep only feature columns (no targets here)
# We'll build the sparse matrices from a combined frame to align one-hot levels
combined <- bind_rows(feat_tr2, feat_te2)
sparseX <- sparse.model.matrix(~ . - 1, data = combined)

n_tr <- nrow(feat_tr2)
X_train <- sparseX[1:n_tr, ]
X_test  <- sparseX[(n_tr + 1):nrow(sparseX), ]

# =========================
# 5) Targets
# =========================
y_win <- train$winner_Red
y_gl  <- log1p(train$gamelength)

k_cols <- c(paste0("k", 1:5, "_Red"), paste0("k", 1:5, "_Blue"))

# ====================================================================================================
# Random Forest. --- V1
# ====================================================================================================
# Example: assuming X_train is your feature matrix and y is the target
# Convert to data frame if needed
train_data <- as.data.frame(as.matrix(X_train))  # if X_train is dgCMatrix
colnames(train_data) <- make.names(colnames(train_data), unique = TRUE)
train_data$label <- y  # your target variable (regression)

# Define max_ft (example: sqrt, log2, or a fixed number)
#max_ft <- floor(sqrt(ncol(X_train)))  # or any value like 0.5 * ncol(X_train), or a fixed int
max_ft <- floor(ncol(X_train) / 3)

# OOB RMSE (like oob_score in regression)
cat("OOB RMSE:", sqrt(final_rf1$prediction.error), "\n")
# Load required library
library(ranger)

# =========================
# Helper function for Random Forest cross-validation (optional, for hyperparameter tuning)
# =========================
best_rf_params <- function(dmat, label, 
                           num.trees = 800, 
                           nfold = 5, 
                           min.node.size = 10, 
                           replace = FALSE, 
                           oob.error = TRUE, 
                           seed = 42,
                           num.threads = NULL) {
  # For simplicity, use fixed parameters; ranger doesn't require nrounds tuning like XGBoost
  # Optional: Grid search for mtry, min.node.size if needed
  rf <- ranger(
    formula = label ~ .,
    data = as.data.frame(dmat),
    num.trees = num.trees,
    #mtry = floor(sqrt(ncol(dmat))),  # Default: sqrt of number of features
    mtry = floor(ncol(dmat) / 3),
    min.node.size = min.node.size,
    replace = replace,
    seed = seed,
    importance = "impurity",
    num.threads = num.threads
  )
  return(rf)
}

# =========================
# 7) Winner_Red (binary classification)
# =========================
# Prepare data (X_train and y_win are assumed to be available)
prior_red <- mean(y_win, na.rm = TRUE)  # ~0.47 if Blue ≈ 3% stronger
train_data_win <- as.data.frame(as.matrix(X_train))  # Convert dgCMatrix to dense data frame
# Clean column names to ensure they are valid
colnames(train_data_win) <- make.names(colnames(train_data_win), unique = TRUE)
train_data_win$label <- as.factor(y_win)  # Convert to factor for classification

# Train Random Forest model for Winner_Red
model_win <- ranger(
  formula = label ~ .,
  data = train_data_win,
  num.trees = 800,
  mtry = floor(sqrt(ncol(as.matrix(X_train)))),
  min.node.size = 10,
  replace = FALSE,
  seed = 42,
  num.threads = NULL,
  oob.error = TRUE,
  probability = TRUE,  # For probability predictions
  importance = "impurity",
  classification = TRUE
)

# Predict probabilities for Winner_Red (probability of class 1)
test_data_win <- as.data.frame(as.matrix(X_test))  # Convert test data to dense data frame
colnames(test_data_win) <- make.names(colnames(test_data_win), unique = TRUE)  # Clean test column names
pred_win <- predict(model_win, data = test_data_win)$predictions[, 2]

# =========================
# 8) Gamelength (regression on log1p)
# =========================
# Prepare data for regression
train_data_gl <- as.data.frame(as.matrix(X_train))  # Convert dgCMatrix to dense data frame
colnames(train_data_gl) <- make.names(colnames(train_data_gl), unique = TRUE)  # Clean column names
train_data_gl$label <- y_gl  # y_gl is the log1p-transformed gamelength

# Train Random Forest model for Gamelength
model_gl <- ranger(
  formula = label ~ .,
  data = train_data_gl,
  num.trees = 800,
  #mtry = floor(sqrt(ncol(as.matrix(X_train)))),
  mtry = floor(ncol(as.matrix(X_train)) / 3),
  min.node.size = 10,
  importance = "impurity",
  replace = FALSE,
  oob.error = TRUE,
  seed = 42,
  num.threads = NULL
)

# Predict and transform back from log1p
test_data_gl <- as.data.frame(as.matrix(X_test))  # Convert test data to dense data frame
colnames(test_data_gl) <- make.names(colnames(test_data_gl), unique = TRUE)  # Clean test column names
pred_gl <- pmax(1, round(expm1(predict(model_gl, data = test_data_gl)$predictions)))

# =========================
# 9) (Optional) Kills regressors (10 models)
# =========================
pred_kills <- list()
for (col in k_cols) {
  # Prepare data for each kill column
  y_k <- train[[col]]
  train_data_k <- as.data.frame(as.matrix(X_train))  # Convert dgCMatrix to dense data frame
  colnames(train_data_k) <- make.names(colnames(train_data_k), unique = TRUE)  # Clean column names
  train_data_k$label <- y_k
  
  # Train Random Forest model for kills
  model_k <- ranger(
    formula = label ~ .,
    data = train_data_k,
    num.trees = 800,
    #mtry = floor(sqrt(ncol(as.matrix(X_train)))),
    mtry = floor(ncol(as.matrix(X_train)) / 3),
    min.node.size = 10,
    replace = FALSE,
    importance = "impurity",
    oob.error = TRUE,
    seed = 42,
    num.threads = NULL
  )
  
  # Predict and ensure non-negative integer outputs
  test_data_k <- as.data.frame(as.matrix(X_test))  # Convert test data to dense data frame
  colnames(test_data_k) <- make.names(colnames(test_data_k), unique = TRUE)  # Clean test column names
  pred_k <- predict(model_k, data = test_data_k)$predictions
  pred_kills[[col]] <- pmax(0, round(pred_k))
}


# ----------
# Precompute common objects ONCE
train_df <- as.data.frame(as.matrix(X_train))
test_df  <- as.data.frame(as.matrix(X_test))
colnames(train_df) <- make.names(colnames(train_df), unique = TRUE)
colnames(test_df)  <- colnames(train_df)  # ensures column alignment safely

#mtry_val <- floor(sqrt(ncol(train_df)))   # compute once

pred_kills <- vector("list", length(k_cols))
names(pred_kills) <- k_cols

# Loop only the label — everything else reused
for (col in k_cols) {
  y_k <- train[[col]]
  
  # attach label without copying full DF
  train_df$label <- y_k
  
  model_k <- ranger(
    label ~ .,
    data = train_df,
    num.trees = 800,
    #mtry = mtry_val,
    mtry = 2,
    min.node.size = 10,
    replace = FALSE,
    importance = "impurity",
    oob.error = FALSE,        # disable if you're not using it
    seed = 42,
    num.threads = parallel::detectCores()  # FULL SPEED
  )
  
  pred_k <- predict(model_k, data = test_df)$predictions
  pred_kills[[col]] <- pmax(0, round(pred_k))
}

# optional: drop label back out to avoid confusion
train_df$label <- NULL

# ------

# Convert predictions to data frame
kills_df <- as.data.frame(pred_kills, check.names = FALSE)



# =========================
# 10) Final predictions
# =========================
preds <- tibble(
  gid = Xte$gid,  # keep gid in the output (not as a feature)
  winner_Red = pred_win,
  k1_Red  = kills_df[["k1_Red"]],  k2_Red  = kills_df[["k2_Red"]],
  k3_Red  = kills_df[["k3_Red"]],  k4_Red  = kills_df[["k4_Red"]],
  k5_Red  = kills_df[["k5_Red"]],
  k1_Blue = kills_df[["k1_Blue"]], k2_Blue = kills_df[["k2_Blue"]],
  k3_Blue = kills_df[["k3_Blue"]], k4_Blue = kills_df[["k4_Blue"]],
  k5_Blue = kills_df[["k5_Blue"]],
  gamelength = as.numeric(pred_gl)
)

write_csv(preds, "rf_preds3.csv")
cat("Wrote rf_preds.csv\n")

