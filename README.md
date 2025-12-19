## League of Legends Match Outcome & Performance Prediction

This project builds a full machine learning pipeline to predict League of Legends match outcomes, player-level kills, and game length using structured match and roster data.
The workflow focuses on robust feature engineering, leakage-safe target encoding, and Random Forest models trained on game-level representations.

### 📌 Project Overview
Built a machine learning pipeline to predict match winners, player kills, and game length using historical League of Legends match data.
- Transformed player-level data into one row per game, aligning with test-set structure (players, positions, champions for both teams).
- Engineered features including team-level aggregates and player/champion target encodings with out-of-fold validation to prevent data leakage.
- Trained Random Forest classification and regression models to predict match outcomes, per-player kills (10 outputs per game), and game duration.
- Implemented a reproducible preprocessing pipeline ensuring consistent feature generation between training and test sets.
Tools: R, dplyr, data.table, caret, ranger, sparse matrices
