# =============================================
# ROBUST MODELING PIPELINE FOR NFL INJURY PREDICTION
# =============================================

library(tidyverse)
library(xgboost)
library(caret)
library(Matrix)
library(pROC)

cat("Starting robust modeling pipeline...\n\n")

# =============================================
# 1. DATA PREPARATION AND CLEANING
# =============================================

cat("Step 1: Data preparation and cleaning...\n")

# Create clean modeling dataset
model_data <- final_model_dataset %>%
  # Select available biomechanical features
  select(
    # Target
    injury_next_week,
    
    # Key biomechanical stress features (most important for soft tissue)
    acute_deceleration,
    acute_eccentric,
    acute_collisions,
    
    # Workload metrics
    total_snaps,
    acute_total,
    
    # Injury history
    recent_injury_4wk,
    
    # ACWR metrics (handle NAs)
    acwr_total,
    acwr_deceleration,
    
    # Chronic loads
    chronic_total,
    chronic_deceleration,
    
    # Position
    position,
    
    # Game context
    week,
    season
  ) %>%
  # Handle missing values more robustly
  mutate(
    # Convert NA in recent_injury_4wk to 0 (no recent injury)
    recent_injury_4wk = replace_na(recent_injury_4wk, 0),
    
    # Handle NA in ACWR metrics - use median imputation
    across(c(acwr_total, acwr_deceleration, chronic_total, chronic_deceleration),
           ~ ifelse(is.na(.), median(., na.rm = TRUE), .)),
    
    # Create safer derived features
    decel_per_snap = ifelse(total_snaps > 0, acute_deceleration / total_snaps, 0),
    has_deceleration = as.numeric(acute_deceleration > 0),
    high_deceleration = as.numeric(acute_deceleration >= 2),
    has_recent_injury = as.numeric(recent_injury_4wk > 0),
    
    # Position groups (simplified)
    # position_group = case_when(
    #   position %in% c("RB", "WR") ~ "HighRisk",
    #   position %in% c("TE", "LB", "OLB", "MLB") ~ "MediumRisk",
    #   position == "QB" ~ "QB",
    #   position %in% c("DE", "DT", "NT") ~ "DL",
    #   position %in% c("CB", "S", "FS", "SS", "SAF", "DB") ~ "DB",
    #   position %in% c("T", "G", "C") ~ "OL",
    #   TRUE ~ "Other"
    # ),
    
    # Week effects
    late_season = as.numeric(week >= 12),
    mid_season = as.numeric(week >= 6 & week <= 11),
    
    # Simple risk score
    risk_score = (acute_deceleration * 3) + 
      (has_recent_injury * 5) + 
      (high_deceleration * 2) 
      #ifelse(position_group == "HighRisk", 3, 0)
  ) %>%
  # Remove any remaining NAs
  drop_na() %>%
  # Filter for reasonable activity levels
  filter(total_snaps >= 10 & acute_total >= 5) %>%
  # Remove the original position column
  #select(position_group) %>%
  # Convert factors
  mutate(
    injury_next_week = as.factor(injury_next_week),
    position = as.factor(position)
  )

cat("Clean dataset dimensions:", dim(model_data), "\n")
cat("Injury rate:", round(mean(as.numeric(as.character(model_data$injury_next_week))) * 100, 2), "%\n")
cat("Unique position groups:", n_distinct(model_data$position_group), "\n")

# =============================================
# PROPHET MODEL - ENHANCED VERSION (NO ROSE)
# =============================================
cat("\n=== PROPHET ENHANCED MODEL: IMPROVED PREDICTIVE POWER ===\n")

# Load additional packages
library(Matrix)
library(glmnet)
library(pROC)
library(caret)
library(xgboost)

# Load the dataset


# =============================================
# 1. ADVANCED FEATURE ENGINEERING
# =============================================
cat("\n1. Advanced feature engineering...\n")

model_data_enhanced <- final_model_dataset %>%
  select(
    # Target
    injury_next_week,
    # ACWR features
    acwr_total,
    acwr_deceleration,
    acwr_explosive,
    # Acute workload
    acute_total,
    acute_deceleration,
    acute_collisions,
    acute_explosive,
    acute_eccentric,
    # Chronic workload
    chronic_total,
    chronic_deceleration,
    chronic_explosive,
    # Context
    total_snaps,
    recent_injury_4wk,
    position
  ) %>%
  # Convert to proper types
  mutate(
    injury_next_week = as.numeric(injury_next_week),
    position = as.factor(position)
  ) %>%
  # Handle NAs
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
  
  # --- ENHANCEMENT 1: Create interaction terms ---
  mutate(
    # ACWR x Position interaction (different positions respond differently)
    acwr_position_interaction = acwr_total * as.numeric(position),
    # Acute-Chronic imbalance (spikes in workload)
    acute_chronic_imbalance = acute_total / (chronic_total + 0.1),
    # Deceleration-to-snaps ratio (intensity metric)
    decel_intensity = acute_deceleration / (total_snaps + 1),
    # Recovery indicator (low acute after high chronic)
    recovery_flag = ifelse(acute_total < chronic_total * 0.7, 1, 0),
    # Stress accumulation
    stress_accumulation = recent_injury_4wk * acwr_total,
    # Position risk multiplier
    # position_risk = case_when(
    #   position_group %in% c("WR", "RB", "CB") ~ 1.5,
    #   position_group %in% c("TE", "LB", "S") ~ 1.2,
    #   position_group %in% c("DE", "DT", "OT", "G", "C") ~ 1.0,
    #   TRUE ~ 1.0
    # ),
    # Normalized ACWR (within position)
    acwr_position_z = ave(acwr_total, position,
                          FUN = function(x) {
                            x_mean <- mean(x, na.rm = TRUE)
                            x_sd <- sd(x, na.rm = TRUE)
                            if(x_sd == 0) return(rep(0, length(x)))
                            (x - x_mean) / x_sd
                          }),
    # Load volatility (standard deviation proxy)
    load_volatility = abs(acwr_total - 1) * acute_total,
    # Composite risk score
    composite_risk = acwr_total * decel_intensity * (1 + recent_injury_4wk/4)
  )

# --- FIX: Create safe break calculation function ---
create_safe_breaks <- function(x, probs = c(0, 0.25, 0.5, 0.75, 1)) {
  # Calculate quantiles
  q <- quantile(x, probs = probs, na.rm = TRUE)
  # Ensure breaks are unique
  if (length(unique(q)) < length(q)) {
    # Add small epsilon to make breaks unique
    eps <- seq(0, 1e-10, length.out = length(q))
    q <- q + eps
  }
  # Ensure proper ordering
  q <- sort(q)
  return(q)
}

# Apply safe breaks for deceleration categorization
if (all(is.na(model_data_enhanced$acute_deceleration)) ||
    length(unique(model_data_enhanced$acute_deceleration)) < 4) {
  # If not enough unique values, create simple binary category
  model_data_enhanced$decel_category <- ifelse(
    model_data_enhanced$acute_deceleration > median(model_data_enhanced$acute_deceleration, na.rm = TRUE),
    "High", "Low"
  )
} else {
  # Use safe breaks
  decel_breaks <- create_safe_breaks(model_data_enhanced$acute_deceleration)
  model_data_enhanced$decel_category <- cut(
    model_data_enhanced$acute_deceleration,
    breaks = decel_breaks,
    labels = c("Q1", "Q2", "Q3", "Q4"),
    include.lowest = TRUE
  )
}

# Continue with other feature engineering
model_data_enhanced <- model_data_enhanced %>%
  mutate(
    # --- ENHANCEMENT 2: Create polynomial features for key predictors ---
    acwr_total_sq = acwr_total^2,
    acute_deceleration_sq = acute_deceleration^2,
    acwr_deceleration_sq = acwr_deceleration^2,
    # Logarithmic transforms for skewed distributions
    acute_deceleration_log = log(acute_deceleration + 1),
    total_snaps_log = log(total_snaps + 1),
    # Binned ACWR (fixed breaks)
    acwr_category = cut(acwr_total,
                        breaks = c(-Inf, 0.8, 1.3, 1.5, Inf),
                        labels = c("Underload", "Optimal", "Caution", "Danger"),
                        include.lowest = TRUE)
  ) %>%
  # Convert categoricals to factors
  mutate(
    acwr_category = as.factor(acwr_category),
    decel_category = as.factor(decel_category)
  )

# Check for any NA categories and fix them
model_data_enhanced <- model_data_enhanced %>%
  mutate(
    decel_category = ifelse(is.na(decel_category), "Q2", as.character(decel_category)),
    decel_category = as.factor(decel_category),
    acwr_category = ifelse(is.na(acwr_category), "Optimal", as.character(acwr_category)),
    acwr_category = as.factor(acwr_category)
  )

# Create dummy variables for categorical features
cat_features <- model_data_enhanced %>%
  select(position, acwr_category, decel_category) %>%
  mutate_all(as.factor)

# Check if we have categorical features to encode
if (ncol(cat_features) > 0 && nrow(cat_features) > 0) {
  # Use model.matrix for dummy encoding with error handling
  tryCatch({
    dummy_matrix <- as.data.frame(model.matrix(~ . - 1, data = cat_features))
    # Remove any problematic columns that might be all zeros
    dummy_matrix <- dummy_matrix[, colSums(dummy_matrix) > 0, drop = FALSE]
  }, error = function(e) {
    cat("Warning: Could not create dummy matrix. Error:", e$message, "\n")
    dummy_matrix <- data.frame()
  })
} else {
  dummy_matrix <- data.frame()
}

# Combine with numeric features
numeric_features <- model_data_enhanced %>%
  select(-position, -acwr_category, -decel_category, -injury_next_week) %>%
  select(where(is.numeric))

# Ensure we have columns to work with
if (ncol(numeric_features) == 0) {
  stop("No numeric features found after preprocessing!")
}

model_data_final <- cbind(
  injury_next_week = model_data_enhanced$injury_next_week,
  numeric_features
)

# Add dummy matrix if it exists and has columns
if (ncol(dummy_matrix) > 0) {
  model_data_final <- cbind(model_data_final, dummy_matrix)
}

cat("Enhanced dataset dimensions:", dim(model_data_final), "\n")
cat("Injury rate:", mean(model_data_final$injury_next_week, na.rm = TRUE) * 100, "%\n")
cat("Number of features:", ncol(model_data_final) - 1, "\n")

# =============================================
# 2. HANDLE CLASS IMBALANCE
# =============================================

cat("\nStep 2: Handling class imbalance...\n")

# Stratified train/test split
set.seed(123)
train_idx <- createDataPartition(
  model_data_final$injury_next_week, 
  p = 0.8, 
  list = FALSE,
  times = 1
)

train_data <- model_data_final[train_idx, ]
test_data <- model_data_final[-train_idx, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# Calculate scale_pos_weight for XGBoost
neg_train <- sum(train_data$injury_next_week == "0")
pos_train <- sum(train_data$injury_next_week == "1")
scale_pos_weight_xgb <- neg_train / pos_train

cat("\nClass balance in training set:\n")
cat("  Positive (injury) cases:", pos_train, "\n")
cat("  Negative (healthy) cases:", neg_train, "\n")
cat("  Scale_pos_weight for XGBoost:", round(scale_pos_weight_xgb, 2), "\n")

# =============================================
# 3. PREPARE DATA FOR XGBOOST
# =============================================

cat("\nStep 3: Preparing data for XGBoost...\n")

# # Convert to numeric matrix (handle factors properly)
# prepare_xgb_data <- function(data) {
#   # One-hot encode position_group
#   position_dummies <- model.matrix(~ position_group - 1, data = data)
# 
#   # Select numeric features
#   numeric_features <- data %>%
#     select(-injury_next_week, -position_group) %>%
#     select(where(is.numeric))
# 
#   # Combine all features
#   features_matrix <- cbind(as.matrix(numeric_features), position_dummies)
# 
#   # Return as list with matrix and labels
#   list(
#     features = features_matrix,
#     labels = as.numeric(as.character(data$injury_next_week))
#   )
# }

# Prepare train and test sets
# train_prepped <- train_data
# test_prepped <- test_data
# #test_prepped <- prepare_xgb_data(test_data)
# 
# # Create DMatrices
# dtrain <- xgb.DMatrix(
#   data = train_prepped$features,
#   label = train_prepped$labels
# )
# 
# dtest <- xgb.DMatrix(
#   data = test_prepped$features,
#   label = test_prepped$labels
# )
# Split features and labels
full_x <- as.matrix(model_data_final |> select(-injury_next_week))
full_y <- model_data_final$injury_next_week
train_x <- as.matrix(train_data %>% select(-injury_next_week))
train_y <- train_data$injury_next_week

test_x <- as.matrix(test_data %>% select(-injury_next_week))
test_y <- test_data$injury_next_week

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)
dfull <- xgb.DMatrix(data = full_x, label = full_y)

# Save feature names for later use
feature_names <- colnames(train_x)


# # Feature names for importance
# feature_names <- colnames(train_prepped$features)

# =============================================
# 4. TRAIN XGBOOST WITH PROPER HYPERPARAMETERS
# =============================================

cat("\nStep 4: Training XGBoost model...\n")

# =============================================
# 4.5 HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# =============================================

cat("\nStep 4.5: Hyperparameter tuning with CV...\n")

set.seed(123)

# Define search space (NFL-injury optimized)
param_grid <- expand.grid(
  max_depth = c(2, 3, 4, 5),
  eta = c(0.01, 0.03, 0.05, 0.1),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.6, 0.8, 1.0),
  colsample_bytree = c(0.6, 0.8, 1.0),
  gamma = c(0, 0.5, 1),
  lambda = c(0.5, 1, 2)
)

# Randomly sample combinations (faster than full grid)
n_trials <- min(40, nrow(param_grid))
param_grid_sample <- param_grid[sample(1:nrow(param_grid), n_trials), ]

cv_results <- list()

for (i in 1:n_trials) {
  
  params_cv <- list(
    objective = "binary:logistic",
    eval_metric = "aucpr",  # better for rare injuries
    max_depth = param_grid_sample$max_depth[i],
    eta = param_grid_sample$eta[i],
    min_child_weight = param_grid_sample$min_child_weight[i],
    subsample = param_grid_sample$subsample[i],
    colsample_bytree = param_grid_sample$colsample_bytree[i],
    gamma = param_grid_sample$gamma[i],
    lambda = param_grid_sample$lambda[i],
    scale_pos_weight = scale_pos_weight_xgb
  )
  
  cv <- xgb.cv(
    params = params_cv,
    data = dtrain,
    nrounds = 400,
    nfold = 5,
    stratified = TRUE,
    early_stopping_rounds = 30,
    verbose = 0
  )
  
  best_aucpr <- max(cv$evaluation_log$test_aucpr_mean)
  best_iter <- cv$best_iteration
  
  cv_results[[i]] <- data.frame(
    trial = i,
    aucpr = best_aucpr,
    best_iter = best_iter,
    max_depth = params_cv$max_depth,
    eta = params_cv$eta,
    min_child_weight = params_cv$min_child_weight,
    subsample = params_cv$subsample,
    colsample_bytree = params_cv$colsample_bytree,
    gamma = params_cv$gamma,
    lambda = params_cv$lambda
  )
  
  cat(sprintf("Trial %d/%d — AUC-PR: %.4f\n", i, n_trials, best_aucpr))
}

cv_results_df <- bind_rows(cv_results) %>%
  arrange(desc(aucpr))

best_params <- cv_results_df[1, ]

cat("\n=== BEST HYPERPARAMETERS FOUND ===\n")
print(best_params)


cat("\nStep 4.6: Training final tuned XGBoost model...\n")

best_xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "aucpr",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  min_child_weight = best_params$min_child_weight,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  gamma = best_params$gamma,
  lambda = best_params$lambda,
  scale_pos_weight = scale_pos_weight_xgb
)

xgb_model <- xgb.train(
  params = best_xgb_params,
  data = dtrain,
  nrounds = best_params$best_iter,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 50,
  verbose = 1
)

full_xgb_model <- xgb.train(
  params = best_xgb_params,
  data = dfull,
  nrounds = best_params$best_iter,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 50,
  verbose = 1
)
full_preds <- predict(full_xgb_model, dfull)
roc_obj <- roc(full_y, full_preds)
thresholds <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
optimal_threshold <- thresholds$threshold

cat("Optimal threshold (Youden):", round(optimal_threshold, 3), "\n")

# Predict with optimal threshold
full_pred_classes <- ifelse(full_preds >= optimal_threshold, 1, 0)
confusionMatrix(
  factor(full_pred_classes, levels = c(0, 1)),
  factor(model_data_final$injury_next_week, levels = c(0, 1)),
  positive = "1"
)

cat("\nFinal tuned model trained.\n")
vi(xgb_model) |> 
  slice_head(n = 10) |> 
  mutate(Variable = recode(
    Variable,
    "stress_accumulation" = "Stress Accumulation",
    "total_snaps" = "Total Plays",
    "acute_total" = "Full Acute Workload",
    "acwr_position_z" = "ACWR Position Z-Score",
    "chronic_explosive" = "Chronic Explosive Plays",
    "chronic_total" = "Full Chronic Workload",
    "positionWR" = "Wide Receivers",
    "recent_injury_4wk" = "Recent Injuries",
    "acute_chronic_imbalance" = "ACWR Ratio",
    "acute_explosive" = "Acute Explosive Plays"
  )) |> 
  ggplot(aes(fct_reorder(Variable, Importance), Importance)) +
  geom_col(fill = "dodgerblue2") +
  coord_flip() +
  ggthemes::theme_clean() +
  labs(title = "AEGIS Model Feature Importance", x = "Variable") +
  theme(
    plot.title = ggtext::element_markdown(size = 15, face = "bold"),
    plot.subtitle = ggtext::element_markdown(size = 10, face = "italic"),
    axis.title = ggtext::element_markdown(size = 13, face = "bold"),
    axis.text = ggtext::element_markdown(size = 12),
    text = element_text(family = "PT Sans")
  )

# =============================================
# 5. FEATURE IMPORTANCE AND INTERPRETATION
# =============================================

cat("\nStep 5: Feature importance analysis...\n")

# Get feature importance
importance_matrix <- xgb.importance(
  feature_names = feature_names,
  model = xgb_model
)

print(head(importance_matrix, 15))

# Plot importance
xgb.plot.importance(
  importance_matrix = importance_matrix[1:15, ],
  xlab = "Feature Importance (Gain)",
  main = "Top 15 Features for Injury Prediction"
)

# =============================================
# 6. MODEL EVALUATION
# =============================================

cat("\nStep 6: Model evaluation...\n")

# Make predictions
test_pred_probs <- predict(xgb_model, dtest)

# Find optimal threshold for imbalanced data
roc_obj <- roc(test_y, test_pred_probs)
thresholds <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
optimal_threshold <- thresholds$threshold

cat("Optimal threshold (Youden):", round(optimal_threshold, 3), "\n")

# Predict with optimal threshold
test_pred_classes <- ifelse(test_pred_probs >= optimal_threshold, 1, 0)

# Confusion matrix
conf_matrix <- confusionMatrix(
  factor(test_pred_classes, levels = c(0, 1)),
  factor(test_data$injury_next_week, levels = c(0, 1)),
  positive = "1"
)

print(conf_matrix)

# Calculate additional metrics
cat("\nAdditional Metrics:\n")
cat("AUC:", round(auc(roc_obj), 4), "\n")
cat("Sensitivity (Recall):", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("Precision:", round(conf_matrix$byClass["Precision"], 4), "\n")
cat("F1 Score:", round(conf_matrix$byClass["F1"], 4), "\n")

# =============================================
# 7. PREDICTIONS BY POSITION GROUP
# =============================================

cat("\nStep 7: Analysis by position group...\n")

# Add predictions back to test data
test_results <- test_data %>%
  mutate(
    predicted_prob = test_pred_probs,
    predicted_class = test_pred_classes,
    actual = test_prepped$labels
  )

vip(xgb_model)

# Performance by position group
position_performance <- test_results %>%
  group_by(position_group) %>%
  summarize(
    n = n(),
    injury_rate = mean(actual) * 100,
    avg_predicted_prob = mean(predicted_prob) * 100,
    true_positives = sum(predicted_class == 1 & actual == 1),
    false_positives = sum(predicted_class == 1 & actual == 0),
    false_negatives = sum(predicted_class == 0 & actual == 1),
    precision = ifelse(true_positives + false_positives > 0,
                       true_positives / (true_positives + false_positives), 0),
    recall = ifelse(true_positives + false_negatives > 0,
                    true_positives / (true_positives + false_negatives), 0),
    f1_score = ifelse(precision + recall > 0,
                      2 * (precision * recall) / (precision + recall), 0),
    .groups = 'drop'
  ) %>%
  arrange(desc(injury_rate))

print(position_performance)

# =============================================
# 8. CREATE RISK STRATIFICATION
# =============================================

cat("\nStep 8: Creating risk stratification...\n")

# Create risk categories based on predicted probabilities
test_results <- test_results %>%
  mutate(
    risk_category = case_when(
      predicted_prob >= 0.3 ~ "High Risk (>30%)",
      predicted_prob >= 0.15 ~ "Elevated Risk (15-30%)",
      predicted_prob >= 0.05 ~ "Moderate Risk (5-15%)",
      TRUE ~ "Low Risk (<5%)"
    ),
    risk_category = factor(risk_category,
                           levels = c("Low Risk (<5%)", "Moderate Risk (5-15%)",
                                      "Elevated Risk (15-30%)", "High Risk (>30%)"))
  )

# Analyze actual injury rates by risk category
risk_analysis <- test_results %>%
  group_by(risk_category) %>%
  summarize(
    n_players = n(),
    actual_injuries = sum(injury_next_week),
    injury_rate = mean(injury_next_week) * 100,
    avg_predicted_risk = mean(predicted_prob) * 100,
    .groups = 'drop'
  )

print(risk_analysis)

# =============================================
# 9. SAVE THE MODEL AND RESULTS
# =============================================

cat("\nStep 9: Saving model and results...\n")

# Save the model
saveRDS(xgb_model, "xgboost_injury_model.rds")
cat("Model saved to: xgboost_injury_model.rds\n")

# Save feature importance
importance_df <- as.data.frame(importance_matrix)
write.csv(importance_df, "feature_importance.csv", row.names = FALSE)
cat("Feature importance saved to: feature_importance.csv\n")

# Save test results for analysis
saveRDS(test_results, "test_results.rds")
cat("Test results saved to: test_results.rds\n")

# =============================================
# 10. CREATE SIMPLE PREDICTION FUNCTION
# =============================================

cat("\nStep 10: Creating prediction function...\n")

predict_injury_risk <- function(new_data, model = xgb_model, threshold = optimal_threshold) {
  # Preprocess new data (must match training preprocessing)
  prepped_new <- new_data %>%
    mutate(
      recent_injury_4wk = replace_na(recent_injury_4wk, 0),
      across(c(acwr_total, acwr_deceleration, chronic_total, chronic_deceleration),
             ~ ifelse(is.na(.), median(., na.rm = TRUE), .)),
      decel_per_snap = ifelse(total_snaps > 0, acute_deceleration / total_snaps, 0),
      has_deceleration = as.numeric(acute_deceleration > 0),
      high_deceleration = as.numeric(acute_deceleration >= 2),
      has_recent_injury = as.numeric(recent_injury_4wk > 0),
      late_season = as.numeric(week >= 12),
      mid_season = as.numeric(week >= 6 & week <= 11),
      risk_score = (acute_deceleration * 3) + 
        (has_recent_injury * 5) + 
        (high_deceleration * 2) +
        ifelse(position_group == "HighRisk", 3, 0)
    )
  
  # Prepare matrix
  position_dummies <- model.matrix(~ position_group - 1, data = prepped_new)
  numeric_features <- prepped_new %>%
    select(-position_group) %>%
    select(where(is.numeric))
  features_matrix <- cbind(as.matrix(numeric_features), position_dummies)
  
  # Align columns with training data
  missing_cols <- setdiff(feature_names, colnames(features_matrix))
  extra_cols <- setdiff(colnames(features_matrix), feature_names)
  
  if (length(missing_cols) > 0) {
    for (col in missing_cols) {
      features_matrix <- cbind(features_matrix, 0)
    }
    colnames(features_matrix) <- c(colnames(features_matrix)[1:(ncol(features_matrix)-length(missing_cols))], missing_cols)
  }
  
  # Reorder to match training
  features_matrix <- features_matrix[, feature_names, drop = FALSE]
  
  # Predict
  dmatrix <- xgb.DMatrix(data = features_matrix)
  probabilities <- predict(model, dmatrix)
  
  # Return results
  results <- prepped_new %>%
    mutate(
      injury_probability = round(probabilities * 100, 2),
      injury_risk = ifelse(probabilities >= threshold, "High Risk", "Normal Risk"),
      recommendation = case_when(
        probabilities >= 0.3 ~ "Consider load reduction or extra recovery",
        probabilities >= 0.15 ~ "Monitor closely, consider modified practice",
        TRUE ~ "Continue normal training"
      )
    )
  
  return(results)
}

cat("\nPrediction function created: predict_injury_risk()\n")




# ============================================
# PLOT 2: INJURY RATE BY WEEK OF SEASON
# ============================================
cat("\n=== CREATING PLOT 2: INJURY RATE BY WEEK ===\n")
week_trend <- final_model_dataset %>%
  filter(!is.na(week)) %>%
  group_by(week) %>%
  summarise(
    injury_rate = mean(injury_next_week, na.rm = TRUE) * 100,
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 500, week <= 16)  # Only weeks with enough data
plot2 <- ggplot(week_trend, aes(x = week, y = injury_rate)) +
  geom_line(color = "#3498DB", size = 2, alpha = 0.8) +
  geom_point(color = "#3498DB", size = 3) +
  geom_smooth(method = "loess", se = FALSE, color = "#E74C3C", size = 1.5) +
              # Highlight peak weeks
  geom_point(data = week_trend %>% filter(injury_rate == max(injury_rate)),
                                          color = "#E74C3C", size = 5) +
  annotate("text", x = week_trend$week[which.max(week_trend$injury_rate)],
                y = max(week_trend$injury_rate) + 0.3,
              label = paste0("Peak: ", round(max(week_trend$injury_rate), 1)),
          color = "#E74C3C", fontface = "bold", size = 5) +
 labs(title = "Injury Rate Decreases Through Regular Season", x = "Week of Season",
      y = "Injury Rate (%)") + theme_minimal() + 
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 15, face = "bold", hjust = 0.5))
plot2
                                        
                                        
MULTI SEASON PLOT

cat("\n=== CREATING PLOT 4: MULTI-SEASON TREND ===\n")
season_trend <- final_model_dataset %>%
  filter(!is.na(season)) %>%
  group_by(season) %>%
  summarise(
    injury_rate = mean(injury_next_week, na.rm = TRUE) * 100,
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 1000)  # Only seasons with enough data
plot4 <- ggplot(season_trend, aes(x = season, y = injury_rate, group = 1)) +
  geom_line(color = "#3498DB", size = 2, alpha = 0.8) +
  geom_point(color = "#3498DB", size = 4) +
  geom_text(aes(label = paste0(round(injury_rate, 1), "%")), vjust = -1, size = 5, fontface = "bold", color = "#3498DB") +
                  # Add 17-game season line
  geom_vline(xintercept = 2020.5, linetype = "dashed", color = "#E74C3C") +
  annotate("text", x = 2020.5, y = max(season_trend$injury_rate) * 0.9,
            label = "", color = "#E74C3C", hjust = -0.1,
            fontface = "bold", size = 5, lineheight = 0.8) +
                               # Add trend line
  geom_smooth(method = "lm", se = FALSE, color = "#E74C3C", size = 1.5) +
  labs(
    title = "Injury Rates Over Time (2016-2024)",
    subtitle = "Trend across 9 NFL seasons",
    x = "Season",
    y = "Average Injury Rate (%)"
  ) + theme_minimal() + 
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 14, color = "gray40", hjust = 0.5),
        axis.title = element_text(size = 14, face = "bold"), axis.text = element_text(size = 12),
        panel.grid.minor = element_blank(), plot.margin = margin(30, 30, 30, 30)) +
  scale_x_continuous(breaks = min(season_trend$season):max(season_trend$season))
                                                                                                        ylim(0, max(season_trend$injury_rate) * 1.2)                                              
plot4


library(ggplot2)
library(dplyr)

# Manually define the data
df <- tibble(
  recent_injuries = factor(
    c("0 recent injuries", "1 recent injury", "2 recent injuries", "3+ recent injuries"),
    levels = c("0 recent injuries", "1 recent injury", "2 recent injuries", "3+ recent injuries")
  ),
  injury_rate = c(5.5, 7.9, 8.8, 14.7)
)

# Custom colors (approximate to your plot)
colors <- c("#4FA3D1", "#4CC38A", "#F4A742", "#E85C4A")

ggplot(df, aes(x = recent_injuries, y = injury_rate, fill = recent_injuries)) +
  geom_col(width = 0.65) +
  
  # Labels on bars
  geom_text(aes(label = paste0(injury_rate, "%")),
            vjust = -0.5, fontface = "bold", size = 4) +
  
  scale_fill_manual(values = colors) +
  
  labs(
    title = "Recent Injury History Impact",
    subtitle = "Recent injuries significantly increase future injury risk",
    x = NULL,
    y = "Injury Rate Next Week (%)"
  ) +
  
  ylim(0, 16) +
  
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    axis.text.x = element_text(size = 11),
    axis.text.y = element_text(size = 11)
  )

# Manually define the data
df2 <- tibble(
  decel_bin = factor(
    c("≤1", "2-3", "4-5", "6-7", "8-9", "≥10"),
    levels = c("≤1", "2-3", "4-5", "6-7", "8-9", "≥10")
  ),
  injury_rate = c(2.5, 5.0, 5.7, 6.2, 6.8, 7.5)
)

ggplot(df2, aes(x = decel_bin, y = injury_rate)) +
  geom_col(fill = "#2C7FB8", width = 0.65) +
  
  # Labels on bars
  geom_text(aes(label = paste0(injury_rate, "%")),
            vjust = -0.5, fontface = "bold", size = 4) +
  
  labs(
    title = "Clear Trend: More Decelerations = Higher Injury Risk",
    subtitle = "Steady increase in injury probability with each deceleration event",
    x = "Decelerations",
    y = "Injury Rate Next Week (%)"
  ) +
  
  ylim(0, 9.5) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    axis.text.x = element_text(size = 11),
    axis.text.y = element_text(size = 11)
  )


model_data_final |> 
  pivot_longer(cols = contains("acwr_category")) |> 
  filter(value == 1) |> 
  group_by(name) |> 
  summarise(players = n(), injuries = sum(injury_next_week),
            inj_pct = injuries / players) |> 
  ggplot(aes(name, inj_pct)) +
  geom_col()

# causal inference --------------------------------------------------------


