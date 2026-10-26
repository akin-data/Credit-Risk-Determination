# ==============================================================================
# Credit Risk Prediction Project: End-to-End Enterprise Production 
# ==============================================================================

# ------------------------------------------------------------------------------
# Load the required library
# ------------------------------------------------------------------------------
library(tidyverse)   # 	Data manipulation and visualization
library(caret)       # 	Machine learning workflow management
library(pROC)        #  ROC curve and AUC analysis
library(MLeval)      #  Automated machine learning model evaluation

# Load Raw Data
credit_data <- read.csv("c:/Users/user/OneDrive/Desktop/CreditRisk_Data1.csv", stringsAsFactors = FALSE)

# Clean out reference index column (OBS#)
refined_data <- credit_data %>% select(-1)

# CRISP-DM FIX: Set "Good" as reference level (0) and "Default" as target event (1)
refined_data$DEFAULT <- factor(
  ifelse(refined_data$DEFAULT == 1, "Default", "Good"), 
  levels = c("Good", "Default") # <--- FIX: "Default" is now the positive event
)

# Generate the distribution plot (Maintained order for visualization mapping)
ggplot(refined_data, aes(x = DEFAULT, fill = DEFAULT)) +
  geom_bar() +
  scale_fill_manual(values = c("Default" = "#E41A1C", "Good" = "#377EB8")) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Distribution of Loan Defaults",
    x = "Loan Status (Target Class)",
    y = "Count of Applicants",
    fill = "Status"
  )


# ------------------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------------------
# Data Splitting (80% Train / 20% Test)
set.seed(123)
train_index <- createDataPartition(refined_data$DEFAULT, p = 0.8, list = FALSE)
train_data  <- refined_data[train_index, ]
test_data   <- refined_data[-train_index, ]

# Isolate target from processing pipeline
train_target <- train_data %>% select(-DEFAULT)
test_target  <- test_data %>% select(-DEFAULT)

# Filter out Near-Zero Variance Predictors
nzv_metrics    <- nearZeroVar(train_target, saveMetrics = TRUE)
valid_target <- rownames(nzv_metrics[nzv_metrics$nzv == FALSE, ])
train_target <- train_target %>% select(all_of(valid_target))
test_target  <- test_target %>% select(all_of(valid_target))

# Robust Feature Engineering Pipeline (Imputation, Center, Scale)
prep_pipeline <- preProcess(train_target, method = c("medianImpute", "center", "scale"))

# Apply transformations and re-bind Target Factor
train_prepared         <- predict(prep_pipeline, train_target)
train_prepared$DEFAULT <- train_data$DEFAULT
test_prepared          <- predict(prep_pipeline, test_target)
test_prepared$DEFAULT  <- test_data$DEFAULT

# ------------------------------------------------------------------------------
# MODELING
# ------------------------------------------------------------------------------
# Primary Selected Model: Logistic Regression
log_model <- glm(DEFAULT ~ ., data = train_prepared, family = binomial(link = "logit"))
log_model

# VISUAL: Risk Driver Importance
# Extracting the Statistical Summary
model_summary     <- as.data.frame(summary(log_model)$coefficients)
colnames(model_summary) <- c("Estimate", "StdError", "zValue", "pValue")
model_summary$Variable  <- rownames(model_summary)
model_summary

# Filtering for Signficant Drivers
significant_vars <- model_summary %>% 
  filter(Variable != "(Intercept)" & pValue < 0.05) %>% 
  mutate(Effect = ifelse(Estimate > 0, "Increases Risk", "Decreases Risk"))

# Creating the Visualization
ggplot(significant_vars, aes(x = reorder(Variable, Estimate), y = Estimate, fill = Effect)) +
  geom_bar(stat = "identity", color = "black", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("Increases Risk" = "#f1a340", "Decreases Risk" = "#998ec3")) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Top Significant Risk Drivers",
    subtitle = "Statistical Features with p-value < 0.05",
    x = "Predictor Variables",
    y = "Log-Odds Estimate Weight",
    fill = "Risk Direction"
  )

# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

# 1. Generate Probabilities on Test Set (Probability of being "Default")
test_probs_log <- predict(log_model, newdata = test_prepared, type = "response")

# 2. Format Framework Data Frame for MLeval Comparison
# Column 1 MUST be the positive class probability (Default), Column 2 the negative (Good)
eval_df_log <- data.frame(
  Default = test_probs_log,
  Good    = 1 - test_probs_log,
  obs     = as.character(test_prepared$DEFAULT), # Ground truth string match
  Group   = "Logistic Regression"
)

# Run evaluation dashboard via MLeval specifying "Default" as target
run_eval <- evalm(eval_df_log, plots = "all", positive = "Default")
run_eval

# 3. Extract Confusion Matrix using a 0.5 Decision Boundary
# Since "Default" is the second factor level, ifelse maps it when prob > 0.5
predictions_log <- factor(
  ifelse(test_probs_log > 0.5, "Default", "Good"), 
  levels = c("Good", "Default") # <--- FIX: Kept consistent with original factor levels
)

# Calculate confusion matrix tracking "Default" as the critical positive metric
cm_log <- confusionMatrix(
  data = predictions_log, 
  reference = test_prepared$DEFAULT, 
  positive = "Default", 
  mode = "everything"
)
cm_log

# Print Summary Performance Tables
cat("\n=============================================\n")
cat("          PERFORMANCE SUMMARY      \n")
cat("=============================================\n")
cat("LOGISTIC REGRESSION METRICS:\n",
    " Sensitivity (Recall) : ", round(cm_log$byClass["Sensitivity"], 4), "\n",
    " Specificity          : ", round(cm_log$byClass["Specificity"], 4), "\n",
    " Precision            : ", round(cm_log$byClass["Precision"], 4), "\n",
    " Balanced Accuracy    : ", round(cm_log$byClass["Balanced Accuracy"], 4), "\n")
cat("=============================================\n")



# ------------------------------------------------------------------------------
# DEPLOYMENT (SCORECARD CONVERSION & EXPORT)
# ------------------------------------------------------------------------------
# Scorecard Parameters
target_score <- 600
target_odds  <- 50 
pdo          <- 20 

# Mathematical Scaling Calculations
factor_scaled <- pdo / log(2)
offset_scaled <- target_score - (factor_scaled * log(target_odds))

# Isolate Estimates
coef_matrix      <- summary(log_model)$coefficients
scorecard_points <- data.frame(
  Variable        = rownames(coef_matrix),
  Raw_Coefficient = coef_matrix[, "Estimate"],
  stringsAsFactors = FALSE
)

#  Maps individual points proportional to feature variance scale
scorecard_points <- scorecard_points %>%
  mutate(
    Scorecard_Points = ifelse(
      Variable == "(Intercept)",
      round((Raw_Coefficient * -factor_scaled) + offset_scaled),
      round(Raw_Coefficient * -factor_scaled)
    )
  )


# Display final business points matrix
cat("\n=== DEPLOYMENT: CREDIT SCORECARD LOOKUP TABLE ===\n")
print(scorecard_points %>% select(Variable, Scorecard_Points))

# VISUAL 4: Scorecard Distributions
ggplot(scorecard_points %>% filter(Variable != "(Intercept)"), 
       aes(x = reorder(Variable, Scorecard_Points), y = Scorecard_Points, fill = Scorecard_Points > 0)) +
  geom_bar(stat = "identity", color = "black", width = 0.6) +
  coord_flip() +
  scale_fill_manual(
    values = c("TRUE" = "#2ca25f", "FALSE" = "#dd1c77"),
    labels = c("TRUE" = "Adds Score Points (Reduces Risk)", "FALSE" = "Deducts Score Points (Increases Risk)")
  ) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Final Point Allocations",
    x = "Predictors",
    y = "Points Impact",
    fill = "System Vector Effect"
  )

# DEPLOYMENT STEP: Serialize items for Production App Integration
saveRDS(prep_pipeline, "credit_risk_pipeline.rds")
saveRDS(log_model, "credit_risk_logistic_model.rds")

cat("\nProduction RDS binary assets written successfully to workspace directory.\n")

