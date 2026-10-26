# Credit-Risk-Determination
An end-to-end credit risk analytics platform that models binary default outcomes, builds a standardized point-allocation scorecard, and deploys an interactive dashboard for credit risk assessment and simulation.

Project Overview

This project focuses on predicting credit default risk across multivariate customer profiles. By transforming a complex logistic regression model into a linear point-scoring system (anchored at a baseline score of 600), this workflow provides risk teams with an interpretable, transparent, and actionable framework for credit decisions. The final model achieves a specificity of 90.09% and a balanced accuracy of 75.66%.

Key Features & Methodology

• Data Preprocessing & Feature Engineering: Managed data irregularities via median imputation for missing values. Reduced high-dimensional feature noise by filtering out near-zero variance (NZV) predictors.
• Statistical Modeling: Implemented a binary logistic regression model. Utilized Wald hypothesis tests to isolate seven statistically significant covariates (p < 0.05), focusing heavily on account liquidity and historical payment patterns.
• Rigorous Debugging: Identified and corrected a critical factor-level mismatch within the design matrix. Resolving this evaluation bias directly improved the model's ability to minimize false positives (90.09% specificity).
• Scorecard Scaling: Transformed log-odds predictive vectors into a standardized, consumer-facing linear point scorecard scaled around a baseline score of 600 points.
• Interactive Deployment: Built a comprehensive R Shiny dashboard to visualize key scorecard metrics and simulate future customer portfolio tracking data.

Repository Structure
├── data/
│   └── (Raw or synthetic credit data files)
├── scripts/
│   ├── 01_data_cleaning.R        # Imputation and NZV filtering
│   ├── 02_logistic_modeling.R     # Regression, Wald tests, & matrix debugging
│   └── 03_scorecard_scaling.R     # Scaling log-odds to the 600-point baseline
├── app/
│   ├── ui.R                      # R Shiny dashboard user interface
│   ├── server.R                  # R Shiny dashboard server logic
│   └── app.R                     # Shiny entry point
└── README.md

How to Run the Project

Prerequisites
Ensure you have R and RStudio installed. You will need the following R packages:
install.packages(c("shiny", "tidyverse", "caret", "stats", "shinythemes"))


Running the Script Workflow

1. Run scripts/01_data_cleaning.R to process raw datasets.
2. Execute scripts/02_logistic_modeling.R to evaluate model diagnostics and view the Wald test outputs.
3. Run scripts/03_scorecard_scaling.R to generate the linear point-scoring system

Launching the Dashboard

To explore the model metrics and run portfolio simulations locally, launch the Shiny application:
library(shiny)
runApp("app/")

Model Performance Results

• Specificity (True Negative Rate): 90.09% — Optimized to effectively protect institutions from high-risk defaulters.
• Balanced Accuracy: 75.66% — Accounting for typical credit dataset class imbalances.
• Core Risk Drivers: Account liquidity and structural payment patterns were identified as the strongest predictors of credit worthiness.
