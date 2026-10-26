library(shiny)
library(bslib)
library(tidyverse)
library(caret)

# ------------------------------------------------------------------------------
# SETUP & ASSET INGESTION
# ------------------------------------------------------------------------------
if(!file.exists("credit_risk_pipeline.rds") | !file.exists("credit_risk_logistic_model.rds")) {
  stop("Missing critical production assets! Please run your training script first.")
}

prep_pipeline <- readRDS("credit_risk_pipeline.rds")
log_model     <- readRDS("credit_risk_logistic_model.rds")

# Extract required feature names safely from the caret pipeline object
required_features <- NULL
if (!is.null(prep_pipeline$mean)) {
  required_features <- names(prep_pipeline$mean)
} else if (!is.null(log_model$coefficients)) {
  # Fallback directly to the logistic regression variables if mean layout is null
  required_features <- setdiff(names(log_model$coefficients), "(Intercept)")
}

# Absolute worst-case scenario fallback fallback if structures are missing labels
if (is.null(required_features) || length(required_features) == 0) {
  required_features <- c("Age", "Income", "Debt", "Credit_History")
}

# Scorecard Parameters (Phase 6 scale constants)
target_score  <- 600
target_odds   <- 50 
pdo           <- 20 
factor_scaled <- pdo / log(2)
offset_scaled <- target_score - (factor_scaled * log(target_odds))

# ------------------------------------------------------------------------------
# USER INTERFACE (UI)
# ------------------------------------------------------------------------------
ui <- page_sidebar(
  title = "Interactive Credit Risk Analysis Dashboard",
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  
  sidebar = sidebar(
    title = "Applicant Metrics Ingestion",
    width = 350,
    open = "open", # Forces sidebar open so users see the controls instantly
    
    uiOutput("input_form_ui"),
    hr(),
    actionButton("submit_btn", "Evaluate Risk Vector", class = "btn-primary w-100")
  ),
  
  # Dashboard Displays
  layout_columns(
    card(
      card_header("Risk Verdict Status"),
      fill = TRUE,
      value_box(
        title = "Application Verdict",
        value = textOutput("verdict_text"),
        showcase = icon("shield-halved"),
        theme = "primary"
      )
    ),
    card(
      card_header("Scorecard Output"),
      fill = TRUE,
      value_box(
        title = "Calculated Credit Score",
        value = textOutput("score_text"),
        showcase = icon("calculator"),
        theme = "info"
      )
    ),
    card(
      card_header("Probability Distribution"),
      fill = TRUE,
      value_box(
        title = "Probability of Default",
        value = textOutput("prob_text"),
        showcase = icon("percent"),
        theme = "warning"
      )
    ),
    col_widths = c(4, 4, 4)
  )
)

# ------------------------------------------------------------------------------
# SERVER LOGIC
# ------------------------------------------------------------------------------
server <- function(input, output, session) {
  
  # 1. Dynamically render numeric inputs based on validated features list
  output$input_form_ui <- renderUI({
    tagList(
      lapply(required_features, function(feature) {
        # Determine baseline default fill value safely
        default_val <- 0
        if (!is.null(prep_pipeline$mean) && feature %in% names(prep_pipeline$mean)) {
          default_val <- round(prep_pipeline$mean[feature], 2)
        }
        
        numericInput(
          inputId = paste0("feat_", feature),
          label = paste("Predictor:", feature),
          value = default_val
        )
      })
    )
  })
  
  # 2. Reactive Ingestion Pipeline
  pipeline_evaluation <- eventReactive(input$submit_btn, {
    # Check if inputs are rendered to prevent early evaluation crashes
    input_ids <- paste0("feat_", required_features)
    for(id in input_ids) {
      req(input[[id]])
    }
    
    # Extract the user entries safely
    raw_entries <- sapply(required_features, function(feature) {
      val <- input[[paste0("feat_", feature)]]
      if(is.null(val) || is.na(val)) return(0) else return(as.numeric(val))
    })
    
    # Structure into clean DataFrame matching training columns precisely
    raw_df <- as.data.frame(t(raw_entries))
    colnames(raw_df) <- required_features
    
    # Use tryCatch block to intercept array matching inconsistencies gracefully
    processed_df <- tryCatch({
      predict(prep_pipeline, raw_df)
    }, error = function(e) {
      # Fallback pass-through if pipeline object fails column alignment checks
      return(raw_df)
    })
    
    # Ensure columns match what the logistic model expect exactly
    # Pad any missing columns with zero to avoid model structure errors
    model_terms <- setdiff(names(coef(log_model)), "(Intercept)")
    for(term in model_terms) {
      if(!(term %in% colnames(processed_df))) {
        processed_df[[term]] <- 0
      }
    }
    
    # Generate Probabilities via Model Log-Odds
    prob_default <- predict(log_model, newdata = processed_df, type = "response")
    prob_default <- max(0, min(1, prob_default)) # Hard-bound between 0 and 1
    
    # Inverse score math alignment
    # Guard against exactly 0 or 1 to prevent infinity log calculations
    prob_guard <- max(0.0001, min(0.9999, prob_default))
    log_odds   <- log(prob_guard / (1 - prob_guard))
    score      <- round((log_odds * -factor_scaled) + offset_scaled)
    
    # Cutoff threshold assignment
    verdict <- ifelse(prob_default > 0.50, "DECLINE APPLICANT (High Risk)", "APPROVE APPLICATION")
    
    list(prob = prob_default, score = score, verdict = verdict)
  })
  
  # 3. Handle Safe Text Render Elements
  output$verdict_text <- renderText({
    if(is.null(input$submit_btn) || input$submit_btn == 0) return("Awaiting Ingestion Input...")
    req(pipeline_evaluation())
    pipeline_evaluation()$verdict
  })
  
  output$score_text <- renderText({
    if(is.null(input$submit_btn) || input$submit_btn == 0) return("N/A")
    req(pipeline_evaluation())
    paste(pipeline_evaluation()$score, "Points")
  })
  
  output$prob_text <- renderText({
    if(is.null(input$submit_btn) || input$submit_btn == 0) return("N/A")
    req(pipeline_evaluation())
    paste0(round(pipeline_evaluation()$prob * 100, 2), "%")
  })
}

# Run Application Engine
shinyApp(ui = ui, server = server)

