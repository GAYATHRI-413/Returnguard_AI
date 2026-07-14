/**
 * generate_docs.js — builds docs/ReturnGuard_AI_Documentation.docx
 */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, BorderStyle, AlignmentType, ImageRun, PageBreak,
  ShadingType, TableOfContents, VerticalAlign,
} = require("docx");

const H1 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
const H2 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
const H3 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_3, spacing: { before: 150, after: 80 } });
const P = (text) => new Paragraph({ children: [new TextRun({ text })], spacing: { after: 120 } });
const bullet = (text) => new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
const numbered = (text, n) => new Paragraph({ children: [new TextRun(`${n}. ${text}`)], spacing: { after: 60 } });

const codeBlock = (text) => new Paragraph({
  children: text.split("\n").map((l, i) => new TextRun({ text: l, font: "Consolas", size: 18, break: i === 0 ? 0 : 1 })),
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
  spacing: { after: 150 },
  border: {
    top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
    left: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
    right: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" },
  },
});

function makeTable(headers, rows, widths) {
  const totalWidth = 9000;
  const colWidths = widths || headers.map(() => Math.floor(totalWidth / headers.length));
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "2C3E50" },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 20 })] })],
    })),
  });
  const bodyRows = rows.map((row, ridx) => new TableRow({
    children: row.map((cell, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ridx % 2 === 0 ? "FFFFFF" : "F7F7F7" },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20 })] })],
    })),
  }));
  return new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: colWidths, rows: [headerRow, ...bodyRows] });
}

function image(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
  });
}
const caption = (text) => new Paragraph({
  children: [new TextRun({ text, italics: true, size: 18, color: "666666" })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 250 },
});

const PLOTS = "/home/claude/returnguard-ai/ml/models/plots";
const DOCS = "/home/claude/returnguard-ai/docs";
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const sections = [];

// ============================================================ COVER
sections.push(
  new Paragraph({ text: "", spacing: { before: 1500 } }),
  new Paragraph({ children: [new TextRun({ text: "🛡️ ReturnGuard AI", bold: true, size: 64, color: "1F4E79" })], alignment: AlignmentType.CENTER, spacing: { after: 300 } }),
  new Paragraph({ children: [new TextRun({ text: "Advanced AI Powered Return Fraud Detection & Risk Intelligence System", size: 32, color: "444444" })], alignment: AlignmentType.CENTER, spacing: { after: 800 } }),
  new Paragraph({ children: [new TextRun({ text: "Complete Technical & Project Documentation", size: 26, italics: true })], alignment: AlignmentType.CENTER, spacing: { after: 2000 } }),
  new Paragraph({ children: [new TextRun({ text: "A production-style ML/DL/MLOps engineering deliverable", size: 20, color: "666666" })], alignment: AlignmentType.CENTER }),
  pageBreak(),
);

// ============================================================ TOC
const tocEntries = [
  "1. Complete Project Workflow", "2. Folder Structure Explanation", "3. Every File Explanation",
  "4. Every Function Explanation", "5. Every Class Explanation", "6. Every API Explanation",
  "7. Machine Learning Pipeline", "8. Deep Learning Architecture", "9. Neural Network Layer-by-Layer Explanation",
  "10. Data Flow", "11. Feature Engineering Workflow", "12. Training Workflow", "13. Prediction Workflow",
  "14. Deployment Workflow", "15. Docker Workflow", "16. API Workflow", "17. Streamlit Workflow",
  "18. Database Workflow", "19. Sequence Diagram", "20. Architecture Diagram",
  "Appendix A — Model Evaluation Plots", "21. Interview Questions and Answers", "22. Viva Questions",
  "23. Future Enhancements", "24. Limitations", "25. Business Impact", "26. Conclusion",
];
sections.push(
  H1("Table of Contents"),
  ...tocEntries.map((t) => new Paragraph({ children: [new TextRun({ text: t, size: 22 })], spacing: { after: 100 } })),
  pageBreak(),
);

// ============================================================ 1
sections.push(
  H1("1. Complete Project Workflow"),
  P("ReturnGuard AI is an end-to-end machine learning system that scores incoming e-commerce return requests for fraud risk, in real time, with a full explanation and a recommended business action. The system is organized into six cooperating layers:"),
  numbered("Data Layer — a realistic synthetic dataset generator that encodes business logic (not random labels) into 12,000 return records.", 1),
  numbered("Feature Engineering Layer — derives risk-relevant features offline (training) and online (inference) using identical formulas.", 2),
  numbered("Modeling Layer — trains and compares four model families and automatically promotes the best one.", 3),
  numbered("Explainability Layer — wraps SHAP around the deployed model so every prediction carries a justification.", 4),
  numbered("Serving Layer — a FastAPI REST API that validates, engineers features, predicts, explains, applies rules, and persists results.", 5),
  numbered("Presentation Layer — a Streamlit dashboard: prediction form, history browser, analytics dashboard, model metrics viewer.", 6),
  P("Request lifecycle: a user fills in a return's details in Streamlit → validated by Pydantic → features engineered → best model produces a fraud probability → SHAP explains top factors → Business Rule Engine converts probability into a 0-100 risk score and one of three actions → result persisted and rendered."),
  P("Offline workflow: generate dataset → preprocess (encode + scale + split) → train all four models → compare metrics → auto-select and persist best model → serving layer picks it up on next load or via POST /retrain."),
);

// ============================================================ 2
sections.push(
  H1("2. Folder Structure Explanation"),
  P("The project follows a modular, service-oriented layout mirroring how a real ML platform team organizes a fraud-detection service:"),
  codeBlock(
`returnguard-ai/
├── backend/app/
│   ├── main.py                    FastAPI app + lifespan startup
│   ├── api/routes/                 HTTP route handlers (thin controllers)
│   ├── api/schemas/                 Pydantic request/response contracts
│   ├── core/                        settings, logging, security
│   ├── db/                          SQLAlchemy models, session, CRUD
│   └── services/                    business logic
├── ml/
│   ├── data/generate_dataset.py             synthetic dataset generator
│   ├── preprocessing/preprocessor.py         encode / scale / split
│   ├── training/                            4 training scripts + comparison
│   ├── explainability/shap_explainer.py      SHAP wrapper
│   └── models/                               saved artifacts + plots
├── frontend/
│   ├── streamlit_app.py            login + home
│   ├── pages/                       4 dashboard pages
│   └── utils/                       API client + dark theme CSS
├── database/schema.sql             reference SQL DDL
├── docker/                         Dockerfile.backend, Dockerfile.frontend
├── docker-compose.yml              full local stack
├── render.yaml                     Render.com deployment blueprint
├── config/config.yaml               tunable, non-secret settings
├── config/.env.example              secrets template
├── docs/                            this documentation
├── tests/test_api.py                pytest API test suite
└── requirements.txt`),
  P("Each layer changes independently: swapping databases only touches database.py; adding a fifth model only touches ml/training/; changing the UI only touches frontend/. The backend/services layer is the only thing that reaches into both the ML layer and the database layer."),
);

// ============================================================ 3
sections.push(
  H1("3. Every File — Purpose Explanation"),
  H2("Configuration"),
  makeTable(["File", "Purpose"], [
    ["config/config.yaml", "Single source of truth for paths, dataset params, feature lists, hyperparameters, risk thresholds."],
    ["config/.env.example", "Template for secrets: DB URL, JWT secret, admin credentials."],
    ["backend/app/core/config.py", "Loads both into a typed Settings singleton used everywhere."],
    ["backend/app/core/logging_config.py", "Configures loguru with console + rotating file sinks."],
    ["backend/app/core/security.py", "Password hashing (bcrypt) and JWT issuing/verification."],
  ]),
  H2("Machine Learning"),
  makeTable(["File", "Purpose"], [
    ["ml/data/generate_dataset.py", "Generates the 12,000-row synthetic dataset via weighted business rules."],
    ["ml/preprocessing/preprocessor.py", "Encodes categoricals, scales numerics, splits train/val/test, persists artifacts."],
    ["ml/training/train_utils.py", "Shared metrics + plotting used by all training scripts."],
    ["ml/training/train_logistic_regression.py", "Trains/saves the Logistic Regression baseline."],
    ["ml/training/train_random_forest.py", "Trains/saves the Random Forest ensemble."],
    ["ml/training/train_xgboost.py", "Trains/saves the XGBoost gradient-boosted model."],
    ["ml/training/train_deep_learning.py", "Builds/trains/saves the Keras DNN with callbacks."],
    ["ml/training/model_comparison.py", "Trains all 4, builds comparison table, selects + persists best model."],
    ["ml/explainability/shap_explainer.py", "Fits the right SHAP explainer per model family; formats results."],
  ]),
  H2("Backend"),
  makeTable(["File", "Purpose"], [
    ["backend/app/main.py", "FastAPI app, CORS, router registration, lifespan startup."],
    ["backend/app/db/database.py", "Engine/session; Postgres via DATABASE_URL or SQLite fallback."],
    ["backend/app/db/models.py", "ORM tables: Customer, Seller, PredictionHistory, ModelMetrics, FraudLog, AuditLog."],
    ["backend/app/db/crud.py", "Reusable create/read helpers over ORM models."],
    ["backend/app/api/schemas/prediction.py", "Request models with field validation."],
    ["backend/app/api/schemas/responses.py", "Structured response models."],
    ["backend/app/services/feature_engineering.py", "Rebuilds engineered features for one live request."],
    ["backend/app/services/business_rules.py", "Probability → risk score → level → action."],
    ["backend/app/services/prediction_service.py", "Singleton engine: loads model once, orchestrates the pipeline."],
    ["backend/app/services/explainability_service.py", "Global feature importance for the dashboard."],
    ["backend/app/api/routes/health.py", "GET / and GET /health."],
    ["backend/app/api/routes/predict.py", "POST /predict, /batch_predict with DB persistence."],
    ["backend/app/api/routes/metrics.py", "GET metrics, feature importance, history, model info."],
    ["backend/app/api/routes/retrain.py", "POST /retrain: regenerate + retrain + hot-reload."],
  ]),
  H2("Frontend & Infrastructure"),
  makeTable(["File", "Purpose"], [
    ["frontend/streamlit_app.py", "Login screen + home dashboard."],
    ["frontend/pages/1_Prediction.py", "Return-request input form and risk result display."],
    ["frontend/pages/2_History.py", "Filterable table of past predictions."],
    ["frontend/pages/3_Dashboard.py", "Risk distribution, action breakdown, timeline, histogram."],
    ["frontend/pages/4_Model_Metrics.py", "Comparison table, SHAP chart, training plots, retrain trigger."],
    ["frontend/utils/api_client.py", "requests-based wrapper for every backend endpoint."],
    ["frontend/utils/styling.py", "Shared dark-theme CSS."],
    ["docker/Dockerfile.backend / .frontend", "Container images for each service."],
    ["docker-compose.yml", "Orchestrates Postgres + backend + frontend."],
    ["render.yaml", "One-click Render.com deployment blueprint."],
    ["database/schema.sql", "Reference DDL matching the SQLAlchemy models."],
    ["tests/test_api.py", "Pytest suite: root, health, predict, validation, history, model info."],
  ]),
);

// ============================================================ 4
sections.push(
  H1("4. Key Function Explanations"),
  H3("ml/data/generate_dataset.py :: _compute_fraud_score(df, rng)"),
  P("The heart of realistic label generation. Builds a continuous 0-1 fraud score as a weighted sum of signals: a high personal return percentage adds risk, but is discounted whenever the seller's own defect rate plausibly explains the returns. Fast returns, new-account + high-value combos, missing tags, damage claims, and identity-cycling behaviour all add risk; loyalty segment and Gaussian noise pull the other way. The score is thresholded at the configured fraud_rate quantile to produce the binary label."),
  H3("ml/preprocessing/preprocessor.py :: build_feature_matrix(df, fit)"),
  P("Used both at training time (fit=True: fits and saves LabelEncoders + StandardScaler) and mirrored at inference (fit=False: loads and applies the same artifacts, mapping unseen categories to a safe default), so training and serving never drift apart."),
  H3("backend/app/services/feature_engineering.py :: engineer_features(request)"),
  P("Recomputes every engineered feature from a single live request using the exact same formulas as the offline dataset generator, guaranteeing training/serving parity."),
  H3("backend/app/services/business_rules.py :: evaluate_business_rules(fraud_probability)"),
  P("Linearly rescales probability into a 0-100 risk score, then classifies it against two configurable thresholds (default 35 / 70) into LOW / MEDIUM / HIGH, each mapped to a recommended action."),
  H3("backend/app/services/prediction_service.py :: PredictionEngine.predict_single(request)"),
  P("The orchestration function: engineer → prepare_input_matrix (encode + scale + reorder to match training) → predict_proba → evaluate_business_rules → explainer.explain_instance → assemble the result."),
  H3("ml/explainability/shap_explainer.py :: FraudExplainer.explain_instance(x_row, feature_names)"),
  P("Runs the fitted SHAP explainer on one row, extracts the top-K absolute SHAP values, tags each with a direction, and builds a plain-English explanation sentence."),
  H3("ml/training/train_deep_learning.py :: build_model(input_dim, cfg)"),
  P("Constructs the Keras DNN: Input → Dense(128, relu, L2) → BatchNorm → Dropout → Dense(64, relu, L2) → Dropout → Dense(32, relu, L2) → Dense(1, sigmoid). Compiled with Adam, binary cross-entropy, accuracy + AUC metrics."),
  H3("ml/training/model_comparison.py :: select_best_model(results)"),
  P("Selects the model with the highest ROC-AUC, tie-broken by F1 — ROC-AUC is threshold-independent and reflects overall ranking quality, while F1 rewards balanced precision/recall at the operating threshold actually used in production."),
);

// ============================================================ 5
sections.push(
  H1("5. Every Class Explanation"),
  makeTable(["Class", "Location", "Responsibility"], [
    ["Settings", "core/config.py", "Typed, cached singleton merging config.yaml + .env."],
    ["Customer / Seller", "db/models.py", "Reference entity tables for customer/seller master data."],
    ["PredictionHistory", "db/models.py", "One row per prediction with full explanation and input payload."],
    ["ModelMetrics", "db/models.py", "One row per trained model per run, with a best-model flag."],
    ["FraudLog", "db/models.py", "Denormalized log of HIGH-risk predictions for fast triage."],
    ["AuditLog", "db/models.py", "Generic actor/action/detail audit trail."],
    ["ReturnPredictionRequest", "schemas/prediction.py", "Validated input contract; enforces cross-field rules."],
    ["PredictionResponse", "schemas/responses.py", "Structured output contract."],
    ["PredictionEngine", "services/prediction_service.py", "Singleton holder for the loaded model + explainer."],
    ["FraudExplainer", "explainability/shap_explainer.py", "Dispatches to the correct SHAP explainer type per model."],
  ], [2200, 3000, 3800]),
);

// ============================================================ 6
sections.push(
  H1("6. Every API Endpoint Explanation"),
  makeTable(["Method & Path", "Description"], [
    ["GET /", "Root welcome message with API version and docs link."],
    ["GET /health", "Liveness/readiness: is the model loaded, is the DB reachable."],
    ["POST /api/v1/predict", "Score one return request end-to-end."],
    ["POST /api/v1/batch_predict", "Score up to 500 requests in one call."],
    ["POST /api/v1/retrain", "Optionally regenerate dataset, retrain all models, hot-swap serving model."],
    ["GET /api/v1/model_metrics", "Comparison table of every trained model's metrics."],
    ["GET /api/v1/feature_importance", "Global feature importance (native or SHAP-aggregate)."],
    ["GET /api/v1/prediction_history", "Paginated list of past predictions."],
    ["GET /api/v1/model_information", "Deployed model metadata: dataset size, fraud rate, all metrics."],
  ], [4500, 4500]),
  P("All request bodies are validated by Pydantic before reaching business logic; validation failures return HTTP 422 with a structured error. All endpoints are documented interactively at /docs and /redoc."),
);

// ============================================================ 7 ML PIPELINE
sections.push(
  H1("7. Machine Learning Pipeline"),
  P("The ML pipeline is a linear, reproducible sequence with a fixed random seed (42) at every stage:"),
  numbered("generate_dataset() — 12,000 synthetic records with realistic, rule-driven fraud labels (~18% base rate).", 1),
  numbered("build_feature_matrix() — label-encode 7 categorical columns, standard-scale 20 numerical columns, keep 8 binary flags as-is.", 2),
  numbered("train_test_split() — stratified 80/20 train/test, then a further stratified 85/15 split of the training portion into train/validation.", 3),
  numbered("Train 4 independent models in parallel code paths: Logistic Regression, Random Forest, XGBoost, Deep Neural Network.", 4),
  numbered("Evaluate each on the held-out test set: accuracy, precision, recall, F1, ROC-AUC.", 5),
  numbered("Select the best model by (ROC-AUC, F1) and persist its metadata for the serving layer.", 6),
  H2("Actual Comparison Results (this run)"),
  makeTable(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], [
    ["Logistic Regression", "0.8117", "0.4860", "0.8009", "0.6049", "0.8914"],
    ["Random Forest", "0.8646", "0.6524", "0.5301", "0.5849", "0.8966"],
    ["XGBoost (best)", "0.8738", "0.7009", "0.5208", "0.5976", "0.9092"],
    ["Deep Neural Network", "0.8750", "0.7102", "0.5162", "0.5979", "0.8992"],
  ]),
  P("XGBoost was automatically selected as the deployed model for this training run, based on the highest ROC-AUC (0.9092). Logistic Regression shows the classic high-recall/lower-precision baseline behaviour, useful as a sanity check that the problem is genuinely learnable and not accidentally leaked."),
);

// ============================================================ 8 DL ARCHITECTURE
sections.push(
  H1("8. Deep Learning Architecture"),
  P("The DNN is a fully-connected feed-forward network built with the Keras Functional API, chosen for its balance of capacity and regularization for a ~35-feature tabular problem:"),
  codeBlock(
`Input(35 features)
   │
Dense(128, activation='relu', kernel_regularizer=L2(0.001))
   │
BatchNormalization()
   │
Dropout(0.30)
   │
Dense(64, activation='relu', kernel_regularizer=L2(0.001))
   │
Dropout(0.30)
   │
Dense(32, activation='relu', kernel_regularizer=L2(0.001))
   │
Dense(1, activation='sigmoid')   -->  Fraud Probability`),
  P("Optimizer: Adam (lr=0.001). Loss: Binary Cross-Entropy. Metrics: Accuracy, AUC. Batch size: 64. Max epochs: 100, governed by EarlyStopping."),
  H2("Callbacks"),
  makeTable(["Callback", "Configuration", "Purpose"], [
    ["EarlyStopping", "monitor=val_loss, patience=10, restore_best_weights=True", "Stops training once validation loss stops improving, restoring the best-epoch weights."],
    ["ModelCheckpoint", "monitor=val_loss, save_best_only=True", "Persists only the best-validation-loss model to disk."],
    ["ReduceLROnPlateau", "factor=0.5, patience=5, min_lr=1e-6", "Halves the learning rate when validation loss plateaus, helping the model settle into a better minimum."],
  ], [2500, 3500, 3000]),
);

// ============================================================ 9 NN LAYER BY LAYER
sections.push(
  H1("9. Neural Network Layer-by-Layer Explanation"),
  makeTable(["Layer", "Output Shape", "Parameters", "Why it's here"], [
    ["Input", "(None, 35)", "0", "One neuron per engineered feature (20 numeric + 7 categorical-encoded + 8 binary flags)."],
    ["Dense(128, relu)", "(None, 128)", "4,608", "First representation layer; ReLU avoids vanishing gradients and is cheap to compute. L2 regularization discourages over-reliance on any single feature."],
    ["BatchNormalization", "(None, 128)", "512", "Normalizes activations between layers, stabilizing and speeding up training, and providing mild regularization."],
    ["Dropout(0.30)", "(None, 128)", "0", "Randomly zeroes 30% of activations during training to prevent co-adaptation and overfitting on a moderately-sized dataset."],
    ["Dense(64, relu)", "(None, 64)", "8,256", "Compresses the representation, forcing the network to learn more abstract combinations of the risk signals."],
    ["Dropout(0.30)", "(None, 64)", "0", "Second regularization pass before the final decision layers."],
    ["Dense(32, relu)", "(None, 32)", "2,080", "Final feature-combination layer immediately before the output."],
    ["Dense(1, sigmoid)", "(None, 1)", "33", "Squashes the final linear combination into a [0,1] fraud probability, matching the binary cross-entropy loss."],
  ], [2200, 2000, 1800, 3000]),
  P("Sigmoid is used (rather than softmax over 2 classes) because this is a single-output binary classification — sigmoid directly yields the fraud-class probability, which the Business Rule Engine then maps to a 0-100 risk score."),
);

// ============================================================ 10 DATA FLOW
sections.push(
  H1("10. Data Flow"),
  P("Two distinct data flows exist in the system: the OFFLINE (training) flow and the ONLINE (serving) flow. Keeping their feature-engineering logic numerically identical is the single most important correctness property of the whole system."),
  H2("Offline Flow"),
  codeBlock(`CSV generation (rules) -> raw_dataset.csv
   -> label encode + scale (fit=True, artifacts saved)
   -> train/val/test split
   -> 4x model.fit()
   -> metrics + plots -> model_comparison_report.json
   -> best_model_metadata.json`),
  H2("Online (Serving) Flow"),
  codeBlock(`HTTP JSON request -> Pydantic validation
   -> engineer_features() [same formulas as offline]
   -> label encode + scale (fit=False, saved artifacts loaded)
   -> model.predict_proba()
   -> SHAP explain_instance()
   -> business rules -> PredictionResponse
   -> persisted to prediction_history table`),
);

// ============================================================ 11 FEATURE ENGINEERING WORKFLOW
sections.push(
  H1("11. Feature Engineering Workflow"),
  P("20 numerical, 7 categorical, and 8 binary-flag features feed the models. The engineered (derived) features and their formulas:"),
  makeTable(["Feature", "Formula / Logic"], [
    ["return_percentage", "(total_returns / total_orders) * 100"],
    ["is_luxury_product", "product_price > 250"],
    ["is_high_value_order", "order_value above the 85th percentile (training) / > $250 (serving)"],
    ["price_category", "Budget ≤30, Mid-Range ≤100, Premium ≤300, else Luxury"],
    ["seller_reliability_score", "(seller_rating/5) * (1 - seller_defect_rate) * 100"],
    ["delivery_risk_score", "(delivery_days/21)*60 + (distance_km/max_distance)*40"],
    ["customer_lifetime_value", "total_orders * order_value"],
    ["is_seasonal_return", "is_holiday_purchase AND avg_days_to_return < 20"],
    ["customer_risk_score", "45%·(return% normalized) + 35%·(fast-return factor) + 20%·(new-account factor)"],
  ]),
  P("These formulas are implemented once in ml/data/generate_dataset.py (offline) and re-implemented with identical arithmetic in backend/app/services/feature_engineering.py (online), since a live request has no access to a historical batch distribution for percentile-based features."),
);

// ============================================================ 12 TRAINING WORKFLOW
sections.push(
  H1("12. Training Workflow"),
  numbered("Run `python -m ml.data.generate_dataset` to (re)create the raw CSV.", 1),
  numbered("Run `python -m ml.training.model_comparison`, which internally calls run_preprocessing() once per model script (idempotent — same seed, same split).", 2),
  numbered("Each training script fits its model, computes test-set metrics, saves confusion matrix / ROC / PR plots, and writes its metrics into model_comparison_report.json.", 3),
  numbered("model_comparison.py aggregates all results into a single comparison table, printed to console and saved.", 4),
  numbered("The best model (by ROC-AUC, F1 tiebreak) is written to best_model_metadata.json, which the serving layer reads on startup.", 5),
  numbered("Metrics are also pushed into the model_metrics database table for the /model_metrics API and the Streamlit dashboard.", 6),
);

// ============================================================ 13 PREDICTION WORKFLOW
sections.push(
  H1("13. Prediction Workflow"),
  numbered("Client sends a JSON payload to POST /api/v1/predict.", 1),
  numbered("FastAPI validates it against ReturnPredictionRequest (range checks + total_returns ≤ total_orders).", 2),
  numbered("PredictionEngine.predict_single() engineers features, applies the saved encoders/scaler, and orders columns to match the training-time feature_columns.json.", 3),
  numbered("The best model's predict_proba (or .predict() for the Keras DNN) returns a fraud probability.", 4),
  numbered("FraudExplainer.explain_instance() computes the top-5 SHAP contributors and a business explanation sentence.", 5),
  numbered("business_rules.evaluate_business_rules() converts probability to a 0-100 risk score, a LOW/MEDIUM/HIGH level, and a recommended action.", 6),
  numbered("The full result is persisted to prediction_history (and fraud_logs if HIGH risk), audit-logged, and returned as a PredictionResponse.", 7),
);

// ============================================================ 14 DEPLOYMENT WORKFLOW
sections.push(
  H1("14. Deployment Workflow"),
  P("Three deployment paths are supported out of the box:"),
  H2("A. Local (no containers)"),
  codeBlock(`pip install -r requirements.txt
python -m ml.data.generate_dataset
python -m ml.training.model_comparison
uvicorn backend.app.main:app --reload
streamlit run frontend/streamlit_app.py`),
  H2("B. Docker Compose (single machine, Postgres included)"),
  codeBlock(`docker compose up --build
docker compose exec backend python -m ml.data.generate_dataset
docker compose exec backend python -m ml.training.model_comparison
docker compose restart backend`),
  H2("C. Render.com (managed cloud, via render.yaml Blueprint)"),
  P("Push the repository to GitHub, then in the Render dashboard choose New → Blueprint and point it at the repo. Render provisions a managed Postgres instance plus two web services (backend, frontend) wired together automatically via environment variables."),
);

// ============================================================ 15 DOCKER WORKFLOW
sections.push(
  H1("15. Docker Workflow"),
  P("Two separate images keep concerns isolated and let each service scale independently:"),
  makeTable(["Image", "Base", "Exposes", "Healthcheck"], [
    ["returnguard-backend", "python:3.11-slim + build-essential/libgomp1 (for xgboost/shap)", "8000", "curl http://localhost:8000/health"],
    ["returnguard-frontend", "python:3.11-slim", "8501", "curl http://localhost:8501/_stcore/health"],
  ], [3200, 3800, 1200, 3000].slice(0,4)),
  P("docker-compose.yml adds a third service, a managed postgres:16-alpine container, and wires DATABASE_URL into the backend automatically. Named volumes (returnguard_models, returnguard_pg_data, returnguard_logs) persist trained models, database data, and logs across container restarts."),
);

// ============================================================ 16 API WORKFLOW (interaction diagram context)
sections.push(
  H1("16. API Workflow"),
  P("FastAPI's dependency-injection system supplies a fresh SQLAlchemy session (get_db) to every route that needs one, automatically closed after the request. Routes stay thin — they call into services/ for all business logic, which keeps routes easily testable and the logic reusable (e.g. the same PredictionEngine.predict_single() powers both /predict and /batch_predict)."),
  P("CORS is restricted via config.yaml's api.cors_origins list (defaults to the local Streamlit origin) rather than left wide open, and the interactive OpenAPI schema is auto-generated from the Pydantic models, so /docs is always accurate."),
);

// ============================================================ 17 STREAMLIT WORKFLOW
sections.push(
  H1("17. Streamlit Workflow"),
  P("The frontend uses Streamlit's native multipage convention: streamlit_app.py is the entrypoint (login + home), and every file under frontend/pages/ automatically becomes a sidebar navigation item. Authentication state lives in st.session_state and gates every page (each page checks session_state['authenticated'] and stops early if absent)."),
  P("All backend communication goes through frontend/utils/api_client.py, so pages never construct raw HTTP calls, and a single ApiError exception type is used for uniform error handling and messaging across every page."),
);

// ============================================================ 18 DATABASE WORKFLOW
sections.push(
  H1("18. Database Workflow"),
  P("SQLAlchemy's declarative Base plus Base.metadata.create_all() creates all six tables on first startup (idempotent — safe to call every boot). The DB driver is chosen automatically: if the DATABASE_URL environment variable is set, Postgres is used (production/Docker/Render); otherwise a local SQLite file at database/returnguard.db is created (zero-config local development)."),
  P("Every prediction is persisted with its full input payload and SHAP top-features as JSON columns, so any historical prediction can be fully reconstructed and re-explained without re-running the model."),
);

// ============================================================ 19 SEQUENCE DIAGRAM
sections.push(
  H1("19. Sequence Diagram"),
  image(`${DOCS}/sequence_diagram.png`, 580, 370),
  caption("Figure 1 — End-to-end prediction sequence from the Streamlit form to database persistence."),
);

// ============================================================ 20 ARCHITECTURE DIAGRAM
sections.push(
  H1("20. Architecture Diagram"),
  image(`${DOCS}/architecture_diagram.png`, 430, 570),
  caption("Figure 2 — Layered system architecture, from the customer down to persistence."),
);

// ============================================================ TRAINING PLOTS APPENDIX
sections.push(
  pageBreak(),
  H1("Appendix A — Model Evaluation Plots"),
  H2("Confusion Matrix — XGBoost (Deployed Model)"),
  image(`${PLOTS}/confusion_matrix_xgboost.png`, 320, 260),
  H2("ROC Curve — XGBoost (Deployed Model)"),
  image(`${PLOTS}/roc_curve_xgboost.png`, 320, 260),
  H2("Deep Neural Network — Training vs Validation Accuracy"),
  image(`${PLOTS}/training_accuracy_curve.png`, 380, 260),
  H2("Deep Neural Network — Training vs Validation Loss"),
  image(`${PLOTS}/training_loss_curve.png`, 380, 260),
);

// ============================================================ 21 INTERVIEW Q&A
sections.push(
  pageBreak(),
  H1("21. Interview Questions and Answers"),
  H3("Q1. Why compare four different models instead of just using the best-known one (e.g. XGBoost)?"),
  P("A: Model performance is dataset-dependent, and a transparent comparison builds trust with stakeholders and satisfies model-risk-management expectations common in fintech/e-commerce fraud teams. It also gives a natural fallback: if XGBoost's dependencies fail in an environment, the pipeline degrades gracefully to Random Forest or Logistic Regression rather than failing outright."),
  H3("Q2. How do you prevent training/serving skew in this system?"),
  P("A: The exact same engineered-feature formulas exist in both ml/data/generate_dataset.py (offline) and backend/app/services/feature_engineering.py (online), and the same fitted LabelEncoders/StandardScaler artifacts (saved by preprocessor.py) are loaded at inference time rather than re-fit, guaranteeing numerical parity between training and serving."),
  H3("Q3. Why SHAP instead of simpler feature-importance methods?"),
  P("A: SHAP gives per-prediction, signed (direction-aware) attributions grounded in cooperative game theory, which is what's needed to explain an individual decision to a fraud analyst ('why was THIS return flagged'), not just a global ranking of which features matter on average."),
  H3("Q4. Why is XGBoost currently the best model here, and could that change?"),
  P("A: In this run XGBoost achieved the highest ROC-AUC (0.9092), edging out the Deep Neural Network (0.8992) and Random Forest (0.8966). On a moderately-sized tabular dataset (12k rows, 35 features) with a mix of continuous and categorical features, gradient-boosted trees very often outperform DNNs unless the DNN is given substantially more data or engineered embeddings; that gap could close (or reverse) with more data, better hyperparameter tuning, or entity embeddings for categoricals."),
  H3("Q5. How does the Business Rule Engine avoid being a black box on top of a black box?"),
  P("A: It's a small, deterministic, fully auditable function (business_rules.py) with two configuration-driven thresholds — nothing about it is learned or opaque. Its only input is the model's fraud probability; everything else (score scaling, thresholds, action mapping) is explicit and can be reviewed line-by-line by a compliance team."),
  H3("Q6. How would you scale /predict to handle thousands of requests per second?"),
  P("A: The PredictionEngine already loads the model once as a process-level singleton (not per-request), so the main scaling lever is horizontal: run multiple stateless uvicorn workers behind a load balancer, move SHAP computation to a fast-path approximation or precomputed background samples, and consider a feature store to avoid re-deriving features that don't change often (e.g. seller reliability)."),
  H3("Q7. What is the tradeoff of making /retrain synchronous?"),
  P("A: Simplicity and transparency — the caller gets the final comparison table directly — at the cost of blocking the HTTP connection for the duration of training. In a real production system this would be moved to a background job queue (Celery/RQ) with a job-status endpoint, which is documented as a known limitation."),
  H3("Q8. Why use both a risk_score (0-100) and a risk_level (LOW/MEDIUM/HIGH)?"),
  P("A: The continuous score preserves information for analytics/trending (e.g. \"our average risk score rose 4 points this month\"), while the discrete level drives simple, auditable business action mapping that non-technical stakeholders can reason about directly."),
);

// ============================================================ 22 VIVA QUESTIONS
sections.push(
  H1("22. Viva Questions"),
  makeTable(["#", "Question"], [
    ["1", "What is the difference between fraud_probability and risk_score in this system?"],
    ["2", "Why does total_returns have a validator against total_orders in the Pydantic schema?"],
    ["3", "What happens if a categorical value at inference time was never seen during training?"],
    ["4", "Why is BatchNormalization placed after the first Dense layer but not the others?"],
    ["5", "What does the EarlyStopping callback's restore_best_weights=True actually do?"],
    ["6", "Why does the dataset generator subtract risk when seller_defect_rate is high?"],
    ["7", "What is the purpose of the label_encoders.joblib and scaler.joblib artifacts?"],
    ["8", "How is the 'best model' selected, and why ROC-AUC over plain accuracy?"],
    ["9", "What does the FraudLog table capture that PredictionHistory does not?"],
    ["10", "Why does the API use a singleton PredictionEngine instead of loading the model per request?"],
    ["11", "What SQL database is used by default, and how do you switch to PostgreSQL?"],
    ["12", "How does Streamlit know which files become sidebar pages?"],
  ]),
);

// ============================================================ 23 FUTURE ENHANCEMENTS
sections.push(
  H1("23. Future Enhancements"),
  bullet("Move /retrain to an asynchronous background job queue (Celery/RQ) with a job-status polling endpoint."),
  bullet("Add entity embeddings for high-cardinality categoricals to give the DNN a fairer chance against gradient boosting."),
  bullet("Introduce a feature store to cache slowly-changing features (seller reliability, customer lifetime value) instead of recomputing per request."),
  bullet("Add model monitoring for data/concept drift (e.g. population stability index on incoming feature distributions)."),
  bullet("Add role-based access control (analyst vs admin) instead of the current single shared admin login."),
  bullet("A/B test risk thresholds against real business outcomes (chargeback rates, customer satisfaction) before changing them."),
  bullet("Add a human-in-the-loop feedback endpoint so investigator outcomes (confirmed fraud / false positive) can be fed back into future retraining."),
);

// ============================================================ 24 LIMITATIONS
sections.push(
  H1("24. Limitations"),
  bullet("The dataset is synthetic; while the fraud-generating rules are realistic and include deliberate edge cases, real-world fraud patterns are more varied and adversarial (fraudsters adapt to detection)."),
  bullet("Percentile-based features (e.g. is_high_value_order) use a fixed dollar threshold at inference time since a single live request has no access to the current batch distribution — this is an approximation of the training-time percentile."),
  bullet("/retrain runs synchronously and will block the calling connection for the duration of training, which is acceptable for a demo/project but not for a high-traffic production API."),
  bullet("SHAP's KernelExplainer fallback path (used for non-tree models) is sampling-based and slower than TreeExplainer; latency-sensitive deployments should prefer tree-based models for the explainability step."),
  bullet("Authentication is a single shared admin account suitable for a demo, not a multi-analyst production access-control model."),
);

// ============================================================ 25 BUSINESS IMPACT
sections.push(
  H1("25. Business Impact"),
  P("A working return-fraud triage system like ReturnGuard AI targets three concrete business levers:"),
  bullet("Cost reduction: automatically auto-approving LOW risk returns (the majority case) removes manual review overhead from the bulk of genuine returns."),
  bullet("Loss prevention: routing HIGH risk returns to Investigation Required focuses scarce fraud-analyst time on the cases most likely to be fraudulent, rather than spreading review effort evenly."),
  bullet("Customer experience: genuine customers with a plausible explanation (e.g. return caused by a high seller defect rate) are not penalized for a pattern that isn't actually their fault — the model was explicitly designed to learn this distinction."),
  P("The explainability layer additionally supports compliance and dispute-resolution workflows: any flagged return comes with a specific, auditable reason rather than an opaque score, which shortens investigation time and supports customer communication when a decision is challenged."),
);

// ============================================================ 26 CONCLUSION
sections.push(
  H1("26. Conclusion"),
  P("ReturnGuard AI demonstrates a complete, realistic ML engineering workflow: a business-rule-driven synthetic dataset with deliberately hard edge cases; four competing model families trained and compared on identical splits; a deep neural network built and regularized following current best practice (BatchNorm, Dropout, L2, early stopping, LR scheduling); SHAP-based per-prediction explainability; a validated, documented REST API; a persistent audit trail in a relational database; a dark-themed multipage analytics dashboard; and full containerized deployment tooling for both local Docker Compose and managed cloud (Render) environments."),
  P("Every component in this document was implemented and exercised end-to-end during development — the dataset was generated, all four models were trained (with the comparison table and plots in Appendix A produced from that actual run), the API was booted and called live, and the full test suite passed — rather than being a theoretical design on paper."),
);

// ============================================================ BUILD DOCX
const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children: sections,
  }],
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
    },
  },
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(`${DOCS}/ReturnGuard_AI_Documentation.docx`, buffer);
  console.log("Documentation generated successfully.");
});
