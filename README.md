# 🛡️ ReturnGuard AI

**Advanced AI Powered Return Fraud Detection & Risk Intelligence System**

ReturnGuard AI predicts the probability that an e-commerce return request is fraudulent, converts that probability into a 0–100 **risk score**, explains **why** with SHAP, and recommends a **business action** (Auto Approve / Manual Review / Investigation Required).

---

## Architecture

```
Customer → Streamlit Frontend → FastAPI REST API → Validation (Pydantic)
    → Feature Engineering → Preprocessing (encode + scale)
    → Model (LogReg / RandomForest / XGBoost / Deep Neural Network)
    → SHAP Explainability → Business Rule Engine → Response
    → PostgreSQL/SQLite (history, metrics, audit) → Streamlit Dashboard
```

---

## Project Structure

```
returnguard-ai/
├── backend/app/
│   ├── main.py                 FastAPI entrypoint
│   ├── api/routes/              health, predict, metrics, retrain
│   ├── api/schemas/              Pydantic request/response models
│   ├── core/                    config, logging, security (JWT/bcrypt)
│   ├── db/                      SQLAlchemy models, database session, CRUD
│   └── services/                 feature engineering, business rules, prediction engine
├── ml/
│   ├── data/generate_dataset.py         synthetic dataset generator (12k rows)
│   ├── preprocessing/preprocessor.py     encode/scale/split pipeline
│   ├── training/                        4 model training scripts + comparison
│   ├── explainability/shap_explainer.py  SHAP wrapper
│   └── models/                          saved model artifacts + plots
├── frontend/
│   ├── streamlit_app.py         login + home dashboard
│   ├── pages/                    Prediction, History, Dashboard, Model Metrics
│   └── utils/                    API client, dark theme styling
├── database/schema.sql          reference SQL schema
├── docker/                      Dockerfile.backend, Dockerfile.frontend
├── docker-compose.yml           full stack orchestration (Postgres + backend + frontend)
├── render.yaml                  Render.com deployment blueprint
├── config/config.yaml            all non-secret configuration
├── config/.env.example           secrets template
├── docs/                        full project documentation (.docx)
└── requirements.txt
```

---

## Quick Start (Local, no Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment template
cp config/.env.example .env

# 3. Generate the synthetic dataset (~12,000 records)
python -m ml.data.generate_dataset

# 4. Train all 4 models + pick the best one
python -m ml.training.model_comparison

# 5. Start the backend API
uvicorn backend.app.main:app --reload --port 8000
# -> Swagger docs at http://localhost:8000/docs

# 6. In a second terminal, start the frontend
streamlit run frontend/streamlit_app.py
# -> http://localhost:8501  (login: admin / admin123)
```

---

## Quick Start (Docker)

```bash
docker compose up --build
```

- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501
- Postgres: localhost:5432 (returnguard_user / returnguard_pass / returnguard_db)

The backend container trains no models by itself on first boot — run once:

```bash
docker compose exec backend python -m ml.data.generate_dataset
docker compose exec backend python -m ml.training.model_comparison
docker compose restart backend
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root info |
| GET | `/health` | Liveness/readiness probe |
| POST | `/api/v1/predict` | Single return fraud prediction |
| POST | `/api/v1/batch_predict` | Batch prediction (up to 500) |
| POST | `/api/v1/retrain` | Retrain all models, hot-swap best model |
| GET | `/api/v1/model_metrics` | Comparison table across all trained models |
| GET | `/api/v1/feature_importance` | Global SHAP-based feature importance |
| GET | `/api/v1/prediction_history` | Paginated past predictions |
| GET | `/api/v1/model_information` | Deployed model metadata |

Full interactive documentation: `/docs` (Swagger) and `/redoc`.

---

## Machine Learning

Four models are trained and compared automatically:

| Model | Library | Notes |
|---|---|---|
| Logistic Regression | scikit-learn | Interpretable baseline |
| Random Forest | scikit-learn | Ensemble, handles non-linearity |
| XGBoost | xgboost | Gradient boosting, typically best ROC-AUC |
| Deep Neural Network | TensorFlow/Keras | Dense(128)→BatchNorm→Dropout→Dense(64)→Dropout→Dense(32)→Sigmoid |

The best model (by ROC-AUC, tie-broken by F1) is auto-selected and served by the API. Metrics, confusion matrices, ROC curves, precision-recall curves, and (for the DNN) training/validation accuracy & loss curves are all saved to `ml/models/plots/`.

### Realistic fraud labeling

Fraud labels are **not random** — they come from a weighted business-rule scoring function in `ml/data/generate_dataset.py`, including the deliberate edge case: a customer with a high personal return rate whose seller has a high defect rate nets out to **lower** fraud risk, teaching the model that not all high-return customers are fraudulent.

---

## Explainability

Every prediction includes:
- Top-5 contributing features (SHAP values, direction of effect)
- A plain-English business explanation string
- The exact model used

---

## Retraining

```bash
curl -X POST http://localhost:8000/api/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{"regenerate_dataset": false}'
```

Or use the **Retrain Models** panel on the Model Metrics page in the Streamlit UI.

---

## Environment Variables

See `config/.env.example` for the full list (database URL, JWT secret, admin credentials, ports). Copy it to `.env` before running locally.

---

## Tech Stack

FastAPI · Pydantic · SQLAlchemy · PostgreSQL/SQLite · scikit-learn · XGBoost · TensorFlow/Keras · SHAP · Streamlit · Plotly · Docker · loguru
