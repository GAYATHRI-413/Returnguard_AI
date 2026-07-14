-- ==============================================================================
-- ReturnGuard AI - database/schema.sql
--
-- Reference SQL schema (PostgreSQL dialect). This mirrors exactly what
-- SQLAlchemy's Base.metadata.create_all() generates from
-- backend/app/db/models.py, provided here for DBA review / manual
-- provisioning / documentation purposes.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS customers (
    id                          SERIAL PRIMARY KEY,
    customer_code               VARCHAR(64) UNIQUE NOT NULL,
    name                        VARCHAR(128),
    age                         INTEGER,
    account_age_days            INTEGER,
    total_orders                INTEGER DEFAULT 0,
    total_returns               INTEGER DEFAULT 0,
    customer_lifetime_value     FLOAT DEFAULT 0.0,
    customer_segment            VARCHAR(64),
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_customers_customer_code ON customers (customer_code);

CREATE TABLE IF NOT EXISTS sellers (
    id                  SERIAL PRIMARY KEY,
    seller_code         VARCHAR(64) UNIQUE NOT NULL,
    name                VARCHAR(128),
    rating              FLOAT DEFAULT 0.0,
    defect_rate         FLOAT DEFAULT 0.0,
    reliability_score   FLOAT DEFAULT 0.0,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_sellers_seller_code ON sellers (seller_code);

CREATE TABLE IF NOT EXISTS prediction_history (
    id                      SERIAL PRIMARY KEY,
    request_id              VARCHAR(64) NOT NULL,
    customer_code           VARCHAR(64),
    seller_code             VARCHAR(64),
    fraud_probability       FLOAT NOT NULL,
    risk_score              FLOAT NOT NULL,
    risk_level              VARCHAR(32) NOT NULL,
    recommended_action      VARCHAR(64) NOT NULL,
    model_used              VARCHAR(64) NOT NULL,
    top_features_json       TEXT,
    business_explanation    TEXT,
    input_payload_json      TEXT,
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_prediction_history_request_id ON prediction_history (request_id);
CREATE INDEX IF NOT EXISTS ix_prediction_history_customer_code ON prediction_history (customer_code);
CREATE INDEX IF NOT EXISTS ix_prediction_history_seller_code ON prediction_history (seller_code);
CREATE INDEX IF NOT EXISTS ix_prediction_history_created_at ON prediction_history (created_at);

CREATE TABLE IF NOT EXISTS model_metrics (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(64) NOT NULL,
    accuracy        FLOAT,
    precision       FLOAT,
    recall          FLOAT,
    f1_score        FLOAT,
    roc_auc         FLOAT,
    is_best_model   BOOLEAN DEFAULT FALSE,
    trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fraud_logs (
    id                  SERIAL PRIMARY KEY,
    prediction_id       INTEGER,
    customer_code       VARCHAR(64),
    risk_level          VARCHAR(32) NOT NULL,
    reason_summary      TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_fraud_logs_customer_code ON fraud_logs (customer_code);

CREATE TABLE IF NOT EXISTS audit_logs (
    id          SERIAL PRIMARY KEY,
    actor       VARCHAR(64),
    action      VARCHAR(128) NOT NULL,
    detail      TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_audit_logs_created_at ON audit_logs (created_at);
