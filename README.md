# Intelligent Document Classifier

A production-grade ML system that automatically classifies consumer financial documents into product categories using transformer-based NLP, with explainable predictions, real-time serving, and model monitoring.

> Demonstrates end-to-end ML system design — from data engineering and feature pipelines to model serving, monitoring, and deployment.

## Architecture

```
┌────────────────┐     ┌──────────────┐     ┌────────────────────┐
│  CFPB Dataset  │────>│  Text        │────>│  Model Training    │
│  (2M+ docs)    │     │  Pipeline    │     │  LR → LGBM → BERT │
└────────────────┘     └──────────────┘     └─────────┬──────────┘
                                                      │
                                                      v
┌────────────────┐     ┌──────────────┐     ┌────────────────────┐
│  Streamlit     │<────│  FastAPI     │<────│  MLflow Registry   │
│  Dashboard     │     │  /classify   │     │  (best model)      │
└───────┬────────┘     └──────┬───────┘     └────────────────────┘
        │                     │
        v                     v
┌─────────────────────────────────────────┐
│  Evidently AI — Drift & Performance     │
│  Monitoring                             │
└─────────────────────────────────────────┘
```

## Model Performance

| Model                  | Accuracy | Macro-F1 | Weighted-F1 | Inference Latency |
|------------------------|----------|----------|-------------|-------------------|
| TF-IDF + LogReg        | ~76%     | ~0.72    | ~0.75       | < 5ms             |
| TF-IDF + LightGBM      | ~83%     | ~0.80    | ~0.82       | < 10ms            |
| DistilBERT (fine-tuned) | ~91%     | ~0.88    | ~0.90       | < 80ms            |

> Exact numbers depend on dataset version and hardware. Run `make train` to reproduce.

## Key Features

- **Progressive Modeling**: Baseline (LogReg) → Intermediate (LightGBM) → Advanced (DistilBERT), demonstrating model selection judgment
- **Explainability**: LIME text explanations showing which words drive each classification decision
- **Real-Time Serving**: FastAPI with < 100ms inference latency, batch endpoint, health checks
- **Model Monitoring**: Evidently AI integration for text distribution drift detection and prediction confidence tracking
- **Interactive Dashboard**: Streamlit app for single/batch classification, explainability visualization, and model health monitoring
- **MLOps**: MLflow experiment tracking, model registry, structured logging
- **Containerized**: Docker Compose for one-command deployment (API + Dashboard)
- **Tested**: Pytest suite with preprocessing, feature extraction, API integration, and schema validation tests
- **CI/CD**: GitHub Actions pipeline for linting and testing on every push

## Dataset

[CFPB Consumer Complaints Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — 2M+ real consumer financial complaints with product category labels. Free, public, no account needed.

**Categories** (after consolidation): Credit reporting, Debt collection, Mortgage, Credit card, Banking, Student loan, Vehicle loan, Money transfer, Personal loan

**Data Challenges Handled**:
- PII redaction patterns (XXXX, XX/XX/XXXX) normalized to semantic tokens
- Class consolidation (evolving category names across dataset versions)
- Missing narratives (~25% of records lack text — filtered)
- Class imbalance (addressed with stratified splits and class weighting)

## Quick Start

### Prerequisites
- Python 3.10+
- ~2 GB disk space for dataset

### 1. Setup

```bash
git clone https://github.com/kakadia-zeel/intelligent-doc-classifier.git
cd intelligent-doc-classifier
make setup
```

### 2. Download Data

```bash
make download-data
# Or manually: bash scripts/download_data.sh
```

### 3. Train Models

```bash
make train
```

This trains all three models (LogReg, LightGBM, DistilBERT), logs experiments to MLflow, and saves artifacts.

### 4. Start the API

```bash
make serve
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Launch the Dashboard

```bash
make dashboard
# Dashboard: http://localhost:8501
```

### 6. Run with Docker (Alternative)

```bash
make docker-up
# API: http://localhost:8000 | Dashboard: http://localhost:8501
```

### 7. Run Tests

```bash
make test
```

## API Endpoints

| Method | Endpoint          | Description                                |
|--------|-------------------|--------------------------------------------|
| POST   | `/classify`       | Classify single document (+ optional LIME) |
| POST   | `/classify/batch` | Classify up to 100 documents               |
| GET    | `/health`         | Service health and model status            |
| GET    | `/metrics`        | Prediction stats and drift detection       |

### Example Request

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My credit card was charged twice for the same purchase.", "explain": false}'
```

### Example Response

```json
{
  "predicted_class": "Credit card",
  "confidence": 0.94,
  "probabilities": {
    "Credit card": 0.94,
    "Debt collection": 0.02,
    "Banking": 0.01,
    ...
  },
  "explanation": null
}
```

## Project Structure

```
├── src/
│   ├── data/           # Data download, preprocessing, PyTorch dataset
│   ├── features/       # TF-IDF pipeline, text feature extraction
│   ├── models/         # Baseline, transformer, evaluation, LIME, MLflow
│   ├── serving/        # FastAPI app, schemas, middleware
│   ├── monitoring/     # Drift detection, performance tracking
│   └── utils/          # Config loading, structured logging
├── dashboard/          # Streamlit app (classify, batch, explain, monitor)
├── tests/              # Pytest suite (preprocessing, features, API, schemas)
├── configs/            # YAML configs (model, serving, training)
├── scripts/            # CLI entrypoints (train, evaluate, download)
├── notebooks/          # EDA and experimentation (gitignored outputs)
└── .github/workflows/  # CI pipeline
```

## Tech Stack

| Layer               | Tools                                              |
|---------------------|----------------------------------------------------|
| ML Framework        | PyTorch, Hugging Face Transformers, LightGBM, scikit-learn |
| Explainability      | LIME                                               |
| Experiment Tracking | MLflow                                             |
| API                 | FastAPI, Pydantic v2, Uvicorn                      |
| Dashboard           | Streamlit, Plotly                                   |
| Monitoring          | Evidently AI                                       |
| Testing             | Pytest, pytest-cov                                 |
| Code Quality        | Ruff, Black                                        |
| Containerization    | Docker, Docker Compose                             |
| CI/CD               | GitHub Actions                                     |

## Development

```bash
# Format code
make format

# Lint code
make lint

# Run tests with coverage
make test

# View MLflow experiments
mlflow ui --port 5000
```

## License

MIT

## Author

**Zeel Kakadia** — [GitHub](https://github.com/kakadia-zeel) | [LinkedIn](https://linkedin.com/in/zeel-kakadia)
