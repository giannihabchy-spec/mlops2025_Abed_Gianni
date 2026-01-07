# ML Project

A machine learning project for trip duration prediction with preprocessing, feature engineering, training, and inference pipelines.

## Project Structure

```
mlops2025_Abed_Gianni/
├── src/
│   └── mlproject/
│       ├── data/              # Place train.csv and test.csv here
│       ├── features/          # Feature engineering modules
│       ├── inference/         # Inference pipeline
│       ├── preprocess/        # Data preprocessing modules
│       ├── train/             # Training pipeline
│       └── utils/             # Utility functions
├── scripts/
│   ├── batch_inference.py    
│   ├── feature_engineering.py
│   ├── preprocess.py
│   └── train.py
├── notebooks/                 # Jupyter notebooks for exploration
├── outputs/                   # Model artifacts and predictions
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Data Setup

**Important**: Before running the project, place your data files in the following location:
- `src/mlproject/data/train.csv` - Training data
- `src/mlproject/data/test.csv` - Test data

## How to Run Locally

### Prerequisites
- Python 3.11
- `uv` package manager (install with: `pip install uv`)

### Steps

1. **Install dependencies and set up the environment:**
   ```bash
   uv venv
   uv sync
   ```

2. **Install the package:**
   ```bash
   uv pip install -e .
   ```

3. **Run training:**
   ```bash
   uv run train
   ```
   This will:
   - Preprocess the raw data
   - Generate features
   - Train multiple models and select the best one
   - Save model artifacts to `outputs/model_artifacts/`

4. **Run inference:**
   ```bash
   uv run inference
   ```
   This will:
   - Preprocess test data
   - Generate features
   - Load the best model and make predictions
   - Save predictions to `outputs/YYYYMMDD_predictions.csv`

## How to Run with Docker

1. **Build and run training:**
   ```bash
   docker-compose run train
   ```

2. **Build and run inference:**
   ```bash
   docker-compose run inference
   ```

## Selected Metric(s) and Justification

**Metric: RMSE on log1p-transformed target (RMSE_log1p)**

**Justification:**
- The target variable (trip_duration) has a heavy-tailed distribution with potential outliers
- Using log1p transformation (log(1 + x)) stabilizes the distribution and reduces the impact of extreme values
- RMSE (Root Mean Squared Error) is appropriate for regression tasks as it penalizes larger errors more heavily
- Training on the transformed target and then exponentiating predictions (expm1) ensures predictions are in the original scale while benefiting from the stabilized training process

## Model Choices

The project evaluates five different regression models and automatically selects the best performing one:

1. **Ridge Regression** - L2 regularization, good for handling multicollinearity
2. **Lasso Regression** - L1 regularization, performs feature selection
3. **ElasticNet** - Combines L1 and L2 regularization, balances feature selection and multicollinearity handling
4. **Random Forest Regressor** - Ensemble method, captures non-linear relationships and interactions
5. **Histogram-based Gradient Boosting Regressor** - Efficient gradient boosting, handles large datasets well

**Model Selection:**
- All models are trained on the same preprocessed and feature-engineered data
- Performance is evaluated using RMSE on the validation set
- The model with the lowest RMSE_log1p is automatically selected and saved
- Model comparison results are saved to `outputs/model_artifacts/results.csv`

