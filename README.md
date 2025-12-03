# AutoML Model Selector

A Flask-based web application that automatically trains, evaluates, and ranks multiple machine learning models on your uploaded dataset.

## Features

- **Upload CSV**: Automatically detects numerical and categorical features.
- **Preprocessing**: Handles missing values, scaling, and encoding.
- **Model Training**: Trains Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, and KNN.
- **Hyperparameter Tuning**: Optional integration with Optuna.
- **Explainability**: SHAP values and summary plots.
- **Reporting**: Generates HTML reports, confusion matrices, and ROC curves.
- **Downloadable Artifacts**: Get the best model (`.pkl`), preprocessed data, and reports.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AutoML-Model-Selector
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload a CSV file (must have a header row).
4. Enter the name of the **Target Column** (the column you want to predict).
5. (Optional) Enable Optuna tuning or SHAP explainability.
6. Click **Run AutoML**.

## Deployment

### Docker

1. Build the Docker image:
   ```bash
   docker build -t automl-app .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 automl-app
   ```

3. Access the app at `http://localhost:5000`.

## Project Structure

- `app.py`: Main Flask application.
- `automl_core.py`: Core logic for preprocessing, training, and evaluation.
- `templates/`: HTML templates for the UI.
- `static/`: CSS and generated plots.
- `uploads/`: Temporary storage for uploaded files.
- `models/`: Saved best models.
- `reports/`: Generated HTML reports.
