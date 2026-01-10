# AutoML Model Selector

A powerful, user-friendly web application that automatically selects and trains the best machine learning model for your dataset. Built with Flask, scikit-learn, and modern web technologies.

## Features

### Core Functionality
- **Automated Model Selection**: Tests multiple ML algorithms and selects the best performer
- **Classification & Regression**: Support for both task types
- **Smart Preprocessing**: Automatic handling of missing values, encoding, and scaling
- **Model Comparison**: Visual leaderboards comparing all tested models
- **Performance Metrics**: Comprehensive evaluation metrics for each model

### Advanced Features
- **Hyperparameter Tuning**: Optional Optuna-based optimization for better accuracy
- **Feature Engineering**: Automated creation of new features from existing data
- **Model Explainability**: SHAP values to understand feature importance
- **Interactive Visualizations**:
  - Model comparison charts
  - Confusion matrices (classification)
  - ROC curves (classification)
  - Actual vs Predicted plots (regression)
  - Residual analysis (regression)
- **Experiment Tracking**: Keep history of all training runs
- **Data Preview**: Preview your dataset before training

### User Experience
- Modern, responsive UI
- Real-time dataset preview
- Column auto-detection
- Progress tracking
- Downloadable artifacts (models, reports, preprocessed data)

## Supported Models

### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors Classifier
- Gaussian Naive Bayes
- XGBoost Classifier (optional)
- LightGBM Classifier (optional)
- CatBoost Classifier (optional)

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor
- XGBoost Regressor (optional)
- LightGBM Regressor (optional)
- CatBoost Regressor (optional)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd AutoML-Model-Selector

# Install required dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality, install these optional packages:

```bash
# Hyperparameter tuning
pip install optuna

# Model explainability
pip install shap

# Advanced gradient boosting models
pip install xgboost lightgbm catboost
```

## Usage

### Starting the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Basic Workflow

1. **Upload Dataset**:
   - Click "Choose File" and select your CSV file
   - Maximum file size: 100MB

2. **Preview Data** (Optional):
   - Click "Preview Dataset" to see data statistics
   - View column names, types, and missing values
   - Select target column from dropdown

3. **Configure Settings**:
   - Enter target column name
   - Select task type (Classification or Regression)
   - Optional: Enable advanced features

4. **Advanced Options**:
   - **Feature Engineering**: Auto-create polynomial and interaction features
   - **Hyperparameter Tuning**: Optimize model parameters (slower but more accurate)
   - **SHAP Explainability**: Generate feature importance analysis
   - **Tuning Trials**: Number of optimization attempts (10-50 recommended)
   - **SHAP Samples**: Sample size for explainability (50-1000)

5. **Run AutoML**:
   - Click "Run AutoML"
   - Wait for training to complete (time varies by dataset size)

6. **View Results**:
   - See model leaderboard with performance metrics
   - Visualize predictions and model performance
   - Download trained model, preprocessed data, and reports

7. **Explore Past Experiments**:
   - Navigate to "Past Experiments" to view training history
   - Click on any experiment to see detailed results

## File Structure

```
AutoML-Model-Selector/
├── app.py                      # Flask application
├── automl_core.py             # Core ML logic
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── templates/
│   ├── upload.html           # Main upload page
│   ├── results.html          # Results display page
│   ├── experiments.html      # Past experiments list
│   └── experiment_detail.html # Individual experiment details
├── uploads/                   # Uploaded datasets
├── models/                    # Trained model files
├── reports/                   # HTML reports
├── preprocessed/             # Preprocessed datasets
├── experiments/              # Experiment metadata (JSON)
└── static/
    └── plots/                # Generated visualizations
```

## API Endpoints

### Web Routes
- `GET /` - Main upload page
- `POST /` - Process dataset and run AutoML
- `GET /experiments` - List all past experiments
- `GET /experiment/<run_id>` - View specific experiment details

### API Routes
- `POST /api/preview` - Preview dataset statistics
- `POST /api/columns` - Get column names from CSV

### Download Routes
- `GET /download/model/<filename>` - Download trained model
- `GET /download/preprocessed/<filename>` - Download preprocessed data
- `GET /download/report/<filename>` - Download HTML report

### Utility Routes
- `GET /health` - Health check endpoint

## Configuration

### Environment Variables

```bash
# Flask secret key (change in production)
export FLASK_SECRET=your-secret-key-here

# Port (default: 5000)
export PORT=5000

# Debug mode (default: True)
export DEBUG=True
```

### File Size Limits

Default maximum upload size is 100MB. To change:

```python
# In app.py
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
```

## Data Requirements

### Supported Format
- CSV files only
- UTF-8 or Latin-1 encoding
- Header row required

### Dataset Requirements
- Minimum 10 rows
- At least one feature column
- Target column specified
- Target column should not have >50% missing values

### Recommendations
- Remove or fix duplicate rows
- Handle high-cardinality categorical features (>50 unique values)
- Address columns with >80% missing values
- Balance highly imbalanced datasets if possible

## Output Files

After training, you'll receive:

1. **Trained Model** (.pkl)
   - Complete scikit-learn pipeline
   - Ready for production use
   - Load with: `joblib.load('model.pkl')`

2. **Preprocessed Data** (.csv)
   - Transformed features
   - Ready for further analysis
   - Same preprocessing as applied to training data

3. **HTML Report** (.html)
   - Comprehensive analysis
   - All metrics and visualizations
   - Shareable document

4. **Visualizations** (.png)
   - Model comparison charts
   - Confusion matrices / ROC curves
   - Feature importance (if SHAP enabled)
   - Actual vs Predicted plots

5. **Experiment Metadata** (.json)
   - Configuration used
   - All model metrics
   - Dataset information
   - Timestamp and run ID

## Metrics Explained

### Classification Metrics
- **Accuracy**: Overall correctness (0-1, higher is better)
- **F1-Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **ROC-AUC**: Area under ROC curve (0-1, higher is better)

### Regression Metrics
- **R² Score**: Proportion of variance explained (-∞ to 1, higher is better, 1 is perfect)
- **MAE**: Mean Absolute Error (lower is better, same units as target)
- **RMSE**: Root Mean Squared Error (lower is better, penalizes large errors)

## Troubleshooting

### Common Issues

**Issue**: "Target column not found"
- **Solution**: Check spelling and ensure column exists in CSV

**Issue**: "Dataset is empty" or "Too few rows"
- **Solution**: Ensure CSV has data and at least 10 rows

**Issue**: "File too large" error
- **Solution**: Reduce dataset size or increase MAX_FILE_SIZE in app.py

**Issue**: Models taking too long to train
- **Solution**:
  - Disable Optuna hyperparameter tuning
  - Reduce number of tuning trials
  - Disable SHAP explainability for large datasets
  - Reduce SHAP sample size

**Issue**: Optional libraries not available warnings
- **Solution**: Install optional dependencies:
  ```bash
  pip install optuna shap xgboost lightgbm catboost
  ```

**Issue**: Memory errors with large datasets
- **Solution**:
  - Reduce dataset size
  - Disable feature engineering
  - Use fewer SHAP samples
  - Close other applications

## Performance Tips

1. **For Large Datasets (>100K rows)**:
   - Disable SHAP or use very small sample size (50-100)
   - Use fewer Optuna trials (10-20)
   - Consider sampling your dataset

2. **For High-Dimensional Data (>100 features)**:
   - Enable feature selection
   - Disable polynomial feature engineering
   - Use tree-based models (faster with many features)

3. **For Quick Prototyping**:
   - Disable Optuna tuning
   - Disable SHAP
   - Disable feature engineering

4. **For Maximum Accuracy**:
   - Enable Optuna with 50+ trials
   - Enable feature engineering
   - Use ensemble models (Random Forest, Gradient Boosting, XGBoost)

## Security Considerations

- **Production Deployment**:
  - Change `FLASK_SECRET` to a strong random value
  - Use a production WSGI server (gunicorn, waitress)
  - Enable HTTPS
  - Add authentication if needed
  - Validate and sanitize all user inputs

- **File Upload Security**:
  - Only CSV files are accepted
  - File size limits enforced
  - Files saved with secure filenames
  - Consider virus scanning for production

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers (responsive design)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

Built with:
- Flask - Web framework
- scikit-learn - Machine learning library
- Pandas - Data manipulation
- Matplotlib/Seaborn - Visualization
- Optuna - Hyperparameter optimization
- SHAP - Model explainability
- XGBoost/LightGBM/CatBoost - Advanced ML models

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

## Version History

### v1.0.0 (Current)
- Initial release
- Classification and regression support
- 10+ ML algorithms
- Hyperparameter tuning
- SHAP explainability
- Feature engineering
- Experiment tracking
- Modern responsive UI

---

Made with ❤️ for the ML community
