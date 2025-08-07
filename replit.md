# ML Project Dashboard

## Overview

This is a comprehensive Machine Learning Dashboard built with Streamlit that provides an interactive interface for ML project analysis, visualization, and predictions. The application serves as a centralized platform for data scientists and ML practitioners to upload datasets, analyze model performance, make predictions, and examine Jupyter notebooks. The dashboard supports multiple file formats and provides detailed insights through interactive visualizations and metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with a multi-page navigation structure
- **Layout**: Wide layout with expandable sidebar for navigation between different functional sections
- **State Management**: Streamlit session state for persisting datasets, models, and notebooks across page navigation
- **UI Components**: Modular page structure with five main sections: File Upload, Data Exploration, Model Performance, Predictions, and Notebook Analysis

### Backend Architecture
- **Modular Design**: Utils-based architecture with separate classes for distinct functionalities
- **Core Components**:
  - `DataExplorer`: Handles dataset analysis and visualization
  - `ModelAnalyzer`: Provides ML model performance analysis and metrics
  - `Predictor`: Manages prediction workflows for trained models
  - `NotebookAnalyzer`: Processes and analyzes Jupyter notebook content
  - `FileHandler`: Centralizes file upload and loading operations
- **Data Processing**: Pandas-based data manipulation with NumPy for numerical operations

### Visualization Framework
- **Primary Library**: Plotly for interactive charts and graphs
- **Secondary Libraries**: Matplotlib and Seaborn for specialized visualizations
- **Chart Types**: Support for pie charts, scatter plots, subplots, and custom dashboard metrics

### Model Support
- **Model Loading**: Support for pickle (.pkl), joblib, and standard Python pickle formats
- **Model Types**: Automatic detection of classification vs regression models
- **Performance Metrics**: Comprehensive evaluation including accuracy, precision, recall, F1-score for classifiers, and MSE, MAE, R² for regressors
- **Cross-validation**: Integration with scikit-learn for model validation

### File Processing
- **Dataset Formats**: CSV, JSON, Excel (.xlsx, .xls)
- **Model Formats**: Pickle (.pkl), Joblib
- **Notebook Format**: Jupyter notebook JSON format (.ipynb)
- **Error Handling**: Robust exception handling for file loading operations

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework and UI components
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### Visualization Libraries
- **Plotly Express & Graph Objects**: Interactive plotting and dashboard visualizations
- **Matplotlib**: Static plotting capabilities
- **Seaborn**: Statistical data visualization

### Machine Learning Libraries
- **Scikit-learn**: Model evaluation metrics, cross-validation, and ML utilities
- **Pickle/Joblib**: Model serialization and deserialization

### File Processing
- **JSON**: Built-in Python library for JSON parsing (notebook analysis)
- **Base64**: Built-in Python library for encoding operations

### Development Dependencies
- Standard Python libraries for regex operations, typing annotations, and general utilities