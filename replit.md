# Customer Segmentation Dashboard

## Overview

This is a specialized Customer Segmentation Dashboard built with Streamlit that recreates and enhances the analysis from the original Customer Segmentation notebook. The dashboard provides an interactive interface for exploring mall customer data, performing K-means clustering analysis, and generating business insights. Based on the GitHub repository: https://github.com/Dharma-aaryan/Customer-Segmentation

**Recent Changes (Aug 8, 2025):**
- Cleaned up codebase by removing unnecessary imports and unused CSS
- Fixed LSP errors by updating KMeans parameters to use 'auto' for n_init
- Simplified app.py to contain only a redirect message (main dashboard is customer_segmentation_dashboard.py)
- Created minimal requirements list: streamlit, pandas, numpy, plotly, scikit-learn
- Added README.md with clear local installation instructions

Key features:
- Interactive K-means clustering with configurable cluster count
- Elbow method analysis for optimal cluster determination  
- Comprehensive customer segment analysis with business descriptions
- Demographic analysis by gender, age, and spending patterns
- Business insights and marketing recommendations
- Professional styling with custom CSS and interactive Plotly visualizations

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with tabbed navigation structure
- **Layout**: Wide layout with sidebar controls for interactive analysis parameters
- **State Management**: Streamlit caching for efficient data loading and processing
- **UI Components**: Five main tabs - Dataset Overview, Elbow Method, Customer Segments, Demographic Analysis, Business Insights
- **Styling**: Custom CSS for professional appearance with metric cards, insight boxes, and branded color scheme

### Backend Architecture
- **Core Functions**: Single-file architecture with specialized functions for customer segmentation analysis
- **Key Components**:
  - `load_data()`: Loads and caches the Mall Customer CSV dataset
  - `perform_clustering()`: Executes K-means clustering on Income and Spending Score features
  - `calculate_elbow_method()`: Determines optimal cluster count using Within-Cluster Sum of Squares
  - `analyze_clusters()`: Generates business insights and customer segment descriptions
  - `create_visualizations()`: Builds interactive Plotly charts for cluster analysis
- **Data Processing**: Pandas for data manipulation, scikit-learn for K-means clustering, NumPy for numerical operations

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