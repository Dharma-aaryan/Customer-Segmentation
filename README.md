# Customer Segmentation Dashboard

An interactive Streamlit dashboard for customer segmentation analysis using K-means clustering on mall customer data.

## Quick Start

### Installation
```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Run the Dashboard
```bash
streamlit run customer_segmentation_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure
```
├── customer_segmentation_dashboard.py  # Main dashboard application
├── data/
│   └── MallCustomers.csv              # Customer dataset
├── app.py                             # Redirect file (not used)
└── README.md                          # This file
```

**Note:** Only `customer_segmentation_dashboard.py` is needed to run the application. The `app.py` file is not used for this project.

## Features
- **Dataset Overview**: Explore customer demographics and distributions
- **Elbow Method**: Determine optimal number of clusters
- **Customer Segments**: Interactive K-means clustering visualization
- **Demographic Analysis**: Age, gender, and spending pattern analysis
- **Business Insights**: Marketing recommendations and segmentation results

## Dataset
The dashboard uses the Mall Customer dataset with the following features:
- CustomerID: Unique customer identifier
- Gender: Customer gender (Male/Female)
- Age: Customer age
- Annual Income (k$): Annual income in thousands of dollars
- Spending Score (1-100): Mall spending behavior score

## Requirements
- Python 3.7+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn