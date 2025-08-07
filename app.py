import streamlit as st
import pandas as pd
import numpy as np
from utils.data_explorer import DataExplorer
from utils.model_analyzer import ModelAnalyzer
from utils.predictor import Predictor
from utils.notebook_analyzer import NotebookAnalyzer
from utils.file_handler import FileHandler

# Page configuration
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 ML Project Dashboard")
    st.markdown("### Comprehensive dashboard for ML project analysis, visualization, and predictions")
    
    # Initialize session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'notebooks' not in st.session_state:
        st.session_state.notebooks = {}
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 File Upload", "📊 Data Exploration", "📈 Model Performance", "🔮 Predictions", "📓 Notebook Analysis"]
    )
    
    file_handler = FileHandler()
    
    if page == "📁 File Upload":
        show_file_upload(file_handler)
    elif page == "📊 Data Exploration":
        show_data_exploration()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "🔮 Predictions":
        show_predictions()
    elif page == "📓 Notebook Analysis":
        show_notebook_analysis()

def show_file_upload(file_handler):
    st.header("📁 File Upload")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Upload Dataset")
        dataset_file = st.file_uploader(
            "Choose a dataset file",
            type=['csv', 'json', 'xlsx'],
            key="dataset_upload"
        )
        
        if dataset_file is not None:
            try:
                with st.spinner("Loading dataset..."):
                    df = file_handler.load_dataset(dataset_file)
                    st.session_state.datasets[dataset_file.name] = df
                    st.success(f"✅ Dataset '{dataset_file.name}' loaded successfully!")
                    st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    # Show preview
                    st.subheader("Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
    
    with col2:
        st.subheader("🤖 Upload Model")
        model_file = st.file_uploader(
            "Choose a model file",
            type=['pkl', 'pickle', 'joblib'],
            key="model_upload"
        )
        
        if model_file is not None:
            try:
                with st.spinner("Loading model..."):
                    model = file_handler.load_model(model_file)
                    st.session_state.models[model_file.name] = model
                    st.success(f"✅ Model '{model_file.name}' loaded successfully!")
                    st.info(f"Model type: {type(model).__name__}")
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    with col3:
        st.subheader("📓 Upload Notebook")
        notebook_file = st.file_uploader(
            "Choose a Jupyter notebook",
            type=['ipynb'],
            key="notebook_upload"
        )
        
        if notebook_file is not None:
            try:
                with st.spinner("Loading notebook..."):
                    notebook_content = file_handler.load_notebook(notebook_file)
                    st.session_state.notebooks[notebook_file.name] = notebook_content
                    st.success(f"✅ Notebook '{notebook_file.name}' loaded successfully!")
                    
                    # Show basic info
                    cells = notebook_content.get('cells', [])
                    code_cells = len([cell for cell in cells if cell.get('cell_type') == 'code'])
                    markdown_cells = len([cell for cell in cells if cell.get('cell_type') == 'markdown'])
                    st.info(f"Cells: {code_cells} code, {markdown_cells} markdown")
                    
            except Exception as e:
                st.error(f"❌ Error loading notebook: {str(e)}")
    
    # Show current files
    if st.session_state.datasets or st.session_state.models or st.session_state.notebooks:
        st.divider()
        st.subheader("📋 Currently Loaded Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.datasets:
                st.write("**Datasets:**")
                for name in st.session_state.datasets.keys():
                    st.write(f"- {name}")
        
        with col2:
            if st.session_state.models:
                st.write("**Models:**")
                for name in st.session_state.models.keys():
                    st.write(f"- {name}")
        
        with col3:
            if st.session_state.notebooks:
                st.write("**Notebooks:**")
                for name in st.session_state.notebooks.keys():
                    st.write(f"- {name}")

def show_data_exploration():
    st.header("📊 Data Exploration")
    
    if not st.session_state.datasets:
        st.warning("⚠️ No datasets loaded. Please upload a dataset in the File Upload section.")
        return
    
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select a dataset to explore:",
        list(st.session_state.datasets.keys())
    )
    
    if selected_dataset:
        df = st.session_state.datasets[selected_dataset]
        explorer = DataExplorer(df)
        
        # Tabs for different exploration views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Visualizations", "🔍 Filtering", "📈 Statistics"])
        
        with tab1:
            explorer.show_overview()
        
        with tab2:
            explorer.show_visualizations()
        
        with tab3:
            filtered_df = explorer.show_filtering()
            if filtered_df is not None:
                st.session_state.datasets[f"{selected_dataset}_filtered"] = filtered_df
        
        with tab4:
            explorer.show_statistics()

def show_model_performance():
    st.header("📈 Model Performance")
    
    if not st.session_state.models:
        st.warning("⚠️ No models loaded. Please upload a model in the File Upload section.")
        return
    
    if not st.session_state.datasets:
        st.warning("⚠️ No datasets loaded. Please upload a dataset for model evaluation.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Select a model:",
            list(st.session_state.models.keys())
        )
    
    with col2:
        selected_dataset = st.selectbox(
            "Select a dataset for evaluation:",
            list(st.session_state.datasets.keys())
        )
    
    if selected_model and selected_dataset:
        model = st.session_state.models[selected_model]
        df = st.session_state.datasets[selected_dataset]
        
        analyzer = ModelAnalyzer(model, df)
        
        # Model analysis tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Performance Metrics", "📊 Visualizations", "🔧 Model Info"])
        
        with tab1:
            analyzer.show_performance_metrics()
        
        with tab2:
            analyzer.show_visualizations()
        
        with tab3:
            analyzer.show_model_info()

def show_predictions():
    st.header("🔮 Predictions")
    
    if not st.session_state.models:
        st.warning("⚠️ No models loaded. Please upload a model in the File Upload section.")
        return
    
    selected_model = st.selectbox(
        "Select a model for predictions:",
        list(st.session_state.models.keys())
    )
    
    if selected_model:
        model = st.session_state.models[selected_model]
        predictor = Predictor(model)
        
        # Prediction tabs
        tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Batch Prediction"])
        
        with tab1:
            predictor.show_manual_input()
        
        with tab2:
            predictor.show_batch_prediction()

def show_notebook_analysis():
    st.header("📓 Notebook Analysis")
    
    if not st.session_state.notebooks:
        st.warning("⚠️ No notebooks loaded. Please upload a notebook in the File Upload section.")
        return
    
    selected_notebook = st.selectbox(
        "Select a notebook to analyze:",
        list(st.session_state.notebooks.keys())
    )
    
    if selected_notebook:
        notebook_content = st.session_state.notebooks[selected_notebook]
        analyzer = NotebookAnalyzer(notebook_content)
        
        # Notebook analysis tabs
        tab1, tab2, tab3 = st.tabs(["📋 Overview", "💻 Code Analysis", "📊 Visualizations"])
        
        with tab1:
            analyzer.show_overview()
        
        with tab2:
            analyzer.show_code_analysis()
        
        with tab3:
            analyzer.show_visualizations()

if __name__ == "__main__":
    main()
