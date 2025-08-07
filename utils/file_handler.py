import streamlit as st
import pandas as pd
import pickle
import joblib
import json
from typing import Any, Dict

class FileHandler:
    """Handles file upload and loading operations for the ML dashboard."""
    
    def load_dataset(self, file) -> pd.DataFrame:
        """Load dataset from uploaded file."""
        try:
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file)
            elif file_extension == 'json':
                df = pd.read_json(file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def load_model(self, file) -> Any:
        """Load ML model from uploaded file."""
        try:
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension in ['pkl', 'pickle']:
                model = pickle.load(file)
            elif file_extension == 'joblib':
                model = joblib.load(file)
            else:
                raise ValueError(f"Unsupported model format: {file_extension}")
            
            return model
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def load_notebook(self, file) -> Dict:
        """Load Jupyter notebook from uploaded file."""
        try:
            notebook_content = json.load(file)
            
            # Validate notebook structure
            if 'cells' not in notebook_content:
                raise ValueError("Invalid notebook format: missing 'cells' key")
            
            return notebook_content
            
        except Exception as e:
            raise Exception(f"Error loading notebook: {str(e)}")
    
    @staticmethod
    def export_dataframe(df: pd.DataFrame, filename: str, format: str = 'csv'):
        """Export dataframe to specified format."""
        try:
            if format == 'csv':
                return df.to_csv(index=False)
            elif format == 'json':
                return df.to_json(orient='records', indent=2)
            elif format == 'excel':
                # For Excel export, we'll use a bytes buffer
                import io
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                return buffer.getvalue()
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None
