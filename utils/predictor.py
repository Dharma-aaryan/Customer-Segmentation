import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Any, Dict, List

class Predictor:
    """Handles model predictions for new data inputs."""
    
    def __init__(self, model):
        self.model = model
    
    def show_manual_input(self):
        """Display interface for manual data input and prediction."""
        st.subheader("📝 Manual Input Prediction")
        
        # Try to get feature information from model
        feature_names = self._get_feature_names()
        n_features = self._get_n_features()
        
        if not feature_names and not n_features:
            st.warning("Cannot determine model input requirements. Please ensure your model has feature information.")
            return
        
        st.info(f"Model expects {n_features} features" + 
               (f": {', '.join(feature_names)}" if feature_names else ""))
        
        # Create input interface
        input_method = st.radio(
            "Choose input method:",
            ["Individual Inputs", "JSON Input"]
        )
        
        if input_method == "Individual Inputs":
            prediction_input = self._create_individual_inputs(feature_names, n_features)
        else:
            prediction_input = self._create_json_input(feature_names, n_features)
        
        if prediction_input is not None:
            if st.button("🔮 Make Prediction", type="primary"):
                self._make_single_prediction(prediction_input, feature_names)
    
    def _get_feature_names(self) -> List[str]:
        """Extract feature names from the model."""
        if hasattr(self.model, 'feature_names_in_'):
            return self.model.feature_names_in_.tolist()
        elif hasattr(self.model, 'feature_names_'):
            return self.model.feature_names_.tolist()
        return []
    
    def _get_n_features(self) -> int:
        """Extract number of features from the model."""
        if hasattr(self.model, 'n_features_in_'):
            return self.model.n_features_in_
        elif hasattr(self.model, 'n_features_'):
            return self.model.n_features_
        return 0
    
    def _create_individual_inputs(self, feature_names: List[str], n_features: int) -> np.ndarray:
        """Create individual input widgets for each feature."""
        if feature_names:
            inputs = []
            cols = st.columns(min(3, len(feature_names)))
            
            for i, feature in enumerate(feature_names):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    value = st.number_input(
                        f"{feature}:",
                        value=0.0,
                        key=f"input_{feature}"
                    )
                    inputs.append(value)
            
            return np.array(inputs).reshape(1, -1)
        
        elif n_features:
            st.write("Enter values for each feature:")
            inputs = []
            cols = st.columns(min(3, n_features))
            
            for i in range(n_features):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    value = st.number_input(
                        f"Feature {i+1}:",
                        value=0.0,
                        key=f"input_feature_{i}"
                    )
                    inputs.append(value)
            
            return np.array(inputs).reshape(1, -1)
        
        return None
    
    def _create_json_input(self, feature_names: List[str], n_features: int) -> np.ndarray:
        """Create JSON input interface."""
        if feature_names:
            example_json = {name: 0.0 for name in feature_names}
        else:
            example_json = {f"feature_{i+1}": 0.0 for i in range(n_features)}
        
        st.write("Enter input as JSON:")
        st.code(str(example_json), language="json")
        
        json_input = st.text_area(
            "Input JSON:",
            value=str(example_json),
            height=150
        )
        
        try:
            import json
            parsed_input = json.loads(json_input.replace("'", '"'))
            
            if feature_names:
                inputs = [parsed_input.get(name, 0.0) for name in feature_names]
            else:
                inputs = list(parsed_input.values())[:n_features]
            
            return np.array(inputs).reshape(1, -1)
        
        except Exception as e:
            st.error(f"Invalid JSON format: {str(e)}")
            return None
    
    def _make_single_prediction(self, input_data: np.ndarray, feature_names: List[str]):
        """Make prediction for single input."""
        try:
            with st.spinner("Making prediction..."):
                prediction = self.model.predict(input_data)
                
                st.success("✅ Prediction completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Prediction Result")
                    if len(prediction) == 1:
                        st.metric("Predicted Value", f"{prediction[0]}")
                    else:
                        for i, pred in enumerate(prediction):
                            st.metric(f"Output {i+1}", f"{pred}")
                
                with col2:
                    # Show prediction probabilities if available
                    if hasattr(self.model, 'predict_proba'):
                        st.subheader("📊 Prediction Probabilities")
                        try:
                            probabilities = self.model.predict_proba(input_data)[0]
                            classes = getattr(self.model, 'classes_', range(len(probabilities)))
                            
                            prob_df = pd.DataFrame({
                                'Class': classes,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            # Show as bar chart
                            fig = px.bar(
                                prob_df, x='Class', y='Probability',
                                title="Class Probabilities"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.warning(f"Could not show probabilities: {str(e)}")
                
                # Show input summary
                st.subheader("📋 Input Summary")
                if feature_names:
                    input_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': input_data[0]
                    })
                else:
                    input_df = pd.DataFrame({
                        'Feature': [f"Feature_{i+1}" for i in range(len(input_data[0]))],
                        'Value': input_data[0]
                    })
                
                st.dataframe(input_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
    
    def show_batch_prediction(self):
        """Display interface for batch predictions from uploaded file."""
        st.subheader("📁 Batch Prediction")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a file for batch predictions:",
            type=['csv', 'json', 'xlsx'],
            key="batch_prediction_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension == 'json':
                    df = pd.read_json(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Feature selection
                feature_names = self._get_feature_names()
                available_columns = df.columns.tolist()
                
                if feature_names:
                    # Try to match feature names
                    matched_features = [col for col in feature_names if col in available_columns]
                    missing_features = [col for col in feature_names if col not in available_columns]
                    
                    if missing_features:
                        st.warning(f"Missing features: {', '.join(missing_features)}")
                    
                    selected_features = st.multiselect(
                        "Select feature columns:",
                        available_columns,
                        default=matched_features
                    )
                else:
                    selected_features = st.multiselect(
                        "Select feature columns:",
                        available_columns,
                        default=available_columns
                    )
                
                if selected_features:
                    if st.button("🔮 Make Batch Predictions", type="primary"):
                        self._make_batch_predictions(df, selected_features)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    def _make_batch_predictions(self, df: pd.DataFrame, feature_columns: List[str]):
        """Make predictions for batch data."""
        try:
            with st.spinner("Making batch predictions..."):
                # Prepare data
                X = df[feature_columns]
                
                # Handle missing values
                if X.isnull().any().any():
                    st.warning("Dataset contains missing values. Filling with median/mode.")
                    for col in X.columns:
                        if X[col].dtype in ['int64', 'float64']:
                            X[col].fillna(X[col].median(), inplace=True)
                        else:
                            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
                
                # Make predictions
                predictions = self.model.predict(X)
                
                # Create results dataframe
                results_df = df.copy()
                
                if len(predictions.shape) == 1:
                    results_df['Prediction'] = predictions
                else:
                    for i in range(predictions.shape[1]):
                        results_df[f'Prediction_{i+1}'] = predictions[:, i]
                
                # Add prediction probabilities if available
                if hasattr(self.model, 'predict_proba'):
                    try:
                        probabilities = self.model.predict_proba(X)
                        classes = getattr(self.model, 'classes_', range(probabilities.shape[1]))
                        
                        for i, cls in enumerate(classes):
                            results_df[f'Prob_{cls}'] = probabilities[:, i]
                    except Exception:
                        pass  # Skip if probabilities can't be calculated
                
                st.success(f"✅ Batch predictions completed for {len(results_df)} samples!")
                
                # Show results
                st.subheader("📊 Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Prediction Summary")
                if 'Prediction' in results_df.columns:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if results_df['Prediction'].dtype in ['int64', 'float64']:
                            st.metric("Mean Prediction", f"{results_df['Prediction'].mean():.4f}")
                        else:
                            st.metric("Most Common", results_df['Prediction'].mode()[0])
                    
                    with col2:
                        if results_df['Prediction'].dtype in ['int64', 'float64']:
                            st.metric("Std Prediction", f"{results_df['Prediction'].std():.4f}")
                        else:
                            st.metric("Unique Values", results_df['Prediction'].nunique())
                    
                    with col3:
                        if results_df['Prediction'].dtype in ['int64', 'float64']:
                            st.metric("Min/Max", f"{results_df['Prediction'].min():.2f} / {results_df['Prediction'].max():.2f}")
                        else:
                            st.metric("Total Samples", len(results_df))
                
                # Visualization
                if 'Prediction' in results_df.columns:
                    if results_df['Prediction'].dtype in ['int64', 'float64']:
                        # Histogram for numeric predictions
                        fig = px.histogram(
                            results_df, x='Prediction',
                            title="Distribution of Predictions"
                        )
                    else:
                        # Count plot for categorical predictions
                        pred_counts = results_df['Prediction'].value_counts()
                        fig = px.bar(
                            x=pred_counts.index, y=pred_counts.values,
                            title="Distribution of Predictions",
                            labels={'x': 'Prediction', 'y': 'Count'}
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.subheader("💾 Download Results")
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Batch prediction failed: {str(e)}")
