import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelAnalyzer:
    """Analyzes model performance and provides visualizations."""
    
    def __init__(self, model, df: pd.DataFrame):
        self.model = model
        self.df = df
        self.is_classifier = self._detect_model_type()
    
    def _detect_model_type(self):
        """Detect if the model is a classifier or regressor."""
        # Check common classifier attributes/methods
        classifier_indicators = [
            'predict_proba', 'decision_function', 'classes_',
            'n_classes_', 'class_weight', 'class_prior_'
        ]
        
        return any(hasattr(self.model, attr) for attr in classifier_indicators)
    
    def show_performance_metrics(self):
        """Display model performance metrics."""
        st.subheader("🎯 Performance Metrics")
        
        # Get target and feature columns
        target_col = st.selectbox(
            "Select target column:",
            self.df.columns.tolist(),
            key="target_selection"
        )
        
        if not target_col:
            st.warning("Please select a target column.")
            return
        
        feature_cols = st.multiselect(
            "Select feature columns:",
            [col for col in self.df.columns if col != target_col],
            default=[col for col in self.df.columns if col != target_col][:5],
            key="feature_selection"
        )
        
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return
        
        try:
            X = self.df[feature_cols]
            y = self.df[target_col]
            
            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                st.warning("Dataset contains missing values. Removing rows with missing values.")
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            if self.is_classifier:
                self._show_classification_metrics(y, y_pred, X)
            else:
                self._show_regression_metrics(y, y_pred)
                
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
    
    def _show_classification_metrics(self, y_true, y_pred, X):
        """Display classification performance metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = accuracy_score(y_true, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
        
        with col2:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            st.metric("Precision", f"{precision:.4f}")
        
        with col3:
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            st.metric("Recall", f"{recall:.4f}")
        
        with col4:
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            st.metric("F1-Score", f"{f1:.4f}")
        
        # Classification report
        st.subheader("📊 Detailed Classification Report")
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating classification report: {str(e)}")
        
        # Cross-validation scores
        st.subheader("🔄 Cross-Validation Scores")
        try:
            cv_scores = cross_val_score(self.model, X, y_true, cv=5, scoring='accuracy')
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
            with col2:
                st.metric("Std CV Score", f"{cv_scores.std():.4f}")
            with col3:
                st.metric("Best CV Score", f"{cv_scores.max():.4f}")
            
            # Plot CV scores
            fig = px.bar(
                x=range(1, len(cv_scores) + 1),
                y=cv_scores,
                title="Cross-Validation Scores by Fold",
                labels={'x': 'Fold', 'y': 'Accuracy Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not perform cross-validation: {str(e)}")
    
    def _show_regression_metrics(self, y_true, y_pred):
        """Display regression performance metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mse = mean_squared_error(y_true, y_pred)
            st.metric("MSE", f"{mse:.4f}")
        
        with col2:
            rmse = np.sqrt(mse)
            st.metric("RMSE", f"{rmse:.4f}")
        
        with col3:
            mae = mean_absolute_error(y_true, y_pred)
            st.metric("MAE", f"{mae:.4f}")
        
        with col4:
            r2 = r2_score(y_true, y_pred)
            st.metric("R² Score", f"{r2:.4f}")
        
        # Residuals analysis
        st.subheader("📈 Residuals Analysis")
        residuals = y_true - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals vs Predicted
            fig = px.scatter(
                x=y_pred, y=residuals,
                title="Residuals vs Predicted Values",
                labels={'x': 'Predicted Values', 'y': 'Residuals'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals distribution
            fig = px.histogram(
                x=residuals,
                title="Distribution of Residuals",
                labels={'x': 'Residuals', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_visualizations(self):
        """Display various model performance visualizations."""
        st.subheader("📊 Performance Visualizations")
        
        # Get target and feature columns
        target_col = st.selectbox(
            "Select target column:",
            self.df.columns.tolist(),
            key="viz_target_selection"
        )
        
        if not target_col:
            st.warning("Please select a target column.")
            return
        
        feature_cols = st.multiselect(
            "Select feature columns:",
            [col for col in self.df.columns if col != target_col],
            default=[col for col in self.df.columns if col != target_col][:5],
            key="viz_feature_selection"
        )
        
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return
        
        try:
            X = self.df[feature_cols]
            y = self.df[target_col]
            
            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
            
            y_pred = self.model.predict(X)
            
            if self.is_classifier:
                self._show_classification_visualizations(y, y_pred, X)
            else:
                self._show_regression_visualizations(y, y_pred)
                
        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
    
    def _show_classification_visualizations(self, y_true, y_pred, X):
        """Show classification-specific visualizations."""
        # Confusion Matrix
        st.subheader("🔄 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve (for binary classification)
        if hasattr(self.model, 'predict_proba') and len(np.unique(y_true)) == 2:
            st.subheader("📈 ROC Curve")
            try:
                y_prob = self.model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.2f})'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Random Classifier'
                ))
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate ROC curve: {str(e)}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            st.subheader("📊 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df, x='Importance', y='Feature',
                orientation='h',
                title="Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_regression_visualizations(self, y_true, y_pred):
        """Show regression-specific visualizations."""
        # Actual vs Predicted
        st.subheader("🎯 Actual vs Predicted")
        
        fig = px.scatter(
            x=y_true, y=y_pred,
            title="Actual vs Predicted Values",
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}
        )
        
        # Add diagonal line for perfect predictions
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction'
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction errors
        st.subheader("📉 Prediction Errors")
        errors = y_true - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                x=errors,
                title="Distribution of Prediction Errors",
                labels={'x': 'Prediction Error', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                y=errors,
                title="Box Plot of Prediction Errors"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_info(self):
        """Display model information and parameters."""
        st.subheader("🔧 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Type:**")
            st.code(str(type(self.model).__name__))
            
            st.write("**Model Module:**")
            st.code(str(type(self.model).__module__))
        
        with col2:
            st.write("**Model Parameters:**")
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                params_df = pd.DataFrame(
                    list(params.items()),
                    columns=['Parameter', 'Value']
                )
                st.dataframe(params_df, use_container_width=True)
            else:
                st.info("Model parameters not available")
        
        # Model attributes
        st.subheader("📋 Model Attributes")
        attributes = []
        
        # Common model attributes to check
        attr_names = [
            'classes_', 'n_classes_', 'feature_importances_',
            'coef_', 'intercept_', 'n_features_in_',
            'feature_names_in_', 'n_outputs_'
        ]
        
        for attr in attr_names:
            if hasattr(self.model, attr):
                value = getattr(self.model, attr)
                if isinstance(value, np.ndarray):
                    attributes.append({
                        'Attribute': attr,
                        'Type': 'numpy.ndarray',
                        'Shape': str(value.shape),
                        'Value': 'Array data (too large to display)'
                    })
                else:
                    attributes.append({
                        'Attribute': attr,
                        'Type': str(type(value).__name__),
                        'Shape': 'N/A',
                        'Value': str(value)
                    })
        
        if attributes:
            attr_df = pd.DataFrame(attributes)
            st.dataframe(attr_df, use_container_width=True)
        else:
            st.info("No common model attributes found")
