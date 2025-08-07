import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

class DataExplorer:
    """Handles data exploration functionality for the ML dashboard."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def show_overview(self):
        """Display dataset overview and basic information."""
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", self.df.shape[0])
        with col2:
            st.metric("Columns", self.df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Missing Values", self.df.isnull().sum().sum())
        
        st.subheader("📊 Data Types")
        col1, col2 = st.columns(2)
        
        with col1:
            # Data types summary
            dtype_df = pd.DataFrame({
                'Column': self.df.columns,
                'Data Type': self.df.dtypes.astype(str),
                'Non-Null Count': self.df.count(),
                'Null Count': self.df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            # Data types visualization
            dtype_counts = self.df.dtypes.value_counts()
            fig = px.pie(
                values=dtype_counts.values, 
                names=dtype_counts.index,
                title="Distribution of Data Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("👀 Data Preview")
        
        # Show head and tail
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 10 rows:**")
            st.dataframe(self.df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Last 10 rows:**")
            st.dataframe(self.df.tail(10), use_container_width=True)
    
    def show_visualizations(self):
        """Display various visualizations of the dataset."""
        st.subheader("📊 Data Visualizations")
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_columns and not categorical_columns:
            st.warning("No suitable columns found for visualization.")
            return
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Correlation Heatmap", "Distribution Plots", "Scatter Plot", "Box Plot", "Count Plot"]
        )
        
        if viz_type == "Correlation Heatmap" and len(numeric_columns) > 1:
            self._show_correlation_heatmap(numeric_columns)
        
        elif viz_type == "Distribution Plots" and numeric_columns:
            self._show_distribution_plots(numeric_columns)
        
        elif viz_type == "Scatter Plot" and len(numeric_columns) >= 2:
            self._show_scatter_plot(numeric_columns, categorical_columns)
        
        elif viz_type == "Box Plot" and numeric_columns:
            self._show_box_plot(numeric_columns, categorical_columns)
        
        elif viz_type == "Count Plot" and categorical_columns:
            self._show_count_plot(categorical_columns)
        
        else:
            st.warning(f"Cannot create {viz_type} with available data types.")
    
    def _show_correlation_heatmap(self, numeric_columns):
        """Display correlation heatmap for numeric columns."""
        correlation_matrix = self.df[numeric_columns].corr(numeric_only=True)
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_distribution_plots(self, numeric_columns):
        """Display distribution plots for numeric columns."""
        selected_columns = st.multiselect(
            "Select columns for distribution plots:",
            numeric_columns,
            default=numeric_columns[:4]  # Default to first 4 columns
        )
        
        if selected_columns:
            n_cols = min(2, len(selected_columns))
            n_rows = (len(selected_columns) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_columns
            )
            
            for i, col in enumerate(selected_columns):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=self.df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(title="Distribution Plots", height=400*n_rows)
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_scatter_plot(self, numeric_columns, categorical_columns):
        """Display scatter plot with customizable axes and color coding."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis:", numeric_columns)
        with col2:
            y_axis = st.selectbox("Y-axis:", numeric_columns)
        with col3:
            color_col = st.selectbox("Color by:", ["None"] + categorical_columns)
        
        if x_axis and y_axis:
            color = color_col if color_col != "None" else None
            
            fig = px.scatter(
                self.df, x=x_axis, y=y_axis, color=color,
                title=f"Scatter Plot: {x_axis} vs {y_axis}",
                hover_data=numeric_columns[:3]  # Add hover data
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_box_plot(self, numeric_columns, categorical_columns):
        """Display box plots for numeric columns."""
        col1, col2 = st.columns(2)
        
        with col1:
            y_axis = st.selectbox("Select numeric column:", numeric_columns)
        with col2:
            x_axis = st.selectbox("Group by (optional):", ["None"] + categorical_columns)
        
        if y_axis:
            x = x_axis if x_axis != "None" else None
            
            fig = px.box(
                self.df, x=x, y=y_axis,
                title=f"Box Plot: {y_axis}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_count_plot(self, categorical_columns):
        """Display count plots for categorical columns."""
        selected_column = st.selectbox("Select categorical column:", categorical_columns)
        
        if selected_column:
            value_counts = self.df[selected_column].value_counts()
            
            fig = px.bar(
                x=value_counts.index, y=value_counts.values,
                title=f"Count Plot: {selected_column}",
                labels={'x': selected_column, 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_filtering(self):
        """Display filtering interface and return filtered dataframe."""
        st.subheader("🔍 Data Filtering")
        
        # Create filters based on column types
        filters = {}
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Filters:**")
            for col in numeric_columns:
                min_val, max_val = float(self.df[col].min()), float(self.df[col].max())
                filters[col] = st.slider(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"filter_{col}"
                )
        
        with col2:
            st.write("**Categorical Filters:**")
            for col in categorical_columns:
                unique_values = self.df[col].dropna().unique().tolist()
                filters[col] = st.multiselect(
                    f"{col}",
                    options=unique_values,
                    default=unique_values,
                    key=f"filter_{col}"
                )
        
        # Apply filters
        filtered_df = self.df.copy()
        
        for col, filter_value in filters.items():
            if col in numeric_columns:
                min_val, max_val = filter_value
                filtered_df = filtered_df[
                    (filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)
                ]
            elif col in categorical_columns and filter_value:
                filtered_df = filtered_df[filtered_df[col].isin(filter_value)]
        
        st.subheader("Filtered Results")
        st.info(f"Filtered dataset: {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export filtered data
        if st.button("Save Filtered Dataset"):
            return filtered_df
        
        return None
    
    def show_statistics(self):
        """Display detailed statistical analysis."""
        st.subheader("📈 Statistical Analysis")
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            # Descriptive statistics
            st.write("**Descriptive Statistics:**")
            st.dataframe(self.df[numeric_columns].describe(), use_container_width=True)
            
            # Additional statistics
            st.write("**Additional Statistics:**")
            stats_df = pd.DataFrame({
                'Column': numeric_columns,
                'Skewness': [self.df[col].skew() for col in numeric_columns],
                'Kurtosis': [self.df[col].kurtosis() for col in numeric_columns],
                'Variance': [self.df[col].var() for col in numeric_columns]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        # Missing values analysis
        st.write("**Missing Values Analysis:**")
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': self.df.isnull().sum(),
            'Missing Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        
        if not missing_data.empty:
            st.dataframe(missing_data, use_container_width=True)
            
            # Visualize missing data
            fig = px.bar(
                missing_data, x='Column', y='Missing Percentage',
                title="Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
