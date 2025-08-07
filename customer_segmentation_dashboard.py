import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border: 1px solid #e6e9f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .segment-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the customer data"""
    df = pd.read_csv('data/MallCustomers.csv')
    return df

@st.cache_data
def perform_clustering(df, n_clusters=5):
    """Perform K-means clustering on Annual Income and Spending Score"""
    # Extract features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    return df_clustered, kmeans, X

@st.cache_data
def calculate_elbow_method(X, max_clusters=10):
    """Calculate WCSS for elbow method"""
    wcss = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    return K_range, wcss

def create_elbow_plot(K_range, wcss):
    """Create interactive elbow method plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(K_range), 
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#ff7f0e')
    ))
    
    fig.update_layout(
        title='Elbow Method for Optimal Number of Clusters',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Within-Cluster Sum of Squares (WCSS)',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_cluster_visualization(df_clustered):
    """Create interactive cluster visualization"""
    fig = px.scatter(
        df_clustered, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)',
        color='Cluster',
        title='Customer Segments based on Annual Income and Spending Score',
        labels={
            'Annual Income (k$)': 'Annual Income (k$)',
            'Spending Score (1-100)': 'Spending Score (1-100)'
        },
        color_continuous_scale='viridis',
        height=600
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(template='plotly_white')
    
    return fig

def analyze_clusters(df_clustered):
    """Analyze and provide insights for each cluster"""
    cluster_analysis = {}
    
    for cluster in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        
        analysis = {
            'count': len(cluster_data),
            'avg_income': cluster_data['Annual Income (k$)'].mean(),
            'avg_spending': cluster_data['Spending Score (1-100)'].mean(),
            'avg_age': cluster_data['Age'].mean(),
            'gender_distribution': cluster_data['Genre'].value_counts().to_dict(),
            'description': ''
        }
        
        # Generate cluster descriptions based on income and spending patterns
        if analysis['avg_income'] < 40 and analysis['avg_spending'] < 40:
            analysis['description'] = "Low Income, Low Spending - Budget-conscious customers"
        elif analysis['avg_income'] < 40 and analysis['avg_spending'] > 60:
            analysis['description'] = "Low Income, High Spending - Impulse buyers with limited budget"
        elif analysis['avg_income'] > 70 and analysis['avg_spending'] < 40:
            analysis['description'] = "High Income, Low Spending - Careful spenders with high earning potential"
        elif analysis['avg_income'] > 70 and analysis['avg_spending'] > 60:
            analysis['description'] = "High Income, High Spending - Premium customers"
        else:
            analysis['description'] = "Moderate Income, Moderate Spending - Average customers"
            
        cluster_analysis[cluster] = analysis
    
    return cluster_analysis

def create_demographic_charts(df_clustered):
    """Create demographic analysis charts"""
    # Age distribution by cluster
    fig_age = px.box(
        df_clustered, 
        x='Cluster', 
        y='Age',
        title='Age Distribution by Customer Segment',
        color='Cluster'
    )
    fig_age.update_layout(template='plotly_white', height=400)
    
    # Gender distribution by cluster
    gender_cluster = df_clustered.groupby(['Cluster', 'Genre']).size().reset_index(name='Count')
    fig_gender = px.bar(
        gender_cluster,
        x='Cluster',
        y='Count',
        color='Genre',
        title='Gender Distribution by Customer Segment',
        barmode='group'
    )
    fig_gender.update_layout(template='plotly_white', height=400)
    
    return fig_age, fig_gender

def main():
    st.title("🛍️ Customer Segmentation Dashboard")
    st.markdown("### Interactive Analysis of Mall Customer Data using K-Means Clustering")
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=5)
    show_elbow = st.sidebar.checkbox("Show Elbow Method Analysis", value=True)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Overview", 
        "📈 Elbow Method", 
        "🎯 Customer Segments", 
        "👥 Demographic Analysis", 
        "💡 Business Insights"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Average Age", f"{df['Age'].mean():.1f} years")
        with col3:
            st.metric("Average Income", f"${df['Annual Income (k$)'].mean():.1f}k")
        with col4:
            st.metric("Average Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")
        
        # Data preview
        st.subheader("Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_income_dist = px.histogram(
                df, x='Annual Income (k$)', 
                title='Distribution of Annual Income',
                nbins=20
            )
            fig_income_dist.update_layout(template='plotly_white')
            st.plotly_chart(fig_income_dist, use_container_width=True)
        
        with col2:
            fig_spending_dist = px.histogram(
                df, x='Spending Score (1-100)', 
                title='Distribution of Spending Score',
                nbins=20
            )
            fig_spending_dist.update_layout(template='plotly_white')
            st.plotly_chart(fig_spending_dist, use_container_width=True)
        
        # Gender and age analysis
        col1, col2 = st.columns(2)
        
        with col1:
            gender_counts = df['Genre'].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values, 
                names=gender_counts.index,
                title='Gender Distribution'
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            fig_age_dist = px.histogram(
                df, x='Age', color='Genre',
                title='Age Distribution by Gender',
                nbins=15
            )
            fig_age_dist.update_layout(template='plotly_white')
            st.plotly_chart(fig_age_dist, use_container_width=True)
    
    with tab2:
        st.header("Elbow Method Analysis")
        st.markdown("Determining the optimal number of clusters using the elbow method.")
        
        # Calculate elbow method
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        K_range, wcss = calculate_elbow_method(X)
        
        # Create elbow plot
        fig_elbow = create_elbow_plot(K_range, wcss)
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>How to interpret the Elbow Method:</strong>
        <ul>
        <li>The "elbow" point indicates the optimal number of clusters</li>
        <li>Look for the point where the rate of decrease in WCSS slows down significantly</li>
        <li>Based on this analysis, 5 clusters appears to be optimal for this dataset</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.header("Customer Segments")
        
        # Perform clustering
        df_clustered, kmeans, X = perform_clustering(df, n_clusters)
        
        # Main cluster visualization
        fig_clusters = create_cluster_visualization(df_clustered)
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        # Cluster analysis
        cluster_analysis = analyze_clusters(df_clustered)
        
        st.subheader("Cluster Characteristics")
        
        # Display cluster information in cards
        for cluster_id, analysis in cluster_analysis.items():
            with st.expander(f"Cluster {cluster_id}: {analysis['description']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Customer Count", analysis['count'])
                with col2:
                    st.metric("Avg Income", f"${analysis['avg_income']:.1f}k")
                with col3:
                    st.metric("Avg Spending Score", f"{analysis['avg_spending']:.1f}")
                with col4:
                    st.metric("Avg Age", f"{analysis['avg_age']:.1f} years")
                
                # Gender distribution
                gender_data = analysis['gender_distribution']
                if len(gender_data) > 0:
                    st.write("**Gender Distribution:**")
                    for gender, count in gender_data.items():
                        percentage = (count / analysis['count']) * 100
                        st.write(f"- {gender}: {count} customers ({percentage:.1f}%)")
    
    with tab4:
        st.header("Demographic Analysis")
        
        # Create demographic charts
        fig_age, fig_gender = create_demographic_charts(df_clustered)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Income vs Spending correlation
        st.subheader("Income vs Spending Correlation Analysis")
        correlation = df['Annual Income (k$)'].corr(df['Spending Score (1-100)'])
        
        fig_scatter = px.scatter(
            df, 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)',
            color='Genre',
            size='Age',
            title=f'Income vs Spending Score (Correlation: {correlation:.3f})',
            hover_data=['Age']
        )
        fig_scatter.update_layout(template='plotly_white', height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info(f"Correlation coefficient: {correlation:.3f} - This indicates a {'weak' if abs(correlation) < 0.3 else 'moderate' if abs(correlation) < 0.7 else 'strong'} correlation between income and spending.")
    
    with tab5:
        st.header("Business Insights & Recommendations")
        
        cluster_analysis = analyze_clusters(df_clustered)
        
        st.markdown("### Key Findings")
        
        # Generate insights based on cluster analysis
        st.markdown("""
        <div class="insight-box">
        <h4>Customer Segmentation Results:</h4>
        """, unsafe_allow_html=True)
        
        for cluster_id, analysis in cluster_analysis.items():
            st.markdown(f"""
            **Cluster {cluster_id}** ({analysis['count']} customers):
            - {analysis['description']}
            - Average Income: ${analysis['avg_income']:.1f}k
            - Average Spending Score: {analysis['avg_spending']:.1f}
            - Average Age: {analysis['avg_age']:.1f} years
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### Marketing Recommendations")
        
        recommendations = [
            "**High-Value Customers (High Income, High Spending)**: Focus on premium products and exclusive offers",
            "**Potential Customers (High Income, Low Spending)**: Target with personalized promotions to increase engagement",
            "**Budget Customers (Low Income, High Spending)**: Offer value deals and loyalty programs",
            "**Careful Spenders (Low Income, Low Spending)**: Focus on essential products with competitive pricing",
            "**Average Customers**: Standard marketing approach with seasonal campaigns"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
        
        st.markdown("### Model Performance")
        
        # Calculate silhouette score if scikit-learn version supports it
        try:
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(X, df_clustered['Cluster'])
            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            st.info("Silhouette Score ranges from -1 to 1. Higher values indicate better clustering quality.")
        except:
            st.info("Silhouette score calculation not available in this environment.")

if __name__ == "__main__":
    main()