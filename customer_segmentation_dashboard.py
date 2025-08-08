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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background-color: #ffffff;
        border: 2px solid #e1e8ed;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .segment-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-left: 4px solid #007bff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #212529;
    }
    .insight-box h4 {
        color: #007bff;
        margin-bottom: 1rem;
    }
    .control-section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stExpander > div > div > div {
        background-color: #ffffff;
    }
    .cluster-description {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the customer data"""
    df = pd.read_csv('data/MallCustomers.csv')
    # Rename Genre column to Gender for better clarity
    df = df.rename(columns={'Genre': 'Gender'})
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
            'gender_distribution': cluster_data['Gender'].value_counts().to_dict(),
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
    gender_cluster = df_clustered.groupby(['Cluster', 'Gender']).size().reset_index(name='Count')
    fig_gender = px.bar(
        gender_cluster,
        x='Cluster',
        y='Count',
        color='Gender',
        title='Gender Distribution by Customer Segment',
        barmode='group'
    )
    fig_gender.update_layout(template='plotly_white', height=400)
    
    return fig_age, fig_gender

def main():
    st.markdown("""
    <div class="main-header">
        <h1>Customer Segmentation Dashboard</h1>
        <p>Interactive Analysis of Mall Customer Data using K-Means Clustering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dataset Overview", 
        "Elbow Method", 
        "Customer Segments", 
        "Demographic Analysis", 
        "Business Insights"
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
            gender_counts = df['Gender'].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values, 
                names=gender_counts.index,
                title='Gender Distribution'
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            fig_age_dist = px.histogram(
                df, x='Age', color='Gender',
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
        
        # Controls for this tab
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Clustering Controls**")
        with col2:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=5, key="cluster_slider")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Perform clustering
        df_clustered, kmeans, X = perform_clustering(df, n_clusters)
        
        # Methodology explanation
        st.markdown("""
        <div class="insight-box">
        <h4>Customer Segmentation Methodology</h4>
        <p>This analysis uses <strong>K-Means clustering</strong> to segment customers based on two key behavioral indicators:</p>
        <ul>
        <li><strong>Annual Income (k$):</strong> Customer's purchasing power and economic capacity</li>
        <li><strong>Spending Score (1-100):</strong> Customer's propensity to spend in the mall environment</li>
        </ul>
        <p>The clustering algorithm groups customers with similar income and spending patterns, revealing distinct market segments for targeted marketing strategies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main cluster visualization
        fig_clusters = create_cluster_visualization(df_clustered)
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        # Cluster analysis
        cluster_analysis = analyze_clusters(df_clustered)
        
        st.subheader("Cluster Characteristics")
        st.markdown("""
        <div class="insight-box">
        <h4>Understanding the Metrics</h4>
        <ul>
        <li><strong>Customer Count:</strong> Total number of customers in each segment</li>
        <li><strong>Average Income:</strong> Mean annual income of customers in the cluster (in thousands)</li>
        <li><strong>Average Spending Score:</strong> Mean propensity to spend (scale 1-100, higher = more likely to spend)</li>
        <li><strong>Average Age:</strong> Mean age of customers in the segment</li>
        <li><strong>Gender Distribution:</strong> Breakdown of male/female customers per cluster</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display cluster information in cards
        for cluster_id, analysis in cluster_analysis.items():
            with st.expander(f"Cluster {cluster_id}: {analysis['description']}", expanded=True):
                st.markdown(f"""
                <div class="cluster-description">
                    <strong>Cluster {cluster_id} Profile:</strong> {analysis['description']}
                </div>
                """, unsafe_allow_html=True)
                
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
                    st.markdown("**Gender Distribution:**")
                    for gender, count in gender_data.items():
                        percentage = (count / analysis['count']) * 100
                        st.write(f"• {gender}: {count} customers ({percentage:.1f}%)")
    
    with tab4:
        st.header("Demographic Analysis")
        
        # Use the same clustering settings from tab3
        n_clusters = 5  # Default value
        if 'cluster_slider' in st.session_state:
            n_clusters = st.session_state.cluster_slider
        
        df_clustered, kmeans, X = perform_clustering(df, n_clusters)
        
        # Create demographic charts
        fig_age, fig_gender = create_demographic_charts(df_clustered)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Income vs Spending correlation with clustering
        st.subheader("Income vs Spending Correlation Analysis")
        correlation = df_clustered['Annual Income (k$)'].corr(df_clustered['Spending Score (1-100)'])
        
        # Create a more detailed scatter plot with cluster information
        fig_scatter = px.scatter(
            df_clustered, 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)',
            color='Cluster',
            size='Age',
            title=f'Income vs Spending Score by Customer Segment (Correlation: {correlation:.3f})',
            hover_data=['Age', 'Gender', 'Cluster'],
            color_continuous_scale='viridis'
        )
        fig_scatter.update_layout(template='plotly_white', height=500)
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color='white')))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation strength interpretation
        strength = 'weak' if abs(correlation) < 0.3 else 'moderate' if abs(correlation) < 0.7 else 'strong'
        st.info(f"Correlation coefficient: {correlation:.3f} - This indicates a {strength} correlation between income and spending patterns.")
        
        # Geographic Distribution Simulation
        st.subheader("Geographic Distribution Analysis")
        st.info("Note: Geographic data is simulated for demonstration as the original dataset doesn't include location information.")
        
        # Create simulated geographic data based on clusters
        np.random.seed(42)
        locations = {
            'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
            'State': ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA']
        }
        
        # Assign locations based on cluster characteristics
        location_data = []
        for _, row in df_clustered.iterrows():
            # Higher income clusters more likely in expensive cities
            if row['Annual Income (k$)'] > 70:
                city_weights = [0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.1, 0.05, 0.1]
            elif row['Annual Income (k$)'] > 40:
                city_weights = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            else:
                city_weights = [0.1, 0.1, 0.1, 0.2, 0.15, 0.1, 0.15, 0.05, 0.15, 0.05]
            
            # Normalize weights to ensure they sum to 1
            city_weights = np.array(city_weights)
            city_weights = city_weights / city_weights.sum()
            
            city_idx = np.random.choice(len(locations['City']), p=city_weights)
            location_data.append({
                'City': locations['City'][city_idx],
                'State': locations['State'][city_idx],
                'Cluster': row['Cluster'],
                'Income': row['Annual Income (k$)'],
                'Spending': row['Spending Score (1-100)']
            })
        
        geo_df = pd.DataFrame(location_data)
        
        # Create geographic distribution chart
        city_cluster_dist = geo_df.groupby(['City', 'Cluster']).size().reset_index(name='Count')
        
        fig_geo = px.bar(
            city_cluster_dist,
            x='City',
            y='Count',
            color='Cluster',
            title='Customer Segment Distribution by City',
            color_continuous_scale='viridis'
        )
        fig_geo.update_layout(
            template='plotly_white',
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_geo, use_container_width=True)
        
        # Summary statistics by location
        col1, col2 = st.columns(2)
        
        with col1:
            city_summary = geo_df.groupby('City').agg({
                'Income': 'mean',
                'Spending': 'mean',
                'Cluster': 'count'
            }).round(1).rename(columns={'Cluster': 'Customer Count'})
            
            st.write("**Average Metrics by City:**")
            st.dataframe(city_summary, use_container_width=True)
        
        with col2:
            cluster_by_state = geo_df.groupby(['State', 'Cluster']).size().reset_index(name='Count')
            fig_state = px.pie(
                cluster_by_state,
                values='Count',
                names='State',
                title='Customer Distribution by State'
            )
            fig_state.update_layout(height=300)
            st.plotly_chart(fig_state, use_container_width=True)
    
    with tab5:
        st.header("Business Insights & Recommendations")
        
        # Use the same clustering settings from tab3
        n_clusters = 5  # Default value
        if 'cluster_slider' in st.session_state:
            n_clusters = st.session_state.cluster_slider
        
        df_clustered, kmeans, X = perform_clustering(df, n_clusters)
        cluster_analysis = analyze_clusters(df_clustered)
        
        st.markdown("### Key Findings")
        
        # Create cluster summary table
        cluster_summary = []
        for cluster_id, analysis in cluster_analysis.items():
            cluster_summary.append({
                'Cluster': f"Cluster {cluster_id}",
                'Description': analysis['description'],
                'Count': analysis['count'],
                'Avg Income ($k)': f"{analysis['avg_income']:.1f}",
                'Avg Spending Score': f"{analysis['avg_spending']:.1f}",
                'Avg Age': f"{analysis['avg_age']:.1f}",
                'Primary Gender': max(analysis['gender_distribution'], key=analysis['gender_distribution'].get) if analysis['gender_distribution'] else 'N/A'
            })
        
        cluster_df = pd.DataFrame(cluster_summary)
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        
        # Customer Segmentation Results in insight box
        st.markdown("""
        <div class="insight-box">
        <h4>Customer Segmentation Results:</h4>
        <p>The K-means clustering analysis reveals distinct customer segments with varying spending behaviors:</p>
        <ul>
        <li><strong>Premium Segment:</strong> High-income customers with strong spending power represent the most valuable segment</li>
        <li><strong>Conservative Segment:</strong> High earners with low spending indicate untapped potential for targeted marketing</li>
        <li><strong>Impulse Buyers:</strong> Lower income but high spending customers show strong engagement despite budget constraints</li>
        <li><strong>Budget Conscious:</strong> Low income, low spending customers require value-focused approaches</li>
        <li><strong>Balanced Customers:</strong> Moderate income and spending patterns represent the stable customer base</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Marketing Recommendations")
        
        st.markdown("""
        <div class="insight-box">
        <h4>Marketing Recommendations:</h4>
        <p>Targeted strategies for each customer segment to maximize engagement and revenue:</p>
        <ul>
        <li><strong>Premium Customers:</strong> Launch exclusive membership programs, premium product lines, and VIP experiences to maintain loyalty</li>
        <li><strong>High Earners/Low Spenders:</strong> Implement personalized email campaigns, limited-time offers, and lifestyle-based marketing to convert potential</li>
        <li><strong>Budget Shoppers:</strong> Create loyalty reward programs, bulk purchase discounts, and seasonal sales to increase purchase frequency</li>
        <li><strong>Price-Sensitive Customers:</strong> Focus on essential items, competitive pricing, and value bundles to build long-term relationships</li>
        <li><strong>Moderate Customers:</strong> Develop balanced marketing mix with regular promotions and product recommendations based on purchase history</li>
        </ul>
        <p><strong>Key Strategy:</strong> Implement dynamic pricing and personalized recommendations based on customer segment classification to optimize revenue across all groups.</p>
        </div>
        """, unsafe_allow_html=True)
        
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