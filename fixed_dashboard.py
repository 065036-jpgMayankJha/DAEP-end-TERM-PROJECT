import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Marketing Campaign Dashboard", 
    layout="wide",
    page_icon="📊"
)

# Load data function with error handling
@st.cache_data
def load_data():
    """Load and preprocess the marketing campaign data"""
    try:
        # Try multiple possible file paths
        possible_paths = [
            "marketing_campaign_dataset.csv",
            "data/marketing_campaign_dataset.csv",
            "./marketing_campaign_dataset.csv"
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            # If no file found, create sample data for demo
            st.warning("Dataset file not found. Using sample data for demonstration.")
            df = create_sample_data()
        
        # Data preprocessing
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        # Clean Acquisition_Cost column if it exists
        if "Acquisition_Cost" in df.columns:
            if df["Acquisition_Cost"].dtype == 'object':
                df["Acquisition_Cost"] = (df["Acquisition_Cost"]
                                        .astype(str)
                                        .str.replace("$", "", regex=False)
                                        .str.replace(",", "", regex=False)
                                        .replace("nan", "0")
                                        .astype(float))
        
        # Ensure required columns exist
        required_columns = ["Company", "Campaign_Type", "Channel_Used", "Location", 
                          "Conversion_Rate", "ROI", "Clicks", "Impressions"]
        
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return create_sample_data()
        
        # Convert percentage strings to float if needed
        if df["Conversion_Rate"].dtype == 'object':
            df["Conversion_Rate"] = (df["Conversion_Rate"]
                                   .astype(str)
                                   .str.replace("%", "", regex=False)
                                   .astype(float) / 100)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    n_rows = 1000
    
    companies = ["TechCorp", "MarketPro", "DataDriven", "GrowthCo", "AnalyticsFirm"]
    campaign_types = ["Social Media", "Email", "Display", "Search", "Content Marketing"]
    channels = ["Facebook", "Google Ads", "Instagram", "LinkedIn", "Twitter", "Email"]
    locations = ["New York", "California", "Texas", "Florida", "Illinois"]
    
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=x) for x in range(365)]
    
    data = {
        "Date": np.random.choice(dates, n_rows),
        "Company": np.random.choice(companies, n_rows),
        "Campaign_Type": np.random.choice(campaign_types, n_rows),
        "Channel_Used": np.random.choice(channels, n_rows),
        "Location": np.random.choice(locations, n_rows),
        "Conversion_Rate": np.random.normal(0.05, 0.02, n_rows).clip(0, 1),
        "ROI": np.random.normal(2.5, 1.0, n_rows).clip(0.5, 10),
        "Clicks": np.random.randint(100, 10000, n_rows),
        "Impressions": np.random.randint(1000, 100000, n_rows),
        "Acquisition_Cost": np.random.normal(50, 20, n_rows).clip(10, 200)
    }
    
    return pd.DataFrame(data)

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Dashboard header
st.title("📊 Marketing Campaign Dashboard")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")

# Get unique values safely
def get_unique_values(column):
    try:
        return sorted(df[column].dropna().unique())
    except:
        return []

company = st.sidebar.multiselect(
    "Select Company", 
    options=get_unique_values("Company"),
    default=[]
)

campaign_type = st.sidebar.multiselect(
    "Select Campaign Type", 
    options=get_unique_values("Campaign_Type"),
    default=[]
)

channel = st.sidebar.multiselect(
    "Select Channel", 
    options=get_unique_values("Channel_Used"),
    default=[]
)

location = st.sidebar.multiselect(
    "Select Location", 
    options=get_unique_values("Location"),
    default=[]
)

# Apply filters
filtered_df = df.copy()

if company:
    filtered_df = filtered_df[filtered_df["Company"].isin(company)]
if campaign_type:
    filtered_df = filtered_df[filtered_df["Campaign_Type"].isin(campaign_type)]
if channel:
    filtered_df = filtered_df[filtered_df["Channel_Used"].isin(channel)]
if location:
    filtered_df = filtered_df[filtered_df["Location"].isin(location)]

# Display filter info
st.sidebar.markdown("---")
st.sidebar.info(f"Showing {len(filtered_df):,} of {len(df):,} records")

# Main dashboard
if filtered_df.empty:
    st.warning("No data matches your current filters. Please adjust your selection.")
else:
    # KPIs
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_conversion = filtered_df['Conversion_Rate'].mean()
        st.metric(
            "Avg Conversion Rate", 
            f"{avg_conversion:.2%}",
            delta=f"{(avg_conversion - df['Conversion_Rate'].mean()):.2%}"
        )
    
    with col2:
        avg_roi = filtered_df['ROI'].mean()
        st.metric(
            "Avg ROI", 
            f"{avg_roi:.2f}x",
            delta=f"{(avg_roi - df['ROI'].mean()):.2f}x"
        )
    
    with col3:
        total_clicks = filtered_df['Clicks'].sum()
        st.metric("Total Clicks", f"{total_clicks:,}")
    
    with col4:
        total_impressions = filtered_df['Impressions'].sum()
        st.metric("Total Impressions", f"{total_impressions:,}")

    # Charts section
    st.markdown("---")
    st.subheader("📊 Visualizations")
    
    # Top row charts
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            channel_data = (filtered_df.groupby("Channel_Used")["Conversion_Rate"]
                          .mean()
                          .reset_index()
                          .sort_values("Conversion_Rate", ascending=False))
            
            fig1 = px.bar(
                channel_data,
                x="Channel_Used", 
                y="Conversion_Rate", 
                title="Average Conversion Rate by Channel",
                labels={"Conversion_Rate": "Conversion Rate", "Channel_Used": "Channel"},
                color="Conversion_Rate",
                color_continuous_scale="viridis"
            )
            fig1.update_layout(showlegend=False)
            fig1.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating channel chart: {str(e)}")

    with col2:
        try:
            campaign_data = (filtered_df.groupby("Campaign_Type")["ROI"]
                           .mean()
                           .reset_index()
                           .sort_values("ROI", ascending=False))
            
            fig2 = px.bar(
                campaign_data,
                x="Campaign_Type", 
                y="ROI", 
                title="Average ROI by Campaign Type",
                labels={"ROI": "ROI (x)", "Campaign_Type": "Campaign Type"},
                color="ROI",
                color_continuous_scale="plasma"
            )
            fig2.update_layout(showlegend=False)
            fig2.update_traces(texttemplate='%{y:.1f}x', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating campaign chart: {str(e)}")

    # Time series chart
    if "Date" in filtered_df.columns and not filtered_df["Date"].isna().all():
        try:
            time_data = (filtered_df.groupby(filtered_df["Date"].dt.date)["Conversion_Rate"]
                        .mean()
                        .reset_index())
            time_data.columns = ["Date", "Conversion_Rate"]
            
            fig3 = px.line(
                time_data,
                x="Date", 
                y="Conversion_Rate", 
                title="Conversion Rate Trend Over Time",
                labels={"Conversion_Rate": "Conversion Rate", "Date": "Date"}
            )
            fig3.update_traces(line_color='#00cc96', line_width=3)
            fig3.update_layout(hovermode='x unified')
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating time series chart: {str(e)}")

    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Top performing companies
            company_performance = (filtered_df.groupby("Company")
                                 .agg({"Conversion_Rate": "mean", "ROI": "mean"})
                                 .round(3)
                                 .sort_values("Conversion_Rate", ascending=False))
            
            st.subheader("🏆 Top Performing Companies")
            st.dataframe(company_performance.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating company performance table: {str(e)}")
    
    with col2:
        try:
            # Campaign efficiency
            if "Acquisition_Cost" in filtered_df.columns:
                efficiency_data = filtered_df.copy()
                efficiency_data["Cost_per_Conversion"] = (
                    efficiency_data["Acquisition_Cost"] / 
                    efficiency_data["Conversion_Rate"].replace(0, 1)
                )
                
                fig4 = px.scatter(
                    efficiency_data.head(100),  # Limit points for better performance
                    x="Acquisition_Cost",
                    y="ROI",
                    size="Clicks",
                    color="Campaign_Type",
                    title="Campaign Efficiency: Cost vs ROI",
                    labels={"Acquisition_Cost": "Acquisition Cost ($)", "ROI": "ROI (x)"}
                )
                st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating efficiency chart: {str(e)}")

    # Data table
    st.markdown("---")
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            filtered_df.round(3),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"marketing_campaign_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data refreshed automatically")
