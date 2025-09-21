import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta

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
st.markdown("Comprehensive analytics for marketing campaign performance")
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

# Reset filters button
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

# Main dashboard
if filtered_df.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust your selection.")
else:
    # KPIs
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_conversion = filtered_df['Conversion_Rate'].mean()
        delta_conversion = avg_conversion - df['Conversion_Rate'].mea