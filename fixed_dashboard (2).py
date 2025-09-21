import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Marketing Campaign Dashboard", 
    layout="wide",
    page_icon="📊"
)

# Load data function with robust error handling
@st.cache_data
def load_data():
    """Load and preprocess the marketing campaign data"""
    try:
        # Try multiple possible file paths
        possible_paths = [
            "marketing_campaign_dataset.csv",
            "data/marketing_campaign_dataset.csv",
            "./marketing_campaign_dataset.csv",
            "marketing_campaign.csv",
            "campaign_data.csv"
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Data loaded from: {path}")
                break
        
        if df is None:
            st.warning("📁 Dataset file not found. Using sample data for demonstration.")
            df = create_sample_data()
            st.info("You can upload your marketing_campaign_dataset.csv file to the same directory as this script.")
        
        # Display original columns for debugging
        st.sidebar.write("**Original Columns:**")
        for i, col in enumerate(df.columns):
            st.sidebar.write(f"{i+1}. {col}")
        
        # Preprocess data based on common column patterns
        df = preprocess_dataframe(df)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return create_sample_data()

def preprocess_dataframe(df):
    """Preprocess the dataframe with flexible column mapping"""
    
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Common column mappings (handle different naming conventions)
    column_mappings = {
        # Date columns
        'date': 'Date',
        'campaign_date': 'Date',
        'start_date': 'Date',
        'Date': 'Date',
        
        # Company/Business columns
        'company': 'Company',
        'business': 'Company',
        'advertiser': 'Company',
        'client': 'Company',
        'Company': 'Company',
        
        # Campaign type
        'campaign_type': 'Campaign_Type',
        'type': 'Campaign_Type',
        'campaign': 'Campaign_Type',
        'Campaign_Type': 'Campaign_Type',
        
        # Channel
        'channel': 'Channel_Used',
        'channel_used': 'Channel_Used',
        'platform': 'Channel_Used',
        'source': 'Channel_Used',
        'Channel_Used': 'Channel_Used',
        
        # Location
        'location': 'Location',
        'region': 'Location',
        'country': 'Location',
        'city': 'Location',
        'Location': 'Location',
        
        # Metrics
        'conversion_rate': 'Conversion_Rate',
        'conv_rate': 'Conversion_Rate',
        'Conversion_Rate': 'Conversion_Rate',
        
        'roi': 'ROI',
        'return_on_investment': 'ROI',
        'ROI': 'ROI',
        
        'clicks': 'Clicks',
        'click': 'Clicks',
        'Clicks': 'Clicks',
        
        'impressions': 'Impressions',
        'impression': 'Impressions',
        'views': 'Impressions',
        'Impressions': 'Impressions',
        
        'cost': 'Acquisition_Cost',
        'acquisition_cost': 'Acquisition_Cost',
        'spend': 'Acquisition_Cost',
        'budget': 'Acquisition_Cost',
        'Acquisition_Cost': 'Acquisition_Cost'
    }
    
    # Apply column mappings
    for original_col in df.columns:
        # Check for exact matches first
        if original_col.lower() in column_mappings:
            new_name = column_mappings[original_col.lower()]
            if original_col != new_name:
                processed_df = processed_df.rename(columns={original_col: new_name})
        
        # Check for partial matches
        elif any(key in original_col.lower() for key in column_mappings.keys()):
            for key, new_name in column_mappings.items():
                if key in original_col.lower():
                    processed_df = processed_df.rename(columns={original_col: new_name})
                    break
    
    # Data cleaning
    try:
        # Handle Date column
        date_columns = [col for col in processed_df.columns if 'date' in col.lower() or col == 'Date']
        if date_columns:
            for date_col in date_columns:
                processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
                if date_col != 'Date':
                    processed_df['Date'] = processed_df[date_col]
    except:
        pass
    
    # Clean monetary columns
    monetary_cols = [col for col in processed_df.columns if 
                    any(term in col.lower() for term in ['cost', 'spend', 'budget', 'revenue'])]
    
    for col in monetary_cols:
        try:
            if processed_df[col].dtype == 'object':
                processed_df[col] = (processed_df[col].astype(str)
                                   .str.replace('$', '', regex=False)
                                   .str.replace(',', '', regex=False)
                                   .str.replace('€', '', regex=False)
                                   .str.replace('£', '', regex=False))
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        except:
            pass
    
    # Clean percentage columns
    percentage_cols = [col for col in processed_df.columns if 
                      any(term in col.lower() for term in ['rate', 'percentage', '%'])]
    
    for col in percentage_cols:
        try:
            if processed_df[col].dtype == 'object':
                processed_df[col] = (processed_df[col].astype(str)
                                   .str.replace('%', '', regex=False))
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                # If values are greater than 1, assume they're in percentage format
                if processed_df[col].max() > 1:
                    processed_df[col] = processed_df[col] / 100
        except:
            pass
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['ROI', 'Clicks', 'Impressions', 'Conversion_Rate', 'Acquisition_Cost']
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Fill missing values
    processed_df = processed_df.fillna({
        'Company': 'Unknown Company',
        'Campaign_Type': 'Unknown Campaign',
        'Channel_Used': 'Unknown Channel',
        'Location': 'Unknown Location'
    })
    
    return processed_df

def create_sample_data():
    """Create sample data matching typical marketing campaign structure"""
    np.random.seed(42)
    n_rows = 1500
    
    companies = ["TechCorp", "MarketPro", "DataDriven", "GrowthCo", "AnalyticsFirm", "InnovateCorp"]
    campaign_types = ["Social Media", "Email Marketing", "Display Ads", "Search Engine", "Content Marketing", "Video Campaign"]
    channels = ["Facebook", "Google Ads", "Instagram", "LinkedIn", "Twitter", "Email", "YouTube"]
    locations = ["New York", "California", "Texas", "Florida", "Illinois", "London", "Toronto"]
    
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=x) for x in range(365)]
    
    data = {
        "Date": np.random.choice(dates, n_rows),
        "Company": np.random.choice(companies, n_rows),
        "Campaign_Type": np.random.choice(campaign_types, n_rows),
        "Channel_Used": np.random.choice(channels, n_rows),
        "Location": np.random.choice(locations, n_rows),
        "Impressions": np.random.randint(1000, 100000, n_rows),
        "Clicks": np.random.randint(50, 5000, n_rows),
        "Conversion_Rate": np.random.normal(0.05, 0.02, n_rows).clip(0.001, 0.25),
        "ROI": np.random.normal(2.5, 1.2, n_rows).clip(0.1, 8.0),
        "Acquisition_Cost": np.random.normal(75, 25, n_rows).clip(10, 500)
    }
    
    df = pd.DataFrame(data)
    
    # Add derived metrics
    df['CTR'] = (df['Clicks'] / df['Impressions']).clip(0, 1)
    df['Conversions'] = (df['Clicks'] * df['Conversion_Rate']).round().astype(int)
    df['Revenue'] = df['Conversions'] * np.random.normal(150, 50, n_rows).clip(50, 500)
    
    return df

def get_safe_column_values(df, column, default_message="No data available"):
    """Safely get unique values from a column"""
    try:
        if column in df.columns:
            return sorted(df[column].dropna().unique())
        else:
            return []
    except:
        return []

def safe_calculate_metric(df, calculation_func, default_value=0):
    """Safely calculate metrics with error handling"""
    try:
        return calculation_func(df)
    except:
        return default_value

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {str(e)}")
    st.stop()

# Dashboard header
st.title("📊 Marketing Campaign Dashboard")
st.markdown("**Comprehensive analytics for marketing campaign performance**")
st.markdown("---")

# Show data info
with st.expander("📋 Dataset Information"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Date Range:** {df['Date'].min().strftime('%Y-%m-%d') if 'Date' in df.columns and not df['Date'].isna().all() else 'N/A'} to {df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns and not df['Date'].isna().all() else 'N/A'}")
    with col2:
        st.write(f"**Columns:** {len(df.columns)}")
        available_metrics = [col for col in ['Impressions', 'Clicks', 'Conversion_Rate', 'ROI'] if col in df.columns]
        st.write(f"**Available Metrics:** {', '.join(available_metrics)}")

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")

# Dynamic filter creation based on available columns
filters = {}

if 'Company' in df.columns:
    company_options = get_safe_column_values(df, 'Company')
    filters['Company'] = st.sidebar.multiselect("Select Company", options=company_options, default=[])

if 'Campaign_Type' in df.columns:
    campaign_options = get_safe_column_values(df, 'Campaign_Type')
    filters['Campaign_Type'] = st.sidebar.multiselect("Select Campaign Type", options=campaign_options, default=[])

if 'Channel_Used' in df.columns:
    channel_options = get_safe_column_values(df, 'Channel_Used')
    filters['Channel_Used'] = st.sidebar.multiselect("Select Channel", options=channel_options, default=[])

if 'Location' in df.columns:
    location_options = get_safe_column_values(df, 'Location')
    filters['Location'] = st.sidebar.multiselect("Select Location", options=location_options, default=[])

# Apply filters
filtered_df = df.copy()
for column, selected_values in filters.items():
    if selected_values and column in df.columns:
        filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

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
    # Dynamic KPIs based on available columns
    st.subheader("📈 Key Performance Indicators")
    
    # First row of KPIs
    kpi_cols = st.columns(4)
    kpi_index = 0
    
    # Conversion Rate
    if 'Conversion_Rate' in filtered_df.columns:
        with kpi_cols[kpi_index % 4]:
            avg_conversion = safe_calculate_metric(filtered_df, lambda x: x['Conversion_Rate'].mean())
            delta_conversion = avg_conversion - safe_calculate_metric(df, lambda x: x['Conversion_Rate'].mean())
            st.metric("Avg Conversion Rate", f"{avg_conversion:.2%}", delta=f"{delta_conversion:.2%}")
        kpi_index += 1
    
    # ROI
    if 'ROI' in filtered_df.columns:
        with kpi_cols[kpi_index % 4]:
            avg_roi = safe_calculate_metric(filtered_df, lambda x: x['ROI'].mean())
            delta_roi = avg_roi - safe_calculate_metric(df, lambda x: x['ROI'].mean())
            st.metric("Avg ROI", f"{avg_roi:.2f}x", delta=f"{delta_roi:.2f}x")
        kpi_index += 1
    
    # Total Clicks
    if 'Clicks' in filtered_df.columns:
        with kpi_cols[kpi_index % 4]:
            total_clicks = safe_calculate_metric(filtered_df, lambda x: x['Clicks'].sum())
            st.metric("Total Clicks", f"{total_clicks:,}")
        kpi_index += 1
    
    # Total Impressions
    if 'Impressions' in filtered_df.columns:
        with kpi_cols[kpi_index % 4]:
            total_impressions = safe_calculate_metric(filtered_df, lambda x: x['Impressions'].sum())
            st.metric("Total Impressions", f"{total_impressions:,}")
        kpi_index += 1
    
    # Second row of KPIs if we have more metrics
    if kpi_index >= 4 or any(col in filtered_df.columns for col in ['CTR', 'Acquisition_Cost', 'Revenue', 'Conversions']):
        kpi_cols_2 = st.columns(4)
        kpi_index_2 = 0
        
        # CTR
        if 'CTR' in filtered_df.columns or ('Clicks' in filtered_df.columns and 'Impressions' in filtered_df.columns):
            with kpi_cols_2[kpi_index_2 % 4]:
                if 'CTR' in filtered_df.columns:
                    avg_ctr = safe_calculate_metric(filtered_df, lambda x: x['CTR'].mean() * 100)
                else:
                    total_clicks = safe_calculate_metric(filtered_df, lambda x: x['Clicks'].sum())
                    total_impressions = safe_calculate_metric(filtered_df, lambda x: x['Impressions'].sum())
                    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                st.metric("Click-Through Rate", f"{avg_ctr:.2f}%")
            kpi_index_2 += 1
        
        # Acquisition Cost
        if 'Acquisition_Cost' in filtered_df.columns:
            with kpi_cols_2[kpi_index_2 % 4]:
                avg_cost = safe_calculate_metric(filtered_df, lambda x: x['Acquisition_Cost'].mean())
                st.metric("Avg Acquisition Cost", f"${avg_cost:.2f}")
            kpi_index_2 += 1
        
        # Revenue
        if 'Revenue' in filtered_df.columns:
            with kpi_cols_2[kpi_index_2 % 4]:
                total_revenue = safe_calculate_metric(filtered_df, lambda x: x['Revenue'].sum())
                st.metric("Total Revenue", f"${total_revenue:,.2f}")
            kpi_index_2 += 1
        
        # Conversions
        if 'Conversions' in filtered_df.columns:
            with kpi_cols_2[kpi_index_2 % 4]:
                total_conversions = safe_calculate_metric(filtered_df, lambda x: x['Conversions'].sum())
                st.metric("Total Conversions", f"{total_conversions:,}")
            kpi_index_2 += 1

    # Charts section
    st.markdown("---")
    st.subheader("📊 Performance Visualizations")
    
    # Dynamic chart generation based on available data
    chart_cols = st.columns(2)
    
    # Chart 1: Conversion Rate by Channel
    if 'Channel_Used' in filtered_df.columns and 'Conversion_Rate' in filtered_df.columns:
        with chart_cols[0]:
            try:
                st.write("**Average Conversion Rate by Channel**")
                channel_data = (filtered_df.groupby("Channel_Used")["Conversion_Rate"]
                              .mean()
                              .sort_values(ascending=True))
                
                if not channel_data.empty:
                    st.bar_chart(channel_data)
                    best_channel = channel_data.index[-1]
                    best_rate = channel_data.iloc[-1]
                    st.success(f"🏆 Best Channel: **{best_channel}** ({best_rate:.2%})")
                else:
                    st.info("No channel data available")
                    
            except Exception as e:
                st.error(f"Error creating channel chart: {str(e)}")
    
    # Chart 2: ROI by Campaign Type
    if 'Campaign_Type' in filtered_df.columns and 'ROI' in filtered_df.columns:
        with chart_cols[1]:
            try:
                st.write("**Average ROI by Campaign Type**")
                campaign_data = (filtered_df.groupby("Campaign_Type")["ROI"]
                               .mean()
                               .sort_values(ascending=True))
                
                if not campaign_data.empty:
                    st.bar_chart(campaign_data)
                    best_campaign = campaign_data.index[-1]
                    best_roi = campaign_data.iloc[-1]
                    st.success(f"🏆 Best Campaign: **{best_campaign}** ({best_roi:.2f}x)")
                else:
                    st.info("No campaign data available")
                    
            except Exception as e:
                st.error(f"Error creating campaign chart: {str(e)}")
    
    # Time series chart
    if "Date" in filtered_df.columns and 'Conversion_Rate' in filtered_df.columns and not filtered_df["Date"].isna().all():
        try:
            st.write("**Conversion Rate Trend Over Time**")
            time_data = (filtered_df.groupby(filtered_df["Date"].dt.date)["Conversion_Rate"]
                        .mean())
            
            if not time_data.empty and len(time_data) > 1:
                st.line_chart(time_data)
                
                # Trend analysis
                recent_avg = time_data.tail(7).mean() if len(time_data) >= 7 else time_data.mean()
                overall_avg = time_data.mean()
                if recent_avg > overall_avg:
                    st.success(f"📈 Recent trend is positive! Last 7 days avg: {recent_avg:.2%}")
                else:
                    st.info(f"📊 Recent 7 days avg: {recent_avg:.2%} vs overall avg: {overall_avg:.2%}")
            else:
                st.info("Insufficient time series data for trending.")
                
        except Exception as e:
            st.error(f"Error creating time series chart: {str(e)}")
    
    # Performance insights
    st.markdown("---")
    st.subheader("🔍 Performance Insights")
    
    insight_cols = st.columns(3)
    
    # Top Performers
    with insight_cols[0]:
        st.write("**🏆 Top Performers**")
        try:
            if 'ROI' in filtered_df.columns and 'Company' in filtered_df.columns:
                top_performers = (filtered_df.groupby('Company')
                                .agg({'ROI': 'mean', 'Conversion_Rate': 'mean'})
                                .round(3)
                                .sort_values('ROI', ascending=False)
                                .head(5))
                
                for company, metrics in top_performers.iterrows():
                    roi_val = metrics.get('ROI', 0)
                    conv_val = metrics.get('Conversion_Rate', 0)
                    st.write(f"• **{company}**")
                    st.write(f"  ROI: {roi_val:.2f}x, Conv: {conv_val:.2%}")
            else:
                st.write("ROI or Company data not available")
        except Exception as e:
            st.write(f"Error: {str(e)}")
    
    # Channel Efficiency
    with insight_cols[1]:
        st.write("**⚡ Channel Efficiency**")
        try:
            if all(col in filtered_df.columns for col in ['Channel_Used', 'Clicks', 'Impressions', 'Conversion_Rate']):
                channel_efficiency = filtered_df.groupby('Channel_Used').agg({
                    'Clicks': 'sum',
                    'Impressions': 'sum',
                    'Conversion_Rate': 'mean'
                }).round(3)
                
                channel_efficiency['CTR'] = (channel_efficiency['Clicks'] / 
                                           channel_efficiency['Impressions'] * 100).round(2)
                
                for channel, metrics in channel_efficiency.head(5).iterrows():
                    ctr_val = metrics.get('CTR', 0)
                    conv_val = metrics.get('Conversion_Rate', 0)
                    st.write(f"• **{channel}**")
                    st.write(f"  CTR: {ctr_val:.2f}%, Conv: {conv_val:.2%}")
            else:
                st.write("Channel efficiency data not available")
        except Exception as e:
            st.write(f"Error: {str(e)}")
    
    # Quick Stats
    with insight_cols[2]:
        st.write("**💡 Quick Stats**")
        try:
            total_companies = filtered_df['Company'].nunique() if 'Company' in filtered_df.columns else 0
            total_campaigns = len(filtered_df)
            avg_clicks = filtered_df['Clicks'].mean() if 'Clicks' in filtered_df.columns else 0
            most_used_channel = filtered_df['Channel_Used'].mode().iloc[0] if 'Channel_Used' in filtered_df.columns and not filtered_df.empty else 'N/A'
            
            st.write(f"• **{total_companies}** companies tracked")
            st.write(f"• **{total_campaigns:,}** total campaigns")
            st.write(f"• **{avg_clicks:,.0f}** avg clicks/campaign")
            st.write(f"• **{most_used_channel}** most used channel")
        except Exception as e:
            st.write(f"Error: {str(e)}")

    # Data table
    st.markdown("---")
    st.subheader("📋 Detailed Data View")
    
    with st.expander("🔍 View Data Table", expanded=False):
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_rows = st.selectbox("Rows to show:", [10, 25, 50, 100], index=1)
        with col2:
            sort_options = [col for col in filtered_df.columns if filtered_df[col].dtype in ['int64', 'float64']]
            if sort_options:
                sort_column = st.selectbox("Sort by:", sort_options)
            else:
                sort_column = filtered_df.columns[0]
        
        # Display data
        try:
            if sort_column in filtered_df.columns:
                display_df = filtered_df.sort_values(sort_column, ascending=False).head(show_rows)
            else:
                display_df = filtered_df.head(show_rows)
                
            st.dataframe(
                display_df.round(3),
                use_container_width=True,
                height=400
            )
        except Exception as e:
            st.error(f"Error displaying data: {str(e)}")
        
        # Download button
        try:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name=f"marketing_campaign_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")

# Footer
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**Built with ❤️ using Streamlit**")
with footer_cols[1]:
    st.markdown("**📊 Data Analytics Dashboard**")
with footer_cols[2]:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
