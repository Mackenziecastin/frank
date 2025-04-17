import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

def show_bob_analysis():
    st.title("Bob's Analysis Dashboard")
    
    st.write("""
    Welcome to Bob's specialized analysis dashboard. This tool provides insights into:
    - Performance metrics
    - Partner analysis
    - Trend visualization
    """)
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        affiliate_file = st.file_uploader("Upload Affiliate Leads QA File (CSV)", type=['csv'])
    
    with col2:
        advanced_file = st.file_uploader("Upload Advanced Action Sheet (CSV)", type=['csv'])
    
    if affiliate_file and advanced_file:
        try:
            # Read files
            affiliate_df = pd.read_csv(affiliate_file)
            advanced_df = pd.read_csv(advanced_file)
            
            # Process the data
            processed_data = process_data(affiliate_df, advanced_df)
            if processed_data:
                affiliate_processed, advanced_processed = processed_data
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Overview", "Partner Analysis", "Trends"])
                
                with tab1:
                    show_overview(affiliate_processed, advanced_processed)
                
                with tab2:
                    show_partner_analysis(affiliate_processed, advanced_processed)
                
                with tab3:
                    show_trends(affiliate_processed)
                
        except Exception as e:
            st.error(f"An error occurred while processing the files: {str(e)}")
            st.error("Please ensure your files contain all required columns and are in the correct format.")
    else:
        st.info("Please upload both required files to generate Bob's analysis.")

def process_data(affiliate_df, advanced_df):
    """Process and clean the input dataframes."""
    try:
        # Make copies to avoid modifying originals
        affiliate_df = affiliate_df.copy()
        advanced_df = advanced_df.copy()
        
        # Convert date columns
        date_cols = [col for col in affiliate_df.columns if 'date' in col.lower()]
        for col in date_cols:
            affiliate_df[col] = pd.to_datetime(affiliate_df[col])
        
        date_cols = [col for col in advanced_df.columns if 'date' in col.lower()]
        for col in date_cols:
            advanced_df[col] = pd.to_datetime(advanced_df[col])
        
        # Extract partner IDs if available
        if 'Click URL' in affiliate_df.columns:
            affiliate_df['Partner_ID'] = affiliate_df['Click URL'].apply(extract_partner_id)
        
        if 'Landing Page URL' in advanced_df.columns:
            advanced_df['Partner_ID'] = advanced_df['Landing Page URL'].apply(extract_partner_id)
        
        return affiliate_df, advanced_df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def extract_partner_id(url):
    """Extract partner ID from URL."""
    try:
        if pd.isna(url):
            return "Unknown"
        
        # Look for pattern after %3D
        match = re.search(r'%3D(\d+)', url)
        if match:
            return match.group(1)
        return "Unknown"
    except:
        return "Unknown"

def show_overview(affiliate_df, advanced_df):
    """Display overview metrics and charts."""
    st.header("Performance Overview")
    
    # Calculate key metrics
    total_leads = len(advanced_df)
    total_sales = affiliate_df['Transaction Count'].sum() if 'Transaction Count' in affiliate_df.columns else 0
    total_revenue = affiliate_df['Net Sales Amount'].sum() if 'Net Sales Amount' in affiliate_df.columns else 0
    total_spend = advanced_df['Action Earnings'].sum() if 'Action Earnings' in advanced_df.columns else 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Leads", f"{total_leads:,}")
    
    with col2:
        st.metric("Total Sales", f"{total_sales:,}")
    
    with col3:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col4:
        st.metric("Total Spend", f"${total_spend:,.2f}")
    
    # Calculate and display conversion metrics
    if total_leads > 0:
        conversion_rate = (total_sales / total_leads) * 100
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    if total_spend > 0:
        cost_per_lead = total_spend / total_leads
        st.metric("Cost per Lead", f"${cost_per_lead:.2f}")
    
    # Show data samples
    st.subheader("Data Samples")
    
    sample_tab1, sample_tab2 = st.tabs(["Affiliate Data", "Advanced Action Data"])
    
    with sample_tab1:
        st.dataframe(affiliate_df.head())
    
    with sample_tab2:
        st.dataframe(advanced_df.head())

def show_partner_analysis(affiliate_df, advanced_df):
    """Display partner-specific analysis."""
    st.header("Partner Analysis")
    
    if 'Partner_ID' in affiliate_df.columns:
        # Get list of partners
        partners = affiliate_df['Partner_ID'].unique()
        selected_partner = st.selectbox("Select Partner", ["All"] + list(partners))
        
        if selected_partner != "All":
            # Filter data for selected partner
            partner_affiliate = affiliate_df[affiliate_df['Partner_ID'] == selected_partner]
            partner_advanced = advanced_df[advanced_df['Partner_ID'] == selected_partner]
            
            # Show partner metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                partner_leads = len(partner_advanced)
                st.metric("Partner Leads", f"{partner_leads:,}")
            
            with col2:
                partner_sales = partner_affiliate['Transaction Count'].sum() if 'Transaction Count' in partner_affiliate.columns else 0
                st.metric("Partner Sales", f"{partner_sales:,}")
            
            with col3:
                partner_revenue = partner_affiliate['Net Sales Amount'].sum() if 'Net Sales Amount' in partner_affiliate.columns else 0
                st.metric("Partner Revenue", f"${partner_revenue:,.2f}")
            
            # Show partner performance over time
            if 'Transaction Count' in partner_affiliate.columns:
                date_col = next((col for col in partner_affiliate.columns if 'date' in col.lower()), None)
                if date_col:
                    daily_sales = partner_affiliate.groupby(partner_affiliate[date_col].dt.date)['Transaction Count'].sum()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    daily_sales.plot(kind='line', ax=ax)
                    plt.title(f'Daily Sales Trend - Partner {selected_partner}')
                    plt.xlabel('Date')
                    plt.ylabel('Sales')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        else:
            # Show comparison of all partners
            if 'Transaction Count' in affiliate_df.columns:
                partner_performance = affiliate_df.groupby('Partner_ID').agg({
                    'Transaction Count': 'sum',
                    'Net Sales Amount': 'sum'
                }).reset_index()
                
                partner_performance = partner_performance.sort_values('Transaction Count', ascending=False)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=partner_performance.head(10), 
                          x='Partner_ID', 
                          y='Transaction Count',
                          ax=ax)
                plt.title('Top 10 Partners by Sales')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    else:
        st.warning("Partner ID information not available in the data.")

def show_trends(affiliate_df):
    """Display trend analysis and visualizations."""
    st.header("Trend Analysis")
    
    if 'Transaction Count' in affiliate_df.columns:
        date_col = next((col for col in affiliate_df.columns if 'date' in col.lower()), None)
        if date_col:
            # Create date filter
            min_date = affiliate_df[date_col].min()
            max_date = affiliate_df[date_col].max()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                
                # Filter data
                filtered_df = affiliate_df[
                    (affiliate_df[date_col].dt.date >= start_date) &
                    (affiliate_df[date_col].dt.date <= end_date)
                ]
                
                # Daily trend
                daily_sales = filtered_df.groupby(filtered_df[date_col].dt.date)['Transaction Count'].sum()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                daily_sales.plot(kind='line', ax=ax)
                plt.title('Daily Sales Trend')
                plt.xlabel('Date')
                plt.ylabel('Sales')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Weekly trend
                weekly_sales = filtered_df.groupby(filtered_df[date_col].dt.isocalendar().week)['Transaction Count'].sum()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                weekly_sales.plot(kind='bar', ax=ax)
                plt.title('Weekly Sales')
                plt.xlabel('Week Number')
                plt.ylabel('Sales')
                st.pyplot(fig)
                
                # Monthly trend
                monthly_sales = filtered_df.groupby(filtered_df[date_col].dt.month)['Transaction Count'].sum()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                monthly_sales.plot(kind='bar', ax=ax)
                plt.title('Monthly Sales')
                plt.xlabel('Month')
                plt.ylabel('Sales')
                st.pyplot(fig)
        else:
            st.warning("No date column found for trend analysis.")
    else:
        st.warning("Sales data not available for trend analysis.")

if __name__ == "__main__":
    show_bob_analysis() 