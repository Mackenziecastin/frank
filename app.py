import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from adtbob_processor import ADTBobProcessor

st.set_page_config(page_title="ADT Analysis Dashboard", layout="wide")

def show_frank_analysis():
    st.title("Frank's Partner Optimization Report Generator")
    st.write("""
    This tool processes your marketing data files and generates a comprehensive optimization report.
    Please upload the required files below.
    """)
    # ... existing code for Frank's analysis ...
    # [Keep all the existing code for Frank's analysis here]

def show_bob_analysis():
    st.title("Bob's ADT Analysis Dashboard")
    st.markdown("""
    Welcome to Bob's ADT Analysis Dashboard. This tool helps you analyze and optimize your marketing performance.
    Upload your data files to get started.
    """)
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        affiliate_file = st.file_uploader("Upload Affiliate Leads QA File", type=['csv'])
    with col2:
        advanced_file = st.file_uploader("Upload Advanced Action Sheet", type=['csv'])
    
    # Optional partner list upload
    partner_list_file = st.file_uploader("Upload Partner List (Optional)", type=['csv'])
    
    if affiliate_file and advanced_file:
        try:
            # Initialize processor
            processor = ADTBobProcessor()
            
            # Load data
            processor.load_data(
                affiliate_file=affiliate_file,
                advanced_file=advanced_file,
                partner_list_file=partner_list_file
            )
            
            # Clean data
            processor.clean_data()
            
            # Generate report
            report = processor.generate_optimization_report()
            
            # Display tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Overview", "Partner Analysis", "Trends"])
            
            with tab1:
                show_overview(report)
            
            with tab2:
                show_partner_analysis(report)
            
            with tab3:
                show_trends(processor.processed_affiliate_df)
            
            # Export button
            if st.button("Export Report"):
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_report_{timestamp}.xlsx"
                
                # Export report
                processor.export_report(filename)
                st.success(f"Report exported successfully as {filename}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload both the Affiliate Leads QA File and Advanced Action Sheet to begin analysis.")

def show_overview(report):
    """Display overview metrics and summary statistics."""
    st.header("Performance Overview")
    
    # Calculate summary metrics
    total_leads = report['Leads'].sum()
    total_sales = report['Sales'].sum()
    total_revenue = report['Revenue'].sum()
    total_spend = report['Spend'].sum()
    overall_conversion = (total_sales / total_leads * 100) if total_leads > 0 else 0
    overall_roas = total_revenue / total_spend if total_spend > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Leads", f"{total_leads:,.0f}")
        st.metric("Total Sales", f"{total_sales:,.0f}")
    with col2:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
        st.metric("Total Spend", f"${total_spend:,.2f}")
    with col3:
        st.metric("Overall Conversion Rate", f"{overall_conversion:.1f}%")
        st.metric("Overall ROAS", f"{overall_roas:.2f}x")
    
    # Display top partners
    st.subheader("Top Performing Partners")
    top_partners = report.nlargest(5, 'Revenue')[['Partner_ID', 'Partner_Name', 'Revenue', 'ROAS']]
    st.dataframe(top_partners)

def show_partner_analysis(report):
    """Display detailed partner analysis."""
    st.header("Partner Analysis")
    
    # Partner selector
    partner_options = report['Partner_ID'].unique()
    selected_partner = st.selectbox("Select Partner", partner_options)
    
    # Display partner metrics
    partner_data = report[report['Partner_ID'] == selected_partner].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Leads", f"{partner_data['Leads']:,.0f}")
        st.metric("Sales", f"{partner_data['Sales']:,.0f}")
    with col2:
        st.metric("Revenue", f"${partner_data['Revenue']:,.2f}")
        st.metric("Spend", f"${partner_data['Spend']:,.2f}")
    with col3:
        st.metric("Conversion Rate", f"{partner_data['Conversion_Rate']:.1f}%")
        st.metric("ROAS", f"{partner_data['ROAS']:.2f}x")
    
    # Partner comparison chart
    st.subheader("Partner Performance Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=report['Partner_ID'],
        y=report['Revenue'],
        name='Revenue'
    ))
    fig.add_trace(go.Bar(
        x=report['Partner_ID'],
        y=report['Spend'],
        name='Spend'
    ))
    fig.update_layout(
        barmode='group',
        title='Revenue vs Spend by Partner',
        xaxis_title='Partner ID',
        yaxis_title='Amount ($)'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_trends(affiliate_df):
    """Display trend analysis."""
    st.header("Trend Analysis")
    
    # Date range selector
    min_date = affiliate_df['Sale_Date'].min()
    max_date = affiliate_df['Sale_Date'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date)
    with col2:
        end_date = st.date_input("End Date", max_date)
    
    # Filter data by date range
    mask = (affiliate_df['Sale_Date'].dt.date >= start_date) & (affiliate_df['Sale_Date'].dt.date <= end_date)
    filtered_df = affiliate_df[mask]
    
    # Daily trends
    st.subheader("Daily Trends")
    daily_data = filtered_df.groupby('Sale_Date').agg({
        'Transaction Count': 'sum',
        'Net Sales Amount': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_data['Sale_Date'],
        y=daily_data['Transaction Count'],
        name='Sales',
        mode='lines+markers'
    ))
    fig.update_layout(
        title='Daily Sales Trend',
        xaxis_title='Date',
        yaxis_title='Number of Sales'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_data['Sale_Date'],
        y=daily_data['Net Sales Amount'],
        name='Revenue',
        mode='lines+markers'
    ))
    fig.update_layout(
        title='Daily Revenue Trend',
        xaxis_title='Date',
        yaxis_title='Revenue ($)'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Create the navigation
    st.sidebar.title("Navigation")
<<<<<<< HEAD
    page = st.sidebar.radio("Go to", ["Frank's Analysis", "Bob's Analysis"])
    
    if page == "Frank's Analysis":
        show_frank_analysis()
    else:
=======
    page = st.sidebar.radio("Go to", ["Frank (LaserAway)", "Bob (ADT)"])
    
    if page == "Frank (LaserAway)":
        show_main_page()
    elif page == "Bob (ADT)":
        # Import and show Bob's analysis page
        from pages.bob_analysis import show_bob_analysis
>>>>>>> a050826157a3e8a68a4be99ac4c3af59579fe81c
        show_bob_analysis()

if __name__ == "__main__":
    main() 