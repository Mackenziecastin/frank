import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO, StringIO
from datetime import datetime, timedelta
import logging
import sys
import os
import uuid
import requests
import io
import tempfile
import chardet

def setup_logging():
    """Set up logging to capture output"""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    
    # Create a logger and add the handler
    logger = logging.getLogger('laseraway_pixel')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger, log_stream

def detect_encoding(file_path):
    """
    Attempt to detect the file encoding
    """
    try:
        # Read a sample of the file to detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read(min(1024 * 1024, os.path.getsize(file_path)))  # Read up to 1MB
        
        # Detect encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        return encoding
    except Exception as e:
        return None

def clean_data(df, start_date, end_date, logger):
    """
    Clean and filter the data according to requirements.
    """
    try:
        logger.info(f"\nStarting with {len(df)} total records")
        
        # Convert Purchased column to datetime if it's not already and remove any null values
        df['Purchased'] = pd.to_datetime(df['Purchased'], errors='coerce')
        df = df.dropna(subset=['Purchased'])
        
        # Log some sample dates for debugging
        sample_dates = df['Purchased'].head()
        logger.info("\nSample Purchased values:")
        for idx, date in enumerate(sample_dates):
            logger.info(f"Record {idx}: {date}")
        
        # Filter for affiliate_directagent_subid1 = 42865
        df['affiliate_directagent_subid1'] = df['affiliate_directagent_subid1'].fillna('')
        affiliate_filter = df['affiliate_directagent_subid1'] == '42865'
        df_after_affiliate = df[affiliate_filter]
        logger.info(f"After filtering for affiliate_directagent_subid1 = 42865: {len(df_after_affiliate)} records")
        
        if len(df_after_affiliate) == 0:
            logger.info("No records found with affiliate_directagent_subid1 = 42865")
            return df_after_affiliate
        
        # Filter for date range
        date_filter = (df_after_affiliate['Purchased'] >= start_date) & (df_after_affiliate['Purchased'] <= end_date)
        df_after_date = df_after_affiliate[date_filter]
        
        # Log all unique dates in the dataset
        unique_dates = sorted(df_after_affiliate['Purchased'].dt.date.unique())
        logger.info("\nAll unique dates in dataset:")
        for date in unique_dates:
            count = len(df_after_affiliate[df_after_affiliate['Purchased'].dt.date == date])
            logger.info(f"Date {date}: {count} records")
        
        logger.info(f"\nLooking for records between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"After filtering for date range: {len(df_after_date)} records")
        
        # Check Net Sales column
        if 'Net Sales' not in df_after_date.columns:
            logger.error("Net Sales column not found in the dataset")
            raise ValueError("Net Sales column is required but not found")
        
        # Convert Net Sales to numeric, handling any non-numeric values
        df_after_date['Net Sales'] = pd.to_numeric(df_after_date['Net Sales'], errors='coerce')
        df_after_date = df_after_date.dropna(subset=['Net Sales'])
        
        logger.info(f"After filtering for valid Net Sales: {len(df_after_date)} records")
        
        # Log some sample Net Sales values
        sample_sales = df_after_date['Net Sales'].head()
        logger.info("\nSample Net Sales values:")
        for idx, sales in enumerate(sample_sales):
            logger.info(f"Record {idx}: {sales}")
        
        return df_after_date
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def fire_pixel(transaction_id, net_sales_amount, purchase_date, logger):
    """
    Fire the pixel for a given transaction ID, net sales amount, and purchase date
    """
    try:
        # Calculate revenue share amount (Net Sales / 1.75)
        revenue_amount = net_sales_amount / 1.75
        revenue_amount_formatted = f"{revenue_amount:.2f}"
        
        # Format the purchase date to the required format: 2016-07-15T15:14:21+00:00
        # Set time to noon for consistency
        pixel_datetime = purchase_date.replace(hour=12, minute=0, second=0)
        iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        
        # Set up parameters
        params = {
            'o': '32067',
            'e': '865',
            'f': 'pb',
            't': transaction_id,
            'pubid': '42865',
            'campid': '96548',
            'dt': iso_datetime,
            'p': revenue_amount_formatted
        }
        
        # Make the request
        pixel_url = 'https://trkstar.com/m.ashx'
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()
        
        logger.info(f"Fired pixel successfully - Transaction ID: {transaction_id}, Revenue Amount: {revenue_amount_formatted}, Date: {iso_datetime}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to fire pixel - Transaction ID: {transaction_id} - Error: {str(e)}")
        return False

def process_laseraway_report(uploaded_file, start_date, end_date, logger):
    """
    Main function to process the LaserAway report and fire pixels
    """
    try:
        logger.info("\n=== LaserAway Revshare Pixel Firing Process Started ===")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Read the uploaded file
        file_content = uploaded_file.getvalue()
        
        # Try different encodings if needed
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        
        df = None
        successful_encoding = None
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Attempting to read file with {encoding} encoding")
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                successful_encoding = encoding
                logger.info(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to read with {encoding} encoding: {str(e)}")
            except Exception as e:
                logger.warning(f"Error reading CSV with {encoding} encoding: {str(e)}")
        
        if df is None:
            raise Exception("Failed to read CSV file with any of the attempted encodings")
        
        # Clean and filter the data
        filtered_df = clean_data(df, start_date, end_date, logger)
        
        if len(filtered_df) == 0:
            logger.info("No qualifying sales found to process.")
            return 0, 0, 0.0
        
        # Initialize counters
        successful_pixels = 0
        total_pixels = 0
        total_revenue = 0
        
        # Process each record
        for _, row in filtered_df.iterrows():
            transaction_id = f"LASERAWAY_{row['Purchased'].strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
            net_sales = row['Net Sales']
            purchase_date = row['Purchased']
            
            total_pixels += 1
            total_revenue += net_sales
            
            # Fire pixel
            if fire_pixel(transaction_id, net_sales, purchase_date, logger):
                successful_pixels += 1
        
        # Log summary
        logger.info("\n=== Summary ===")
        logger.info(f"File processed successfully with encoding: {successful_encoding}")
        logger.info(f"Total pixels fired successfully: {successful_pixels} out of {total_pixels}")
        logger.info(f"Total Net Sales processed: ${total_revenue:.2f}")
        logger.info(f"Total revenue share amount: ${total_revenue / 1.75:.2f}")
        
        return successful_pixels, total_pixels, total_revenue
        
    except Exception as e:
        logger.error(f"Error processing LaserAway report: {str(e)}")
        raise

def show_laseraway_pixel():
    """Display the LaserAway Revshare Pixel Firing interface"""
    st.title("LaserAway Revshare Pixel Firing")
    
    st.write("""
    This tool processes LaserAway reports and fires pixels for qualifying sales based on revenue share calculations.
    Upload your LaserAway report (CSV format) and specify the date range to begin.
    """)
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    
    # File upload
    uploaded_file = st.file_uploader("Upload LaserAway Report (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        st.info(f"📁 **File uploaded:** {uploaded_file.name}")
        st.info(f"📅 **Date range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Show required columns info
        st.info("""
        **Required CSV columns:**
        - `affiliate_directagent_subid1` (filtered for value "42865")
        - `Purchased` (date column)
        - `Net Sales` (revenue amount)
        """)
        
        if st.button("🚀 Process and Fire Pixels", type="primary"):
            try:
                # Set up logging
                logger, log_stream = setup_logging()
                
                # Process the report
                successful_pixels, total_pixels, total_revenue = process_laseraway_report(
                    uploaded_file, start_date, end_date, logger
                )
                
                if total_pixels > 0:
                    # Show success message with summary
                    st.success(f"✅ **Pixels fired successfully!**")
                    
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Pixels Fired", f"{successful_pixels}/{total_pixels}")
                    
                    with col2:
                        st.metric("Total Net Sales", f"${total_revenue:,.2f}")
                    
                    with col3:
                        revenue_share = total_revenue / 1.75
                        st.metric("Revenue Share", f"${revenue_share:,.2f}")
                    
                    # Show logs
                    with st.expander("📋 View Processing Logs"):
                        st.text(log_stream.getvalue())
                else:
                    st.warning("⚠️ **No qualifying sales found** for the specified date range and filters.")
                    
                    # Show logs
                    with st.expander("📋 View Processing Logs"):
                        st.text(log_stream.getvalue())
                    
            except Exception as e:
                st.error(f"❌ **Error processing LaserAway report:** {str(e)}")
                
                # Show error logs
                with st.expander("📋 View Error Logs"):
                    st.text(log_stream.getvalue())
    
    # Show pixel details
    with st.expander("🔧 Pixel Configuration Details"):
        st.code("""
Pixel URL: https://trkstar.com/m.ashx

Parameters:
- o=32067 (Organization ID)
- e=865 (Event ID) 
- f=pb (Format)
- t=TRANSACTION_ID (Unique transaction ID)
- pubid=42865 (Publisher ID)
- campid=96548 (Campaign ID)
- dt=YYYY-MM-DDTHH:MM:SS+00:00 (Purchase date in ISO format)
- p=REVENUE_AMOUNT (Calculated revenue share amount)

Revenue Calculation:
Revenue Amount = Net Sales / 1.75
        """)
    
    # Show filtering criteria
    with st.expander("📊 Filtering Criteria"):
        st.write("""
        **Data Processing Steps:**
        1. Filter for `affiliate_directagent_subid1 = 42865` entries only
        2. Filter for sales within the specified date range on `Purchased` column
        3. Calculate revenue share using formula: `Net Sales / 1.75`
        4. Fire pixels with proper date formatting and revenue amounts
        5. Generate unique transaction IDs for each pixel firing
        """)
