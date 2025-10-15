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
# Optional import of chardet
CHARDET_AVAILABLE = False
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    pass  # We'll handle this gracefully

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
        if CHARDET_AVAILABLE:
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(min(1024 * 1024, os.path.getsize(file_path)))  # Read up to 1MB
            
            # Detect encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            return encoding
        else:
            return None
    except Exception as e:
        return None

def clean_data(df, start_date, end_date, logger):
    """
    Clean and filter the data according to requirements.
    """
    try:
        logger.info(f"\nStarting with {len(df)} total records")
        
        # Log all column names for debugging
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Check if 'Purchased Date' exists in column D (index 3)
        if len(df.columns) > 3:
            logger.info(f"Column D (index 3): {df.columns[3]}")
        
        # Check for any column containing 'Purchased'
        purchased_columns = [col for col in df.columns if 'Purchased' in col]
        logger.info(f"Columns containing 'Purchased': {purchased_columns}")
        
        # Try to find the purchased date column with different possible names
        purchased_col = None
        possible_purchased_names = ['Purchased Date', 'Purchase Date', 'Purchase_Date', 'Date', 'date']
        
        logger.info(f"Looking for purchased date column in: {possible_purchased_names}")
        
        # Force use 'Purchased Date' if it exists
        if 'Purchased Date' in df.columns:
            purchased_col = 'Purchased Date'
            logger.info(f"Found 'Purchased Date' column - using it directly")
        else:
            logger.warning("'Purchased Date' not found, trying other options...")
            # Try to find any column that contains 'Purchased'
            for col in df.columns:
                if 'Purchased' in col:
                    purchased_col = col
                    logger.info(f"Found column containing 'Purchased': {col}")
                    break
        
        logger.info(f"FINAL purchased_col value: {purchased_col}")
        logger.info(f"purchased_col is None: {purchased_col is None}")
        
        # Only look for 'Purchased Date' - no fallbacks
        if purchased_col is None:
            logger.error("'Purchased Date' column is required but not found!")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError("'Purchased Date' column is required but not found!")
        
        logger.info(f"Using purchased date column: {purchased_col}")
        
        # Double-check that the column exists before trying to access it
        if purchased_col not in df.columns:
            logger.error(f"Column '{purchased_col}' not found in dataframe. Available columns: {list(df.columns)}")
            raise ValueError(f"Column '{purchased_col}' not found in dataframe. Available columns: {list(df.columns)}")
        
        # Convert the found column to datetime if it's not already and remove any null values
        logger.info(f"About to convert column '{purchased_col}' to datetime")
        try:
            df[purchased_col] = pd.to_datetime(df[purchased_col], errors='coerce')
            df = df.dropna(subset=[purchased_col])
            logger.info(f"Successfully converted column '{purchased_col}' to datetime")
        except Exception as e:
            logger.error(f"Error converting column '{purchased_col}' to datetime: {str(e)}")
            raise
        
        # Log some sample dates for debugging
        logger.info(f"Attempting to access column: {purchased_col}")
        logger.info(f"Column exists: {purchased_col in df.columns}")
        try:
            sample_dates = df[purchased_col].head()
            logger.info("\nSample Purchased Date values:")
            for idx, date in enumerate(sample_dates):
                logger.info(f"Record {idx}: {date}")
        except Exception as e:
            logger.error(f"Error accessing column '{purchased_col}': {str(e)}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise
        
        # Try to find the affiliate column with different possible names
        affiliate_col = None
        possible_affiliate_names = ['affiliate_directagent_subid1', 'affiliate_directagent_subidi', 'affiliate_subid1', 'subid1', 'affiliate_id']
        
        for col_name in possible_affiliate_names:
            if col_name in df.columns:
                affiliate_col = col_name
                logger.info(f"Found affiliate column: {col_name}")
                break
        
        if affiliate_col is None:
            logger.error(f"Could not find affiliate column. Available columns: {list(df.columns)}")
            raise ValueError(f"Could not find affiliate column. Available columns: {list(df.columns)}")
        
        # Filter for affiliate_directagent_subid1 = 42865
        df[affiliate_col] = df[affiliate_col].fillna('')
        affiliate_filter = df[affiliate_col] == '42865'
        df_after_affiliate = df[affiliate_filter]
        logger.info(f"After filtering for {affiliate_col} = 42865: {len(df_after_affiliate)} records")
        
        if len(df_after_affiliate) == 0:
            logger.info("No records found with affiliate_directagent_subid1 = 42865")
            return df_after_affiliate
        
        # Filter for date range (convert date objects to datetime for comparison)
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        date_filter = (df_after_affiliate[purchased_col] >= start_datetime) & (df_after_affiliate[purchased_col] <= end_datetime)
        df_after_date = df_after_affiliate[date_filter]
        
        # Log all unique dates in the dataset
        unique_dates = sorted(df_after_affiliate[purchased_col].dt.date.unique())
        logger.info("\nAll unique dates in dataset:")
        for date in unique_dates:
            count = len(df_after_affiliate[df_after_affiliate[purchased_col].dt.date == date])
            logger.info(f"Date {date}: {count} records")
        
        logger.info(f"\nLooking for records between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"After filtering for date range: {len(df_after_date)} records")
        
        # Try to find the Net Sales column with different possible names
        net_sales_col = None
        possible_net_sales_names = ['Net Sales', 'net_sales', 'Net_Sales', 'Revenue', 'revenue', 'Amount', 'amount', 'Sales', 'sales']
        
        for col_name in possible_net_sales_names:
            if col_name in df_after_date.columns:
                net_sales_col = col_name
                logger.info(f"Found Net Sales column: {col_name}")
                break
        
        if net_sales_col is None:
            logger.error(f"Could not find Net Sales column. Available columns: {list(df_after_date.columns)}")
            raise ValueError(f"Could not find Net Sales column. Available columns: {list(df_after_date.columns)}")
        
        # Convert Net Sales to numeric, handling any non-numeric values
        df_after_date[net_sales_col] = pd.to_numeric(df_after_date[net_sales_col], errors='coerce')
        df_after_date = df_after_date.dropna(subset=[net_sales_col])
        
        # Rename the column to 'Net Sales' for consistency
        if net_sales_col != 'Net Sales':
            df_after_date = df_after_date.rename(columns={net_sales_col: 'Net Sales'})
            logger.info(f"Renamed column '{net_sales_col}' to 'Net Sales'")
        
        logger.info(f"After filtering for valid Net Sales: {len(df_after_date)} records")
        
        # Log some sample Net Sales values
        sample_sales = df_after_date['Net Sales'].head()
        logger.info("\nSample Net Sales values:")
        for idx, sales in enumerate(sample_sales):
            logger.info(f"Record {idx}: {sales}")
        
        return df_after_date, purchased_col
        
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
        
        # Read the uploaded file based on file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_content = uploaded_file.getvalue()
        
        df = None
        successful_encoding = None
        
        if file_extension == '.xlsx':
            # Handle Excel files
            try:
                logger.info("Attempting to read Excel file")
                df = pd.read_excel(io.BytesIO(file_content))
                successful_encoding = "excel"
                logger.info("Successfully read Excel file")
            except Exception as e:
                logger.error(f"Failed to read Excel file: {str(e)}")
                raise Exception(f"Failed to read Excel file: {str(e)}")
        else:
            # Handle CSV files
            # Try different encodings if needed
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
            
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
        filtered_df, purchased_col = clean_data(df, start_date, end_date, logger)
        
        if len(filtered_df) == 0:
            logger.info("No qualifying sales found to process.")
            return 0, 0, 0.0
        
        # Initialize counters
        successful_pixels = 0
        total_pixels = 0
        total_revenue = 0
        
        # Process each record
        for _, row in filtered_df.iterrows():
            transaction_id = f"LASERAWAY_{row[purchased_col].strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
            net_sales = row['Net Sales']
            purchase_date = row[purchased_col]
            
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
    st.title("LaserAway Revshare Pixel Firing - v2.1 DATETIME FIX")
    
    st.success("🔄 **UPDATED VERSION - Oct 15, 2024 v2.1** - Fixed datetime comparison issue")
    st.warning("⚠️ If you still see v2.0, please clear your browser cache and refresh!")
    
    st.write("""
    This tool processes LaserAway reports and fires pixels for qualifying sales based on revenue share calculations.
    Upload your LaserAway report (CSV or XLSX format) and specify the date range to begin.
    """)
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    
    # File upload
    uploaded_file = st.file_uploader("Upload LaserAway Report (CSV or XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        st.info(f"📁 **File uploaded:** {uploaded_file.name}")
        st.info(f"📅 **Date range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Show required columns info
        st.info("""
        **Required columns (case-insensitive):**
        - `affiliate_directagent_subid1` or similar (filtered for value "42865")
        - `Purchased Date` or similar date column
        - `Net Sales` or similar revenue column
        
        **Supported column name variations:**
        - Date columns: Purchased Date, Purchase Date, Date, etc.
        - Revenue columns: Net Sales, Revenue, Amount, Sales, etc.
        - Affiliate columns: affiliate_directagent_subid1, affiliate_subid1, subid1, etc.
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
                
                # Try to show available columns if it's a column-related error
                if "column" in str(e).lower() or "not found" in str(e).lower():
                    try:
                        # Read the file to show available columns
                        file_content = uploaded_file.getvalue()
                        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                        
                        if file_extension == '.xlsx':
                            df_preview = pd.read_excel(io.BytesIO(file_content))
                        else:
                            df_preview = pd.read_csv(io.BytesIO(file_content))
                        
                        st.warning(f"📋 **Available columns in your file:** {list(df_preview.columns)}")
                    except:
                        pass
                
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
        2. Filter for sales within the specified date range on `Purchased Date` column
        3. Calculate revenue share using formula: `Net Sales / 1.75`
        4. Fire pixels with proper date formatting and revenue amounts
        5. Generate unique transaction IDs for each pixel firing
        """)
