"""
Streamlit page for NDR (National Debt Relief) Revshare Pixel Firing
"""

import streamlit as st
import pandas as pd
import requests
import logging
import os
from datetime import datetime, timedelta
import uuid
import io

# Try to import chardet for encoding detection
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

class StreamlitLogger:
    """Custom logger that writes to Streamlit"""
    def __init__(self, log_container):
        self.log_container = log_container
        self.logs = []
    
    def info(self, message):
        self.logs.append(message)
        self.log_container.text(message)
    
    def warning(self, message):
        self.logs.append(f"⚠️ WARNING: {message}")
        self.log_container.warning(message)
    
    def error(self, message):
        self.logs.append(f"❌ ERROR: {message}")
        self.log_container.error(message)
    
    def get_logs(self):
        return "\n".join(self.logs)

def clean_data(df, start_date, end_date, logger):
    """
    Clean and filter the data based on the requirements
    """
    try:
        logger.info(f"\nStarting with {len(df)} total records")
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Try to find the Enrollment Datetime column with different possible names
        enrollment_col = None
        possible_enrollment_names = ['Enrollment Datetime', 'Enrollment_Datetime', 'Enrollment Date', 
                                     'Enrollment_Date', 'EnrollmentDatetime', 'Enrollment', 'enrollment datetime']
        
        # First try exact match (case-sensitive)
        for col_name in possible_enrollment_names:
            if col_name in df.columns:
                enrollment_col = col_name
                logger.info(f"Found '{col_name}' column - using it directly")
                break
        
        # If not found, try case-insensitive match
        if enrollment_col is None:
            df_columns_lower = {col.lower(): col for col in df.columns}
            for col_name in possible_enrollment_names:
                if col_name.lower() in df_columns_lower:
                    enrollment_col = df_columns_lower[col_name.lower()]
                    logger.info(f"Found '{enrollment_col}' column via case-insensitive match")
                    break
        
        if enrollment_col is None:
            raise ValueError(f"Could not find enrollment datetime column. Available columns: {list(df.columns)}")
        
        logger.info(f"FINAL enrollment_col value: {enrollment_col}")
        logger.info(f"Using enrollment date column: {enrollment_col}")
        
        # Convert Enrollment Datetime to datetime
        logger.info(f"About to convert column '{enrollment_col}' to datetime")
        df[enrollment_col] = pd.to_datetime(df[enrollment_col], errors='coerce')
        logger.info(f"Successfully converted column '{enrollment_col}' to datetime")
        
        # Drop rows with invalid dates
        df = df.dropna(subset=[enrollment_col])
        
        # Log some sample dates
        sample_dates = df[enrollment_col].head()
        logger.info(f"\nSample {enrollment_col} values:")
        for idx, date in enumerate(sample_dates):
            logger.info(f"Record {idx}: {date}")
        
        # Try to find the Affiliate SubID 1 column
        affiliate_col = None
        possible_affiliate_names = ['Affiliate SubID 1', 'Affiliate_SubID_1', 'Affiliate SubID1', 
                                   'AffiliateSubID1', 'affiliate_subid1', 'subid1', 'SubID 1']
        
        for col_name in possible_affiliate_names:
            if col_name in df.columns:
                affiliate_col = col_name
                logger.info(f"Found affiliate column: {col_name}")
                break
        
        # If not found, try case-insensitive match
        if affiliate_col is None:
            df_columns_lower = {col.lower(): col for col in df.columns}
            for col_name in possible_affiliate_names:
                if col_name.lower() in df_columns_lower:
                    affiliate_col = df_columns_lower[col_name.lower()]
                    logger.info(f"Found affiliate column via case-insensitive match: {affiliate_col}")
                    break
        
        if affiliate_col is None:
            raise ValueError(f"Could not find Affiliate SubID 1 column. Available columns: {list(df.columns)}")
        
        # Filter for Affiliate SubID 1 = 43305
        df_after_affiliate = df[df[affiliate_col].astype(str).str.strip() == '43305'].copy()
        logger.info(f"After filtering for {affiliate_col} = 43305: {len(df_after_affiliate)} records")
        
        if len(df_after_affiliate) == 0:
            logger.warning(f"No records found with {affiliate_col} = 43305")
            return pd.DataFrame(), enrollment_col
        
        # Log all unique dates in the dataset after affiliate filter
        unique_dates = sorted(df_after_affiliate[enrollment_col].dt.date.unique())
        logger.info(f"\nAll unique dates in dataset:")
        for date in unique_dates:
            count = len(df_after_affiliate[df_after_affiliate[enrollment_col].dt.date == date])
            logger.info(f"Date {date}: {count} records")
        
        # Filter for date range (dates are already datetime objects)
        logger.info(f"\nLooking for records between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        date_filter = (df_after_affiliate[enrollment_col] >= start_date) & (df_after_affiliate[enrollment_col] <= end_date)
        df_after_date = df_after_affiliate[date_filter].copy()
        logger.info(f"After filtering for date range: {len(df_after_date)} records")
        
        if len(df_after_date) == 0:
            logger.warning(f"No records found in the date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return pd.DataFrame(), enrollment_col
        
        # Try to find the Total Enrolled Debt column
        debt_col = None
        possible_debt_names = ['Total Enrolled Debt', 'Total_Enrolled_Debt', 'TotalEnrolledDebt', 
                              'Enrolled Debt', 'enrolled_debt', 'Debt', 'debt', 'Total Debt']
        
        for col_name in possible_debt_names:
            if col_name in df_after_date.columns:
                debt_col = col_name
                logger.info(f"Found Total Enrolled Debt column: {col_name}")
                break
        
        # If not found, try case-insensitive match
        if debt_col is None:
            df_columns_lower = {col.lower(): col for col in df_after_date.columns}
            for col_name in possible_debt_names:
                if col_name.lower() in df_columns_lower:
                    debt_col = df_columns_lower[col_name.lower()]
                    logger.info(f"Found debt column via case-insensitive match: {debt_col}")
                    break
        
        if debt_col is None:
            raise ValueError(f"Could not find Total Enrolled Debt column. Available columns: {list(df_after_date.columns)}")
        
        # Convert Total Enrolled Debt to numeric, handling any non-numeric values
        df_after_date[debt_col] = pd.to_numeric(df_after_date[debt_col], errors='coerce')
        df_after_date = df_after_date.dropna(subset=[debt_col])
        
        # Rename the column to 'Total Enrolled Debt' for consistency
        if debt_col != 'Total Enrolled Debt':
            df_after_date = df_after_date.rename(columns={debt_col: 'Total Enrolled Debt'})
            logger.info(f"Renamed column '{debt_col}' to 'Total Enrolled Debt'")
        
        logger.info(f"After filtering for valid Total Enrolled Debt: {len(df_after_date)} records")
        
        # Log some sample debt values
        sample_debt = df_after_date['Total Enrolled Debt'].head()
        logger.info("\nSample Total Enrolled Debt values:")
        for idx, debt in enumerate(sample_debt):
            logger.info(f"Record {idx}: ${debt:,.2f}")
        
        return df_after_date, enrollment_col
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def fire_pixel(transaction_id, enrolled_debt_amount, enrollment_date, logger):
    """
    Fire the pixel for a given transaction ID, enrolled debt amount, and enrollment date
    """
    try:
        logger.info("🚀 FIRE_PIXEL FUNCTION - NDR REVSHARE 🚀")
        # Calculate revenue share amount (4% of Total Enrolled Debt)
        revenue_amount = enrolled_debt_amount * 0.04
        revenue_amount_formatted = f"{revenue_amount:.2f}"
        
        # Format the enrollment date to the required format: 2016-07-15T15:14:21+00:00
        # Set time to noon for consistency
        logger.info(f"Original enrollment_date: {enrollment_date}, type: {type(enrollment_date)}")
        pixel_datetime = enrollment_date.replace(hour=12, minute=0, second=0)
        iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        logger.info(f"ISO datetime for pixel: {iso_datetime}")
        
        # Set up parameters
        params = {
            'o': '32031',
            'e': '811',
            'f': 'pb',
            't': transaction_id,
            'pubid': '43305',
            'campid': '96583',
            'crtvid': '22281',
            'dt': iso_datetime,
            'p': revenue_amount_formatted
        }
        
        # Make the request
        pixel_url = 'https://trkfocus.com/m.ashx'
        logger.info(f"Firing pixel with URL: {pixel_url}")
        logger.info(f"Parameters: {params}")
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()
        logger.info(f"Full pixel URL: {response.url}")
        
        logger.info(f"Fired pixel successfully - Transaction ID: {transaction_id}, Revenue Amount: ${revenue_amount_formatted}, Date: {iso_datetime}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to fire pixel - Transaction ID: {transaction_id} - Error: {str(e)}")
        return False

def process_ndr_report(uploaded_file, start_date, end_date, logger):
    """
    Main function to process the NDR report and fire pixels
    """
    try:
        logger.info("\n🔥🔥🔥 NDR Revshare Pixel Firing Process Started 🔥🔥🔥")
        logger.info("=== NDR Revshare Pixel Firing Process Started ===")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Read the uploaded file based on file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_content = uploaded_file.getvalue()
        
        if file_extension == '.xlsx':
            logger.info("Attempting to read Excel file")
            # Try reading with header on row 12 first (NDR format)
            try:
                df = pd.read_excel(io.BytesIO(file_content), header=11)  # 0-indexed, so row 12 = index 11
                # Verify we got valid columns
                if 'Lead Source' in df.columns or 'Enrollment Datetime' in df.columns or 'Affiliate SubID 1' in df.columns:
                    logger.info("Successfully read Excel file with header on row 12")
                    successful_encoding = "excel (header row 12)"
                else:
                    # If not NDR format, try default header
                    logger.info("Row 12 didn't have expected columns, trying default header row")
                    df = pd.read_excel(io.BytesIO(file_content))
                    successful_encoding = "excel"
                    logger.info("Successfully read Excel file with default header")
            except Exception as e:
                logger.warning(f"Failed to read with header on row 12: {str(e)}, trying default")
                df = pd.read_excel(io.BytesIO(file_content))
                successful_encoding = "excel"
                logger.info("Successfully read Excel file with default header")
        else:
            # Try to read the CSV file with different encodings
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            if CHARDET_AVAILABLE:
                result = chardet.detect(file_content)
                detected_encoding = result['encoding']
                encodings_to_try.insert(0, detected_encoding)
                logger.info(f"Detected encoding: {detected_encoding}")
            
            df = None
            successful_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Attempting to read CSV with {encoding} encoding")
                    df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                    successful_encoding = encoding
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except (UnicodeDecodeError, Exception) as e:
                    logger.warning(f"Failed to read with {encoding} encoding")
                    continue
            
            if df is None:
                raise ValueError("Could not read the CSV file with any encoding")
        
        # Convert dates to pandas datetime objects before passing to clean_data
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # Clean and filter the data
        filtered_df, enrollment_col = clean_data(df, start_date_dt, end_date_dt, logger)
        
        if len(filtered_df) == 0:
            logger.warning("No qualifying records found. No pixels will be fired.")
            return {
                'success': True,
                'encoding': successful_encoding,
                'pixels_fired': 0,
                'total_attempted': 0,
                'total_debt': 0,
                'total_revenue': 0
            }
        
        # Fire pixels for each qualifying record
        pixels_fired = 0
        total_debt = 0
        total_revenue = 0
        
        for _, row in filtered_df.iterrows():
            # Generate a unique transaction ID using enrollment date
            enrollment_date = row[enrollment_col]
            transaction_id = f"NDR_{enrollment_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
            
            enrolled_debt = row['Total Enrolled Debt']
            
            if fire_pixel(transaction_id, enrolled_debt, enrollment_date, logger):
                pixels_fired += 1
                total_debt += enrolled_debt
                total_revenue += (enrolled_debt * 0.04)
        
        logger.info("\n=== Summary ===")
        logger.info(f"File processed successfully with encoding: {successful_encoding}")
        logger.info(f"Total pixels fired successfully: {pixels_fired} out of {len(filtered_df)}")
        logger.info(f"Total Enrolled Debt processed: ${total_debt:,.2f}")
        logger.info(f"Total revenue share amount (4%): ${total_revenue:,.2f}")
        
        return {
            'success': True,
            'encoding': successful_encoding,
            'pixels_fired': pixels_fired,
            'total_attempted': len(filtered_df),
            'total_debt': total_debt,
            'total_revenue': total_revenue
        }
        
    except Exception as e:
        logger.error(f"Error processing NDR report: {str(e)}")
        raise

def show_ndr_pixel():
    """Display the NDR Revshare Pixel Firing interface"""
    st.title("NDR Revshare Pixel Firing - v1.1")
    
    st.success("🔄 **NDR (National Debt Relief) Pixel Firing** - 4% Revenue Share")
    st.info("✅ **v1.1 Update**: Now automatically detects NDR report format with headers starting on row 12")
    
    st.write("""
    This tool processes NDR reports and fires pixels for qualifying enrollments based on revenue share calculations.
    Upload your NDR report (CSV or XLSX format) and specify the date range to begin.
    """)
    
    st.info("""
    **Required columns (case-insensitive):**
    - `Affiliate SubID 1` or similar (filtered for value "43305")
    - `Enrollment Datetime` or similar date column
    - `Total Enrolled Debt` or similar revenue column
    
    **Supported column name variations:**
    - Date columns: Enrollment Datetime, Enrollment Date, EnrollmentDatetime, etc.
    - Debt columns: Total Enrolled Debt, Enrolled Debt, Total Debt, Debt, etc.
    - Affiliate columns: Affiliate SubID 1, Affiliate SubID1, AffiliateSubID1, etc.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload NDR Report (CSV or XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        st.markdown("---")
        st.subheader("📅 Select Date Range")
        st.write("Choose the enrollment date range to filter which records should fire pixels:")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=7),
                help="Select the start date for filtering enrollments"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Select the end date for filtering enrollments"
            )
        
        # Validate date range
        if start_date > end_date:
            st.error("❌ Start date must be before or equal to end date.")
        else:
            # Process button
            if st.button("🚀 Process Report and Fire Pixels", type="primary"):
                with st.spinner("Processing NDR report..."):
                    # Create a container for logs
                    log_container = st.expander("📋 Processing Logs", expanded=True)
                    logger = StreamlitLogger(log_container)
                    
                    try:
                        # Process the report
                        result = process_ndr_report(uploaded_file, start_date, end_date, logger)
                        
                        if result['success']:
                            st.success("✅ Processing completed successfully!")
                            
                            # Display summary
                            st.subheader("📊 Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Pixels Fired", f"{result['pixels_fired']}/{result['total_attempted']}")
                            with col2:
                                st.metric("Total Enrolled Debt", f"${result['total_debt']:,.2f}")
                            with col3:
                                st.metric("Revenue Share (4%)", f"${result['total_revenue']:,.2f}")
                            with col4:
                                st.metric("File Encoding", result['encoding'])
                            
                            # Download logs button
                            logs_text = logger.get_logs()
                            st.download_button(
                                label="📥 Download Processing Logs",
                                data=logs_text,
                                file_name=f"ndr_pixel_firing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error processing report: {str(e)}")
                        logger.error(f"Fatal error: {str(e)}")
    
    # Pixel configuration info
    with st.expander("🔧 Pixel Configuration"):
        st.code("""
Pixel URL: https://trkfocus.com/m.ashx

Parameters:
- o=32031 (Organization ID)
- e=811 (Event ID)
- f=pb (Format)
- t=TRANSACTION_ID (Unique transaction ID)
- pubid=43305 (Publisher ID)
- campid=96583 (Campaign ID)
- crtvid=22281 (Creative ID)
- dt=YYYY-MM-DDTHH:MM:SS+00:00 (Enrollment date in ISO format)
- p=REVENUE_AMOUNT (4% of Total Enrolled Debt)

Example:
https://trkfocus.com/m.ashx?o=32031&e=811&f=pb&t=NDR_20251015_a1b2c3d4&pubid=43305&campid=96583&crtvid=22281&dt=2025-10-15T12:00:00+00:00&p=1000.00
        """, language="text")

if __name__ == "__main__":
    show_ndr_pixel()

