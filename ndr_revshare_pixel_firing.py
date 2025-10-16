"""
NDR (National Debt Relief) Revshare Pixel Firing Script

This script processes NDR reports and fires pixels for qualifying sales based on revenue share calculations.

Usage:
    python ndr_revshare_pixel_firing.py <file_path> <start_date> <end_date>
    
Arguments:
    file_path: Path to the NDR report (CSV or XLSX format)
    start_date: Start date for filtering (YYYY-MM-DD format)
    end_date: End date for filtering (YYYY-MM-DD format)

Example:
    python ndr_revshare_pixel_firing.py ndr_report.xlsx 2025-10-01 2025-10-15

Required columns in your file:
- Affiliate SubID 1: Filtered for value "43305"
- Enrollment Datetime: Date column for enrollment dates
- Total Enrolled Debt: Revenue amount for calculations (4% commission)

Pixel configuration:
    URL: https://trkfocus.com/m.ashx
    Parameters:
        - o=32031 (Organization ID)
        - e=811 (Event ID)
        - f=pb (Format)
        - t=TRANSACTION_ID (Unique transaction ID)
        - pubid=43305 (Publisher ID)
        - campid=96583 (Campaign ID)
        - crtvid=22281 (Creative ID)
        - p=REVENUE_AMOUNT (4% of Total Enrolled Debt)
        - dt=YYYY-MM-DDTHH:MM:SS+00:00 (Enrollment date in ISO format)
"""

import pandas as pd
import requests
import logging
import sys
import os
from datetime import datetime
import uuid

# Try to import chardet for encoding detection
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning("chardet not available. Will use default UTF-8 encoding for CSV files.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ndr_pixel_firing.log')
    ]
)

def detect_encoding(file_path):
    """
    Detect the encoding of a CSV file using chardet
    """
    if not CHARDET_AVAILABLE:
        return 'utf-8'
    
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def clean_data(df, file_path, start_date, end_date):
    """
    Clean and filter the data based on the requirements
    """
    try:
        logging.info(f"\nStarting with {len(df)} total records")
        logging.info(f"Available columns: {list(df.columns)}")
        
        # Try to find the Enrollment Datetime column with different possible names
        enrollment_col = None
        possible_enrollment_names = ['Enrollment Datetime', 'Enrollment_Datetime', 'Enrollment Date', 
                                     'Enrollment_Date', 'EnrollmentDatetime', 'Enrollment', 'enrollment datetime']
        
        # First try exact match (case-sensitive)
        for col_name in possible_enrollment_names:
            if col_name in df.columns:
                enrollment_col = col_name
                logging.info(f"Found '{col_name}' column - using it directly")
                break
        
        # If not found, try case-insensitive match
        if enrollment_col is None:
            df_columns_lower = {col.lower(): col for col in df.columns}
            for col_name in possible_enrollment_names:
                if col_name.lower() in df_columns_lower:
                    enrollment_col = df_columns_lower[col_name.lower()]
                    logging.info(f"Found '{enrollment_col}' column via case-insensitive match")
                    break
        
        if enrollment_col is None:
            raise ValueError(f"Could not find enrollment datetime column. Available columns: {list(df.columns)}")
        
        logging.info(f"FINAL enrollment_col value: {enrollment_col}")
        logging.info(f"enrollment_col is None: {enrollment_col is None}")
        logging.info(f"Using enrollment date column: {enrollment_col}")
        
        # Convert Enrollment Datetime to datetime
        logging.info(f"About to convert column '{enrollment_col}' to datetime")
        df[enrollment_col] = pd.to_datetime(df[enrollment_col], errors='coerce')
        logging.info(f"Successfully converted column '{enrollment_col}' to datetime")
        
        # Drop rows with invalid dates
        df = df.dropna(subset=[enrollment_col])
        
        # Log some sample dates
        sample_dates = df[enrollment_col].head()
        logging.info(f"\nSample {enrollment_col} values:")
        for idx, date in enumerate(sample_dates):
            logging.info(f"Record {idx}: {date}")
        
        # Try to find the Affiliate SubID 1 column
        affiliate_col = None
        possible_affiliate_names = ['Affiliate SubID 1', 'Affiliate_SubID_1', 'Affiliate SubID1', 
                                   'AffiliateSubID1', 'affiliate_subid1', 'subid1', 'SubID 1']
        
        for col_name in possible_affiliate_names:
            if col_name in df.columns:
                affiliate_col = col_name
                logging.info(f"Found affiliate column: {col_name}")
                break
        
        # If not found, try case-insensitive match
        if affiliate_col is None:
            df_columns_lower = {col.lower(): col for col in df.columns}
            for col_name in possible_affiliate_names:
                if col_name.lower() in df_columns_lower:
                    affiliate_col = df_columns_lower[col_name.lower()]
                    logging.info(f"Found affiliate column via case-insensitive match: {affiliate_col}")
                    break
        
        if affiliate_col is None:
            raise ValueError(f"Could not find Affiliate SubID 1 column. Available columns: {list(df.columns)}")
        
        # Filter for Affiliate SubID 1 = 43305
        df_after_affiliate = df[df[affiliate_col].astype(str).str.strip() == '43305'].copy()
        logging.info(f"After filtering for {affiliate_col} = 43305: {len(df_after_affiliate)} records")
        
        if len(df_after_affiliate) == 0:
            logging.warning(f"No records found with {affiliate_col} = 43305")
            return pd.DataFrame(), enrollment_col
        
        # Log all unique dates in the dataset after affiliate filter
        unique_dates = sorted(df_after_affiliate[enrollment_col].dt.date.unique())
        logging.info(f"\nAll unique dates in dataset:")
        for date in unique_dates:
            count = len(df_after_affiliate[df_after_affiliate[enrollment_col].dt.date == date])
            logging.info(f"Date {date}: {count} records")
        
        # Filter for date range (dates are already datetime objects)
        logging.info(f"\nLooking for records between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        date_filter = (df_after_affiliate[enrollment_col] >= start_date) & (df_after_affiliate[enrollment_col] <= end_date)
        df_after_date = df_after_affiliate[date_filter].copy()
        logging.info(f"After filtering for date range: {len(df_after_date)} records")
        
        if len(df_after_date) == 0:
            logging.warning(f"No records found in the date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return pd.DataFrame(), enrollment_col
        
        # Try to find the Total Enrolled Debt column
        debt_col = None
        possible_debt_names = ['Total Enrolled Debt', 'Total_Enrolled_Debt', 'TotalEnrolledDebt', 
                              'Enrolled Debt', 'enrolled_debt', 'Debt', 'debt', 'Total Debt']
        
        for col_name in possible_debt_names:
            if col_name in df_after_date.columns:
                debt_col = col_name
                logging.info(f"Found Total Enrolled Debt column: {col_name}")
                break
        
        # If not found, try case-insensitive match
        if debt_col is None:
            df_columns_lower = {col.lower(): col for col in df_after_date.columns}
            for col_name in possible_debt_names:
                if col_name.lower() in df_columns_lower:
                    debt_col = df_columns_lower[col_name.lower()]
                    logging.info(f"Found debt column via case-insensitive match: {debt_col}")
                    break
        
        if debt_col is None:
            raise ValueError(f"Could not find Total Enrolled Debt column. Available columns: {list(df_after_date.columns)}")
        
        # Convert Total Enrolled Debt to numeric, handling any non-numeric values
        df_after_date[debt_col] = pd.to_numeric(df_after_date[debt_col], errors='coerce')
        df_after_date = df_after_date.dropna(subset=[debt_col])
        
        # Rename the column to 'Total Enrolled Debt' for consistency
        if debt_col != 'Total Enrolled Debt':
            df_after_date = df_after_date.rename(columns={debt_col: 'Total Enrolled Debt'})
            logging.info(f"Renamed column '{debt_col}' to 'Total Enrolled Debt'")
        
        logging.info(f"After filtering for valid Total Enrolled Debt: {len(df_after_date)} records")
        
        # Log some sample debt values
        sample_debt = df_after_date['Total Enrolled Debt'].head()
        logging.info("\nSample Total Enrolled Debt values:")
        for idx, debt in enumerate(sample_debt):
            logging.info(f"Record {idx}: ${debt:,.2f}")
        
        return df_after_date, enrollment_col
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

def fire_pixel(transaction_id, enrolled_debt_amount, enrollment_date):
    """
    Fire the pixel for a given transaction ID, enrolled debt amount, and enrollment date
    """
    try:
        logging.info("🚀 FIRE_PIXEL FUNCTION - NDR REVSHARE 🚀")
        # Calculate revenue share amount (4% of Total Enrolled Debt)
        revenue_amount = enrolled_debt_amount * 0.04
        revenue_amount_formatted = f"{revenue_amount:.2f}"
        
        # Format the enrollment date to the required format: 2016-07-15T15:14:21+00:00
        # Set time to noon for consistency
        logging.info(f"Original enrollment_date: {enrollment_date}, type: {type(enrollment_date)}")
        pixel_datetime = enrollment_date.replace(hour=12, minute=0, second=0)
        iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        logging.info(f"ISO datetime for pixel: {iso_datetime}")
        
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
        logging.info(f"Firing pixel with URL: {pixel_url}")
        logging.info(f"Parameters: {params}")
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()
        logging.info(f"Full pixel URL: {response.url}")
        
        logging.info(f"Fired pixel successfully - Transaction ID: {transaction_id}, Revenue Amount: ${revenue_amount_formatted}, Date: {iso_datetime}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to fire pixel - Transaction ID: {transaction_id} - Error: {str(e)}")
        return False

def process_ndr_report(file_path, start_date, end_date):
    """
    Main function to process the NDR report and fire pixels
    """
    try:
        logging.info("\n🔥🔥🔥 NDR Revshare Pixel Firing Process Started 🔥🔥🔥")
        logging.info("=== NDR Revshare Pixel Firing Process Started ===")
        logging.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Read the file based on file type
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.xlsx':
            logging.info("Attempting to read Excel file")
            df = pd.read_excel(file_path)
            successful_encoding = "excel"
            logging.info("Successfully read Excel file")
        else:
            # Try to read the CSV file with detected encoding
            encodings_to_try = []
            
            if CHARDET_AVAILABLE:
                detected_encoding = detect_encoding(file_path)
                encodings_to_try.append(detected_encoding)
                logging.info(f"Detected encoding: {detected_encoding}")
            
            encodings_to_try.extend(['utf-8', 'latin-1', 'iso-8859-1', 'cp1252'])
            
            df = None
            successful_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    logging.info(f"Attempting to read CSV with {encoding} encoding")
                    df = pd.read_csv(file_path, encoding=encoding)
                    successful_encoding = encoding
                    logging.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except (UnicodeDecodeError, Exception) as e:
                    logging.warning(f"Failed to read with {encoding} encoding: {str(e)}")
                    continue
            
            if df is None:
                raise ValueError("Could not read the CSV file with any encoding")
        
        # Convert dates to pandas datetime objects before passing to clean_data
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # Clean and filter the data
        filtered_df, enrollment_col = clean_data(df, file_path, start_date_dt, end_date_dt)
        
        if len(filtered_df) == 0:
            logging.warning("No qualifying records found. No pixels will be fired.")
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
            
            if fire_pixel(transaction_id, enrolled_debt, enrollment_date):
                pixels_fired += 1
                total_debt += enrolled_debt
                total_revenue += (enrolled_debt * 0.04)
        
        logging.info("\n=== Summary ===")
        logging.info(f"File processed successfully with encoding: {successful_encoding}")
        logging.info(f"Total pixels fired successfully: {pixels_fired} out of {len(filtered_df)}")
        logging.info(f"Total Enrolled Debt processed: ${total_debt:,.2f}")
        logging.info(f"Total revenue share amount (4%): ${total_revenue:,.2f}")
        
        return {
            'success': True,
            'encoding': successful_encoding,
            'pixels_fired': pixels_fired,
            'total_attempted': len(filtered_df),
            'total_debt': total_debt,
            'total_revenue': total_revenue
        }
        
    except Exception as e:
        logging.error(f"Error processing NDR report: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("\nUsage: python ndr_revshare_pixel_firing.py <file_path> <start_date> <end_date>")
        print("\nArguments:")
        print("  file_path   : Path to the NDR report (CSV or XLSX)")
        print("  start_date  : Start date in YYYY-MM-DD format")
        print("  end_date    : End date in YYYY-MM-DD format")
        print("\nExample:")
        print("  python ndr_revshare_pixel_firing.py ndr_report.xlsx 2025-10-01 2025-10-15")
        print("\n**Note:** The script supports both CSV and XLSX file formats.")
        print("\nRequired columns in your CSV or XLSX:")
        print("- Affiliate SubID 1")
        print("- Enrollment Datetime (date column)")
        print("- Total Enrolled Debt (revenue amount)")
        sys.exit(1)
    
    file_path = sys.argv[1]
    start_date_str = sys.argv[2]
    end_date_str = sys.argv[3]
    
    # Validate file exists
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    
    # Validate date format
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD format.")
        sys.exit(1)
    
    # Validate date range
    if start_date > end_date:
        logging.error("Start date must be before or equal to end date.")
        sys.exit(1)
    
    # Process the report
    try:
        result = process_ndr_report(file_path, start_date, end_date)
        logging.info("\n✅ Process completed successfully!")
        sys.exit(0)
    except Exception as e:
        logging.error(f"\n❌ Process failed: {str(e)}")
        sys.exit(1)

