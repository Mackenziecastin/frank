# First try to import required packages with error handling
import sys
import os
from datetime import datetime, timedelta
import logging
import uuid
import re

# Optional import of chardet
CHARDET_AVAILABLE = False
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    pass  # We'll handle this gracefully

# Set up logging first
log_filename = f'adt_pixel_firing_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite the log file each run
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Log system info
logging.info("=== Starting ADT Pixel Firing Process ===")
logging.info(f"Python version: {sys.version}")
logging.info(f"Working directory: {os.getcwd()}")

# Try importing each required package with error handling
try:
    import pandas as pd
    logging.info("Successfully imported pandas")
except ImportError as e:
    logging.error(f"Failed to import pandas: {str(e)}")
    print(f"Error: Failed to import pandas. Please ensure it's installed: {str(e)}")
    print("Try running: pip install pandas==2.2.0")
    sys.exit(1)

try:
    import requests
    logging.info("Successfully imported requests")
except ImportError as e:
    logging.error(f"Failed to import requests: {str(e)}")
    print(f"Error: Failed to import requests. Please ensure it's installed: {str(e)}")
    print("Try running: pip install requests")
    sys.exit(1)

# We don't exit if chardet is missing, since it's optional
if CHARDET_AVAILABLE:
    logging.info("Successfully imported chardet (optional)")
else:
    logging.warning("chardet library not available. Automatic encoding detection will be limited.")
    logging.info("Consider installing with: pip install chardet")

# Print Python environment information for debugging
print("\nPython Environment Information:")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

def clean_data(df, file_path):
    """
    Clean and filter the data according to requirements.
    """
    try:
        # Extract date from filename (format: ADT_Athena_DLY_Lead_CallData_Direct_Agnts_YYYYMMDD.csv)
        filename = os.path.basename(file_path)
        report_date_str = filename.split('_')[-1].replace('.csv', '')
        report_date = datetime.strptime(report_date_str, '%Y%m%d').date()
        yesterday = report_date - timedelta(days=1)
        logging.info(f"Report date: {report_date.strftime('%Y-%m-%d')}, Using yesterday: {yesterday.strftime('%Y-%m-%d')}")
        
        # Convert Sale_Date to datetime if it's not already and remove any null values
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
        df = df.dropna(subset=['Sale_Date'])
        
        # Print initial count
        total_records = len(df)
        logging.info(f"\nStarting with {total_records} total records")
        
        # Log some sample dates for debugging
        sample_dates = df['Sale_Date'].head()
        logging.info("\nSample Sale_Date values:")
        for idx, date in enumerate(sample_dates):
            logging.info(f"Record {idx}: {date}")
        
        # Apply filters one by one and show counts
        # Remove health leads from Ln_of_Busn
        df['Ln_of_Busn'] = df['Ln_of_Busn'].fillna('')
        health_business_filter = ~df['Ln_of_Busn'].str.contains('Health', case=False, na=False)
        df_after_health_business = df[health_business_filter]
        logging.info(f"After excluding Health from Ln_of_Busn: {len(df_after_health_business)} records")
        
        # Remove US: Health from DNIS_BUSN_SEG_CD
        df_after_health_business['DNIS_BUSN_SEG_CD'] = df_after_health_business['DNIS_BUSN_SEG_CD'].fillna('')
        health_dnis_filter = ~df_after_health_business['DNIS_BUSN_SEG_CD'].str.contains('US: Health', case=False, na=False)
        df_after_health_dnis = df_after_health_business[health_dnis_filter]
        logging.info(f"After excluding US: Health from DNIS_BUSN_SEG_CD: {len(df_after_health_dnis)} records")
        
        # Filter for yesterday's date based on Sale_Date
        # Convert dates to string format for comparison to avoid timezone issues
        yesterday_str = yesterday.strftime('%Y-%m-%d')
        df_after_health_dnis['Sale_Date_Str'] = df_after_health_dnis['Sale_Date'].dt.strftime('%Y-%m-%d')
        date_filter = (df_after_health_dnis['Sale_Date_Str'] == yesterday_str)
        df_after_date = df_after_health_dnis[date_filter]
        
        # Log all unique dates in the dataset
        unique_dates = sorted(df_after_health_dnis['Sale_Date_Str'].unique())
        logging.info("\nAll unique dates in dataset:")
        for date in unique_dates:
            count = len(df_after_health_dnis[df_after_health_dnis['Sale_Date_Str'] == date])
            logging.info(f"Date {date}: {count} records")
        
        logging.info(f"\nLooking for records with date {yesterday_str}")
        logging.info(f"After filtering for yesterday ({yesterday_str}): {len(df_after_date)} records")
        
        # Log some filtered dates for debugging
        filtered_dates = df_after_date['Sale_Date'].head()
        logging.info("\nFiltered Sale_Date values:")
        for idx, date in enumerate(filtered_dates):
            logging.info(f"Record {idx}: {date}")
        
        # Check Lead_DNIS values before filtering
        unique_dnis = df_after_date['Lead_DNIS'].unique()
        logging.info("\nUnique Lead_DNIS values before filtering:")
        for dnis in unique_dnis:
            count = len(df_after_date[df_after_date['Lead_DNIS'] == dnis])
            logging.info(f"DNIS {dnis}: {count} records")
        
        df_after_date['Lead_DNIS'] = df_after_date['Lead_DNIS'].fillna('')
        
        # Filter for all allowed DNIS values and keep track separately
        web_dnis_codes = ['WEB0021011', 'WEB0021042', 'WEB0021044', 'WEB0021008']
        phone_dnis_codes = ['8669765334']
        
        web_dnis_filter = df_after_date['Lead_DNIS'].isin(web_dnis_codes)
        phone_dnis_filter = df_after_date['Lead_DNIS'].isin(phone_dnis_codes)
        
        df_web_dnis = df_after_date[web_dnis_filter].copy()
        df_phone_dnis = df_after_date[phone_dnis_filter].copy()
        
        # Add a column to track the source for pixel firing
        df_web_dnis['pixel_source'] = 'web'
        df_phone_dnis['pixel_source'] = 'phone'
        
        # Combine the dataframes
        df_after_lead_dnis = pd.concat([df_web_dnis, df_phone_dnis])
        
        logging.info(f"After filtering for Lead_DNIS:")
        logging.info(f"  Web DNIS codes {web_dnis_codes}: {len(df_web_dnis)} records")
        logging.info(f"  Phone DNIS codes {phone_dnis_codes}: {len(df_phone_dnis)} records")
        logging.info(f"  Total: {len(df_after_lead_dnis)} records")
        
        # Check Order Types before filtering
        unique_order_types = df_after_lead_dnis['Ordr_Type'].unique()
        logging.info("\nUnique Order Types before filtering:")
        for order_type in unique_order_types:
            count = len(df_after_lead_dnis[df_after_lead_dnis['Ordr_Type'] == order_type])
            logging.info(f"Order Type '{order_type}': {count} records")
        
        # Fill NA values for remaining filters
        df_after_lead_dnis['Ordr_Type'] = df_after_lead_dnis['Ordr_Type'].fillna('')
        df_after_lead_dnis['INSTALL_METHOD'] = df_after_lead_dnis['INSTALL_METHOD'].fillna('')
        
        order_type_filter = (
            df_after_lead_dnis['Ordr_Type'].str.contains('New', case=False, na=False) |
            df_after_lead_dnis['Ordr_Type'].str.contains('Resale', case=False, na=False)
        )
        filtered_df = df_after_lead_dnis[order_type_filter]
        logging.info(f"\nAfter filtering for New/Resale order types: {len(filtered_df)} records")
        
        # Check Install Methods before final split
        unique_install_methods = filtered_df['INSTALL_METHOD'].unique()
        logging.info("\nUnique Install Methods in final dataset:")
        for method in unique_install_methods:
            count = len(filtered_df[filtered_df['INSTALL_METHOD'] == method])
            logging.info(f"Install Method '{method}': {count} records")
        
        # Separate DIFM and DIY records
        difm_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIFM', case=False, na=False)]
        diy_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIY', case=False, na=False)]
        
        # Add detailed logging for each record
        logging.info("\nDetailed record analysis:")
        for idx, row in filtered_df.iterrows():
            logging.info(f"Record {idx}:")
            logging.info(f"  Install Method: {row['INSTALL_METHOD']}")
            logging.info(f"  Order Type: {row['Ordr_Type']}")
            logging.info(f"  Sale Date: {row['Sale_Date']}")
            logging.info(f"  Lead DNIS: {row['Lead_DNIS']}")
            logging.info(f"  Pixel Source: {row['pixel_source']}")
        
        # Count DIFM and DIY records by source
        web_difm_count = len(difm_records[difm_records['pixel_source'] == 'web'])
        web_diy_count = len(diy_records[diy_records['pixel_source'] == 'web'])
        phone_difm_count = len(difm_records[difm_records['pixel_source'] == 'phone'])
        phone_diy_count = len(diy_records[diy_records['pixel_source'] == 'phone'])
        
        logging.info(f"\nFinal counts:")
        logging.info(f"WEB0021011 (speedtrkzone.com):")
        logging.info(f"  DIFM Sales: {web_difm_count}")
        logging.info(f"  DIY Sales: {web_diy_count}")
        logging.info(f"8669765334 (trkfocus.com):")
        logging.info(f"  DIFM Sales: {phone_difm_count}")
        logging.info(f"  DIY Sales: {phone_diy_count}")
        
        # Drop the temporary date string column
        filtered_df = filtered_df.drop('Sale_Date_Str', axis=1)
        return filtered_df
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

# DNIS to pixel configuration mapping
DNIS_PIXEL_CONFIG = {
    'WEB0021011': {
        'url': 'https://speedtrkzone.com/m.ashx',
        'campaigns': {'DIFM': '91149', 'DIY': '91162'}
    },
    '8669765334': {
        'url': 'https://trkfocus.com/m.ashx',
        'campaigns': {'DIFM': '93166', 'DIY': '93168'}
    },
    'WEB0021042': {
        'url': 'https://trkfocus.com/m.ashx',
        'campaigns': {'DIFM': '95377', 'DIY': '95378'}
    },
    'WEB0021044': {
        'url': 'https://trkfocus.com/m.ashx',
        'campaigns': {'DIFM': '95379', 'DIY': '95380'}
    },
    'WEB0021008': {
        'url': 'https://trkfocus.com/m.ashx',
        'campaigns': {'DIFM': '95385', 'DIY': '95386'}
    }
}

def fire_pixel(transaction_id, install_method, sale_date, dnis_code):
    """
    Fire the pixel for a given transaction ID, install method, sale date, and DNIS code
    """
    try:
        # Get pixel configuration for this DNIS code
        if dnis_code not in DNIS_PIXEL_CONFIG:
            logging.error(f"Unknown DNIS code: {dnis_code}")
            return False
            
        config = DNIS_PIXEL_CONFIG[dnis_code]
        pixel_url = config['url']
        
        # Set campaign ID based on install method
        install_type = 'DIFM' if 'DIFM' in str(install_method).upper() else 'DIY'
        campaign_id = config['campaigns'][install_type]
        
        # Set the time to noon on the sale date
        pixel_datetime = sale_date.replace(hour=12, minute=0, second=0)
        iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        
        # Set up parameters
        params = {
            'o': '32022',
            'e': '565',
            'f': 'pb',
            't': transaction_id,
            'pubid': '42865',
            'campid': campaign_id,
            'dt': iso_datetime
        }
        
        # Make the request
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()
        
        logging.info(f"Fired {install_type} pixel for {dnis_code} successfully - Transaction ID: {transaction_id}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to fire {install_type} pixel for {dnis_code} - Transaction ID: {transaction_id} - Error: {str(e)}")
        return False


def process_adt_report(file_path):
    """
    Main function to process the ADT report and fire pixels
    """
    try:
        logging.info("\n=== ADT Pixel Firing Process Started ===")
        
        # Try to detect encoding first
        detected_encoding = detect_encoding(file_path)
        
        # Try different encodings if needed
        encodings_to_try = [detected_encoding] if detected_encoding else []
        encodings_to_try.extend(['utf-8', 'latin1', 'cp1252', 'ISO-8859-1'])
        
        # Remove duplicates while preserving order
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        df = None
        successful_encoding = None
        
        for encoding in encodings_to_try:
            try:
                logging.info(f"Attempting to read file with {encoding} encoding")
                df = pd.read_csv(file_path, encoding=encoding)
                successful_encoding = encoding
                logging.info(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                logging.warning(f"Failed to read with {encoding} encoding: {str(e)}")
            except Exception as e:
                logging.warning(f"Error reading CSV with {encoding} encoding: {str(e)}")
        
        if df is None:
            raise Exception("Failed to read CSV file with any of the attempted encodings")
        
        # Clean and filter the data
        filtered_df = clean_data(df, file_path)
        
        if len(filtered_df) == 0:
            logging.info("No qualifying sales found to process.")
            return
        
        # Initialize success counters by DNIS code
        success_counters = {}
        total_counters = {}
        
        # Initialize counters for all DNIS codes
        for dnis_code in DNIS_PIXEL_CONFIG.keys():
            success_counters[dnis_code] = {'DIFM': 0, 'DIY': 0}
            total_counters[dnis_code] = {'DIFM': 0, 'DIY': 0}
        
        # Process each record
        for _, row in filtered_df.iterrows():
            transaction_id = f"ADT_{row['Sale_Date'].strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
            dnis_code = row['Lead_DNIS']
            is_difm = 'DIFM' in str(row['INSTALL_METHOD']).upper()
            install_type = 'DIFM' if is_difm else 'DIY'
            
            # Count total records for this DNIS/install type combination
            total_counters[dnis_code][install_type] += 1
            
            # Fire pixel using the new unified function
            if fire_pixel(transaction_id, install_type, row['Sale_Date'], dnis_code):
                success_counters[dnis_code][install_type] += 1
        
        # Log summary by DNIS code
        logging.info("\n=== Summary ===")
        logging.info(f"File processed successfully with encoding: {successful_encoding}")
        
        total_successful = 0
        total_processed = 0
        
        for dnis_code in DNIS_PIXEL_CONFIG.keys():
            config = DNIS_PIXEL_CONFIG[dnis_code]
            difm_success = success_counters[dnis_code]['DIFM']
            diy_success = success_counters[dnis_code]['DIY']
            difm_total = total_counters[dnis_code]['DIFM']
            diy_total = total_counters[dnis_code]['DIY']
            
            if difm_total > 0 or diy_total > 0:  # Only show DNIS codes that had records
                logging.info(f"\n{dnis_code} ({config['url']}):")
                if difm_total > 0:
                    logging.info(f"  DIFM pixels fired successfully: {difm_success} out of {difm_total}")
                if diy_total > 0:
                    logging.info(f"  DIY pixels fired successfully: {diy_success} out of {diy_total}")
                
                total_successful += difm_success + diy_success
                total_processed += difm_total + diy_total
        
        logging.info(f"\nTotal pixels fired: {total_successful}")
        logging.info(f"Total records processed: {total_processed}")
        
    except Exception as e:
        logging.error(f"Error processing ADT report: {str(e)}")
        raise

def detect_encoding(file_path):
    """
    Attempt to detect the file encoding
    """
    try:
        # If chardet is available, use it for better detection
        if CHARDET_AVAILABLE:
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(min(1024 * 1024, os.path.getsize(file_path)))  # Read up to 1MB
            
            # Detect encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            logging.info(f"Detected encoding: {encoding} with {confidence:.1%} confidence")
            return encoding
        else:
            # Basic detection without chardet
            # Check for BOM markers
            with open(file_path, 'rb') as f:
                raw = f.read(4)
                
            if raw.startswith(b'\xef\xbb\xbf'):
                logging.info("Detected UTF-8 BOM encoding")
                return 'utf-8-sig'
            elif raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
                logging.info("Detected UTF-16 encoding")
                return 'utf-16'
            
            # Default to utf-8 as a starting point
            logging.info("No specific encoding detected, will try common encodings")
            return None
            
    except Exception as e:
        logging.error(f"Error detecting encoding: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("\nError: Missing report file argument")
        logging.info("\nHow to use this script:")
        logging.info("1. Make sure your ADT Athena report file is in the same folder as this script")
        logging.info("2. The file should be named like: ADT_Athena_DLY_Lead_CallData_Direct_Agnts_[date].csv")
        logging.info("3. Run the script with: python3 adt_pixel_firing.py your_report_file.csv")
        logging.info("\nExample:")
        logging.info("python3 adt_pixel_firing.py ADT_Athena_DLY_Lead_CallData_Direct_Agnts_20250326.csv")
        logging.info("\nIf you're having issues with file encoding:")
        logging.info("This script will try multiple encodings automatically.")
        logging.info("For better encoding detection, you can install the optional chardet library:")
        logging.info("pip install chardet")
        logging.info("\nIf you're still having issues, try converting your file to UTF-8 encoding:")
        logging.info("1. Open the file in Excel")
        logging.info("2. Save As -> CSV UTF-8 (Comma delimited) (*.csv)")
        sys.exit(1)
    
    report_path = sys.argv[1]
    logging.info(f"\nAttempting to process file: {report_path}")
    logging.info(f"Absolute path: {os.path.abspath(report_path)}")
    
    # Check if file exists
    if not os.path.exists(report_path):
        logging.error(f"\nError: File '{report_path}' not found!")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info("\nPlease check that:")
        logging.info("1. The file name is correct")
        logging.info("2. The file is in the same folder as this script")
        logging.info("3. You have the correct file permissions")
        logging.info("\nCurrent directory contents:")
        try:
            files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if files:
                logging.info("\nAvailable CSV files:")
                for f in files:
                    logging.info(f"- {f}")
            else:
                logging.info("No CSV files found in current directory")
        except Exception as e:
            logging.error(f"Could not list directory contents: {str(e)}")
        sys.exit(1)
    
    # Check if file is readable
    try:
        with open(report_path, 'rb') as f:  # Open in binary mode to check for non-UTF-8 characters
            first_few_lines = f.read(4096)  # Read first 4KB
            
            # Check if file starts with BOM (Byte Order Mark)
            has_bom = False
            if first_few_lines.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
                has_bom = True
                logging.info("File has UTF-8 BOM marker")
            
            # Check for potential encoding issues
            try:
                first_few_lines.decode('utf-8')
            except UnicodeDecodeError:
                logging.warning("File contains non-UTF-8 characters. Will try alternative encodings.")
                
        if not first_few_lines:
            logging.error(f"\nError: File '{report_path}' is empty!")
            sys.exit(1)
    except Exception as e:
        logging.error(f"\nError: Could not read file '{report_path}': {str(e)}")
        logging.info("Please check file permissions and format")
        sys.exit(1)
    
    try:
        process_adt_report(report_path)
    except Exception as e:
        logging.error(f"Error processing report: {str(e)}")
        import traceback
        logging.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

    