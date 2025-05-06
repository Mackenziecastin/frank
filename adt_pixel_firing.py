# First try to import required packages with error handling
import sys
import os
from datetime import datetime, timedelta
import logging
import uuid
import re
import requests

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
    logging.info("Successfully imported all other required packages")
except ImportError as e:
    logging.error(f"Failed to import required package: {str(e)}")
    print(f"Error: Failed to import required package: {str(e)}")
    sys.exit(1)

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
        # Extract date from filename or use current date
        try:
            # First try direct extraction from filename pattern
            filename = os.path.basename(file_path)
            if '_20' in filename:  # Look for pattern like _20230415 in filename
                date_part = filename.split('_')[-1].split('.')[0]
                if len(date_part) == 8 and date_part.isdigit():
                    report_date_str = date_part
                    logging.info(f"Extracted date from filename: {report_date_str}")
                else:
                    raise ValueError("Date part doesn't match expected format")
            else:
                # Use today's date as fallback
                report_date_str = datetime.now().strftime('%Y%m%d')
                logging.info(f"Using current date: {report_date_str}")
        except Exception as e:
            # Fall back to current date on any error
            logging.warning(f"Error extracting date from filename: {str(e)}")
            report_date_str = datetime.now().strftime('%Y%m%d')
            logging.info(f"Using current date as fallback: {report_date_str}")
            
        # Parse the report date
        report_date = datetime.strptime(report_date_str, '%Y%m%d').date()
        yesterday = report_date - timedelta(days=1)
        logging.info(f"Report date: {report_date}, Processing data for: {yesterday}")
        
        # Convert Sale_Date to datetime if it's not already and remove any null values
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
        df = df.dropna(subset=['Sale_Date'])
        
        # Print initial count
        total_records = len(df)
        logging.info(f"\nStarting with {total_records} total records")
        
        # Apply filters one by one and show counts
        # Remove health leads from Ln_of_Busn
        health_business_filter = ~df['Ln_of_Busn'].str.contains('Health', case=False, na=False)
        df_after_health_business = df[health_business_filter]
        logging.info(f"After excluding Health from Ln_of_Busn: {len(df_after_health_business)} records")
        
        # Remove US: Health from DNIS_BUSN_SEG_CD
        health_dnis_filter = ~df_after_health_business['DNIS_BUSN_SEG_CD'].str.contains('US: Health', case=False, na=False)
        df_after_health_dnis = df_after_health_business[health_dnis_filter]
        logging.info(f"After excluding US: Health from DNIS_BUSN_SEG_CD: {len(df_after_health_dnis)} records")
        
        # Filter for yesterday's date based on Sale_Date
        date_filter = (df_after_health_dnis['Sale_Date'].dt.date == yesterday)
        df_after_date = df_after_health_dnis[date_filter]
        logging.info(f"After filtering for yesterday ({yesterday}): {len(df_after_date)} records")
        
        dnis_filter = (df_after_date['Lead_DNIS'] == 'WEB0021011')
        df_after_lead_dnis = df_after_date[dnis_filter]
        logging.info(f"After filtering for Lead_DNIS 'WEB0021011': {len(df_after_lead_dnis)} records")
        
        # Log details about records before order type filtering
        logging.info("\nChecking order types before filtering:")
        for idx, row in df_after_lead_dnis.iterrows():
            logging.info(f"Record {idx}: Order Type = '{row['Ordr_Type']}'")
        
        order_type_filter = (
            df_after_lead_dnis['Ordr_Type'].str.contains('New', case=False, na=False) |
            df_after_lead_dnis['Ordr_Type'].str.contains('Resale', case=False, na=False)
        )
        filtered_df = df_after_lead_dnis[order_type_filter]
        logging.info(f"\nAfter filtering for New/Resale order types: {len(filtered_df)} records")
        
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
        
        # Count DIFM and DIY records
        difm_count = len(difm_records)
        diy_count = len(diy_records)
        
        logging.info(f"\nFinal counts:")
        logging.info(f"DIFM Sales: {difm_count}")
        logging.info(f"DIY Sales: {diy_count}")
        
        return filtered_df
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

def fire_pixel(transaction_id, install_method, sale_date):
    """
    Fire the pixel for a given transaction ID, install method, and sale date
    """
    try:
        # Use the correct Cake pixel URL
        pixel_url = "https://speedtrkzone.com/m.ashx"
        
        # Set campaign ID based on install method
        if 'DIFM' in str(install_method).upper():
            campaign_id = "91149"
        else:
            campaign_id = "91162"  # DIY
        
        # Convert sale_date to ISO 8601 format with timezone
        # Set the time to noon on the sale date
        try:
            # First ensure sale_date is a datetime object
            if isinstance(sale_date, pd.Timestamp):
                pixel_datetime = sale_date.to_pydatetime()
            else:
                pixel_datetime = pd.to_datetime(sale_date)
            
            # Set time to noon
            pixel_datetime = pixel_datetime.replace(hour=12, minute=0, second=0)
            
            # Format as ISO 8601 with UTC timezone
            iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        except Exception as e:
            logging.error(f"Error formatting date: {str(e)}. Using current datetime.")
            # Use current time as fallback
            pixel_datetime = datetime.now().replace(hour=12, minute=0, second=0)
            iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        
        # The exact parameters that worked before
        params = {
            'o': '32022',
            'e': '565',
            'f': 'pb',
            't': transaction_id,
            'pubid': '42865',
            'campid': campaign_id,
            'dt': iso_datetime
        }
        
        # Construct and log the complete URL
        url_with_params = requests.Request('GET', pixel_url, params=params).prepare().url
        logging.info(f"  Firing pixel: {url_with_params}")
        
        # Send the request to Cake
        try:
            response = requests.get(url_with_params, timeout=30)  # Use the full URL with params
            
            # Check if successful (HTTP 200-299)
            if 200 <= response.status_code < 300:
                logging.info(f"✓ Pixel fired successfully:")
                logging.info(f"  Transaction ID: {transaction_id}")
                logging.info(f"  Install Method: {install_method}")
                logging.info(f"  Campaign ID: {campaign_id}")
                logging.info(f"  Sale Date: {sale_date}")
                logging.info(f"  Response: {response.status_code}, {response.text[:100]}")
                return True
            else:
                logging.error(f"✗ Pixel firing failed with status {response.status_code}:")
                logging.error(f"  URL: {url_with_params}")
                logging.error(f"  Response: {response.text[:100]}")
                return False
                
        except requests.exceptions.RequestException as e:
            logging.error(f"✗ Request error firing pixel:")
            logging.error(f"  URL: {url_with_params}")
            logging.error(f"  Error: {str(e)}")
            return False
            
    except Exception as e:
        logging.error(f"Unexpected error in fire_pixel: {str(e)}")
        return False

def process_adt_report(file_path):
    """
    Main function to process the ADT report and fire pixels
    """
    try:
        logging.info("\n=== ADT Pixel Firing Process Started ===")
        logging.info(f"Reading file: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                logging.info(f"\nTrying to read file with {encoding} encoding...")
                df = pd.read_csv(file_path, encoding=encoding)
                logging.info(f"Successfully read file with {encoding} encoding!")
                break
            except UnicodeDecodeError:
                logging.info(f"Failed to read with {encoding} encoding")
                continue
        
        if df is None:
            raise ValueError("Could not read the file with any of the attempted encodings")
        
        # Clean and filter the data
        filtered_df = clean_data(df, file_path)
        
        # Count of sales to process
        total_sales = len(filtered_df)
        
        # If we don't have any filtered records, return early
        if total_sales == 0:
            logging.info("\n=== No qualifying sales found for pixel firing ===")
            return
            
        # Get the first date to use in the log
        first_sale_date = filtered_df['Sale_Date'].iloc[0].date() if len(filtered_df) > 0 else "unknown date"
        logging.info(f"\n=== Found {total_sales} qualifying sales for date: {first_sale_date} ===")
        
        # Count DIFM and DIY sales
        difm_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIFM', case=False, na=False)]
        diy_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIY', case=False, na=False)]
        difm_sales = len(difm_records)
        diy_sales = len(diy_records)
        
        logging.info(f"\n=== Firing {difm_sales} DIFM pixels and {diy_sales} DIY pixels ===")
        
        # Fire pixels for each sale
        logging.info("\nFiring pixels...")
        successful_difm = 0
        successful_diy = 0
        
        # Fire DIFM pixels
        if difm_sales > 0:
            logging.info(f"\n=== Firing DIFM Pixels (Campaign ID: 91149) ===")
            for idx, row in difm_records.iterrows():
                # Generate a unique transaction ID
                transaction_id = f"ADT_{row['Sale_Date'].strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
                
                # Fire the pixel
                logging.info(f"  Firing DIFM pixel {successful_difm + 1} of {difm_sales}...")
                if fire_pixel(transaction_id, row['INSTALL_METHOD'], row['Sale_Date']):
                    successful_difm += 1
        
        # Fire DIY pixels
        if diy_sales > 0:
            logging.info(f"\n=== Firing DIY Pixels (Campaign ID: 91162) ===")
            for idx, row in diy_records.iterrows():
                # Generate a unique transaction ID
                transaction_id = f"ADT_{row['Sale_Date'].strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
                
                # Fire the pixel
                logging.info(f"  Firing DIY pixel {successful_diy + 1} of {diy_sales}...")
                if fire_pixel(transaction_id, row['INSTALL_METHOD'], row['Sale_Date']):
                    successful_diy += 1
        
        # Print summary
        logging.info("\n=== Pixel Firing Summary ===")
        logging.info(f"Date processed: {first_sale_date}")
        logging.info(f"DIFM Pixels: {successful_difm} of {difm_sales} fired successfully")
        logging.info(f"DIY Pixels: {successful_diy} of {diy_sales} fired successfully")
        logging.info(f"Total Pixels: {successful_difm + successful_diy} of {total_sales} fired successfully")
        logging.info(f"Check the log file for detailed information: {log_filename}")
        logging.info("\n=== Pixel Firing Complete ===")
        
    except Exception as e:
        logging.error(f"Error processing ADT report: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("\nError: Missing report file argument")
        logging.info("\nHow to use this script:")
        logging.info("1. Make sure your ADT Athena report file is in the same folder as this script")
        logging.info("2. The file should be named like: ADT_Athena_DLY_Lead_CallData_Direct_Agnts_[date].csv")
        logging.info("3. Run the script with: python3 adt_pixel_firing.py your_report_file.csv")
        logging.info("\nExample:")
        logging.info("python3 adt_pixel_firing.py ADT_Athena_DLY_Lead_CallData_Direct_Agnts_20250326.csv")
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
        with open(report_path, 'r') as f:
            first_line = f.readline()
        if not first_line:
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

    