# LaserAway Revshare Pixel Firing Script
# Fires pixels for Schemathics on LaserAway based on Net Sales revenue share
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
log_filename = f'laseraway_revshare_pixel_firing_{datetime.now().strftime("%Y%m%d")}.log'
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
logging.info("=== Starting LaserAway Revshare Pixel Firing Process ===")
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

def clean_data(df, file_path, start_date, end_date):
    """
    Clean and filter the data according to requirements.
    """
    try:
        logging.info(f"\nStarting with {len(df)} total records")
        
        # Convert Purchased Date column to datetime if it's not already and remove any null values
        df['Purchased Date'] = pd.to_datetime(df['Purchased Date'], errors='coerce')
        df = df.dropna(subset=['Purchased Date'])
        
        # Log some sample dates for debugging
        sample_dates = df['Purchased Date'].head()
        logging.info("\nSample Purchased values:")
        for idx, date in enumerate(sample_dates):
            logging.info(f"Record {idx}: {date}")
        
        # Filter for affiliate_directagent_subid1 = 42865
        df['affiliate_directagent_subid1'] = df['affiliate_directagent_subid1'].fillna('')
        affiliate_filter = df['affiliate_directagent_subid1'] == '42865'
        df_after_affiliate = df[affiliate_filter]
        logging.info(f"After filtering for affiliate_directagent_subid1 = 42865: {len(df_after_affiliate)} records")
        
        if len(df_after_affiliate) == 0:
            logging.info("No records found with affiliate_directagent_subid1 = 42865")
            return df_after_affiliate
        
        # Filter for date range (convert date objects to datetime for comparison)
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        date_filter = (df_after_affiliate['Purchased Date'] >= start_datetime) & (df_after_affiliate['Purchased Date'] <= end_datetime)
        df_after_date = df_after_affiliate[date_filter]
        
        # Log all unique dates in the dataset
        unique_dates = sorted(df_after_affiliate['Purchased Date'].dt.date.unique())
        logging.info("\nAll unique dates in dataset:")
        for date in unique_dates:
            count = len(df_after_affiliate[df_after_affiliate['Purchased Date'].dt.date == date])
            logging.info(f"Date {date}: {count} records")
        
        logging.info(f"\nLooking for records between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        logging.info(f"After filtering for date range: {len(df_after_date)} records")
        
        # Check Net Sales column
        if 'Net Sales' not in df_after_date.columns:
            logging.error("Net Sales column not found in the dataset")
            raise ValueError("Net Sales column is required but not found")
        
        # Convert Net Sales to numeric, handling any non-numeric values
        df_after_date['Net Sales'] = pd.to_numeric(df_after_date['Net Sales'], errors='coerce')
        df_after_date = df_after_date.dropna(subset=['Net Sales'])
        
        logging.info(f"After filtering for valid Net Sales: {len(df_after_date)} records")
        
        # Log some sample Net Sales values
        sample_sales = df_after_date['Net Sales'].head()
        logging.info("\nSample Net Sales values:")
        for idx, sales in enumerate(sample_sales):
            logging.info(f"Record {idx}: {sales}")
        
        return df_after_date
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

def fire_pixel(transaction_id, net_sales_amount, purchase_date):
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
        
        logging.info(f"Fired pixel successfully - Transaction ID: {transaction_id}, Revenue Amount: {revenue_amount_formatted}, Date: {iso_datetime}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to fire pixel - Transaction ID: {transaction_id} - Error: {str(e)}")
        return False

def process_laseraway_report(file_path, start_date, end_date):
    """
    Main function to process the LaserAway report and fire pixels
    """
    try:
        logging.info("\n=== LaserAway Revshare Pixel Firing Process Started ===")
        logging.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
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
        filtered_df = clean_data(df, file_path, start_date, end_date)
        
        if len(filtered_df) == 0:
            logging.info("No qualifying sales found to process.")
            return
        
        # Initialize counters
        successful_pixels = 0
        total_pixels = 0
        total_revenue = 0
        
        # Process each record
        for _, row in filtered_df.iterrows():
            transaction_id = f"LASERAWAY_{row['Purchased Date'].strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
            net_sales = row['Net Sales']
            purchase_date = row['Purchased Date']
            
            total_pixels += 1
            total_revenue += net_sales
            
            # Fire pixel
            if fire_pixel(transaction_id, net_sales, purchase_date):
                successful_pixels += 1
        
        # Log summary
        logging.info("\n=== Summary ===")
        logging.info(f"File processed successfully with encoding: {successful_encoding}")
        logging.info(f"Total pixels fired successfully: {successful_pixels} out of {total_pixels}")
        logging.info(f"Total Net Sales processed: ${total_revenue:.2f}")
        logging.info(f"Total revenue share amount: ${total_revenue / 1.75:.2f}")
        
    except Exception as e:
        logging.error(f"Error processing LaserAway report: {str(e)}")
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
    if len(sys.argv) != 4:
        logging.error("\nError: Missing required arguments")
        logging.info("\nHow to use this script:")
        logging.info("1. Make sure your LaserAway report file is in the same folder as this script")
        logging.info("2. Run the script with: python laseraway_revshare_pixel_firing.py your_report_file.csv start_date end_date")
        logging.info("\nExample:")
        logging.info("python laseraway_revshare_pixel_firing.py laseraway_report.csv 2024-06-01 2024-06-30")
        logging.info("\nDate format: YYYY-MM-DD")
        logging.info("\nRequired columns in your CSV:")
        logging.info("- affiliate_directagent_subid1")
        logging.info("- Purchased (date column)")
        logging.info("- Net Sales (revenue amount)")
        sys.exit(1)
    
    report_path = sys.argv[1]
    start_date_str = sys.argv[2]
    end_date_str = sys.argv[3]
    
    # Parse dates
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError as e:
        logging.error(f"Invalid date format: {str(e)}")
        logging.info("Please use YYYY-MM-DD format for dates")
        sys.exit(1)
    
    logging.info(f"\nAttempting to process file: {report_path}")
    logging.info(f"Absolute path: {os.path.abspath(report_path)}")
    logging.info(f"Date range: {start_date_str} to {end_date_str}")
    
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
        process_laseraway_report(report_path, start_date, end_date)
    except Exception as e:
        logging.error(f"Error processing report: {str(e)}")
        import traceback
        logging.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)
