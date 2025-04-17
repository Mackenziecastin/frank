import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import uuid
import os
import sys
import re

# Set up logging
logging.basicConfig(
    filename=f'adt_pixel_firing_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_data(df):
    """
    Clean and filter the data according to requirements:
    1. Filter out any "Health" rows in the Ln_of_Busn column
    2. Filter out any US: Health rows in the DNIS_BUSN_SEG_CD column
    3. Filter for yesterday's date based on the report filename date
    4. Filter for WEB0021011 in Lead_DNIS
    5. Filter for order types containing 'New' or 'Resale'
    """
    try:
        # Get the report filename from sys.argv[1]
        report_filename = sys.argv[1]
        
        # Extract date from filename (assuming format: *_YYYYMMDD.csv)
        date_match = re.search(r'(\d{8})', report_filename)
        if not date_match:
            raise ValueError("Could not extract date from filename. Expected format: *_YYYYMMDD.csv")
        
        report_date = datetime.strptime(date_match.group(1), '%Y%m%d').date()
        yesterday = report_date - timedelta(days=1)
        logging.info(f"Report date: {report_date}, Using yesterday's date: {yesterday}")
        
        # Convert Sale_Date to datetime if it's not already and remove any null values
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
        df = df.dropna(subset=['Sale_Date'])
        
        # Print initial count
        total_records = len(df)
        print(f"\nStarting with {total_records} total records")
        
        # Apply filters one by one and show counts
        # Remove health leads from Ln_of_Busn
        health_business_filter = ~df['Ln_of_Busn'].str.contains('Health', case=False, na=False)
        df_after_health_business = df[health_business_filter]
        print(f"After excluding Health from Ln_of_Busn: {len(df_after_health_business)} records")
        
        # Remove US: Health from DNIS_BUSN_SEG_CD
        health_dnis_filter = ~df_after_health_business['DNIS_BUSN_SEG_CD'].str.contains('US: Health', case=False, na=False)
        df_after_health_dnis = df_after_health_business[health_dnis_filter]
        print(f"After excluding US: Health from DNIS_BUSN_SEG_CD: {len(df_after_health_dnis)} records")
        
        # Filter for yesterday's date based on Sale_Date
        date_filter = (df_after_health_dnis['Sale_Date'].dt.date == yesterday)
        df_after_date = df_after_health_dnis[date_filter]
        print(f"After filtering for yesterday ({yesterday.strftime('%Y-%m-%d')}): {len(df_after_date)} records")
        
        dnis_filter = (df_after_date['Lead_DNIS'] == 'WEB0021011')
        df_after_lead_dnis = df_after_date[dnis_filter]
        print(f"After filtering for Lead_DNIS 'WEB0021011': {len(df_after_lead_dnis)} records")
        
        # Log details about records before order type filtering
        print("\nChecking order types before filtering:")
        for idx, row in df_after_lead_dnis.iterrows():
            print(f"Record {idx}: Order Type = '{row['Ordr_Type']}'")
        
        order_type_filter = (
            df_after_lead_dnis['Ordr_Type'].str.contains('New', case=False, na=False) |
            df_after_lead_dnis['Ordr_Type'].str.contains('Resale', case=False, na=False)
        )
        filtered_df = df_after_lead_dnis[order_type_filter]
        print(f"\nAfter filtering for New/Resale order types: {len(filtered_df)} records")
        
        # Log details about which records were kept vs filtered
        print("\nOrder types that were filtered out:")
        filtered_out = df_after_lead_dnis[~order_type_filter]
        for idx, row in filtered_out.iterrows():
            print(f"Filtered out - Record {idx}: Order Type = '{row['Ordr_Type']}'")
        
        if len(filtered_df) == 0:
            print(f"\nNo records matched all criteria. Showing sample of Lead_DNIS 'WEB0021011' records from {yesterday.strftime('%Y-%m-%d')}:")
            web_records = df[
                (df['Lead_DNIS'] == 'WEB0021011') & 
                (df['Sale_Date'].dt.date == yesterday)
            ]
            if len(web_records) > 0:
                print("\nSample record:")
                sample = web_records.iloc[0]
                print(f"Sale_Date: {sample['Sale_Date']}")
                print(f"Ln_of_Busn: {sample['Ln_of_Busn']}")
                print(f"DNIS_BUSN_SEG_CD: {sample['DNIS_BUSN_SEG_CD']}")
                print(f"Ordr_Type: {sample['Ordr_Type']}")
                print(f"INSTALL_METHOD: {sample['INSTALL_METHOD']}")
            else:
                print(f"No records found with Lead_DNIS 'WEB0021011' for {yesterday.strftime('%Y-%m-%d')}")
        
        # Debug: Print all install methods to verify counting
        print("\nDebug: All install methods in filtered data:")
        for idx, row in filtered_df.iterrows():
            print(f"Record {idx}: Install Method = '{row['INSTALL_METHOD']}'")
        
        # Separate DIFM and DIY records
        difm_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIFM', case=False, na=False)]
        diy_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIY', case=False, na=False)]
        
        # Count DIFM and DIY records
        difm_count = len(difm_records)
        diy_count = len(diy_records)
        
        # Combine all records
        filtered_df = pd.concat([difm_records, diy_records])
        
        print(f"\nFinal counts:")
        print(f"DIFM Sales: {difm_count}")
        print(f"DIY Sales: {diy_count}")
        
        logging.info(f"Data cleaned successfully. Found {len(filtered_df)} qualifying sales for {yesterday.strftime('%Y-%m-%d')}.")
        return filtered_df
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

def fire_pixel(transaction_id, install_method, sale_date):
    """
    Fire the pixel for a given transaction ID, install method, and sale date
    """
    pixel_url = "https://speedtrkzone.com/m.ashx"
    # Set campaign ID based on install method
    campaign_id = "91149" if "DIFM" in str(install_method).upper() else "91162"
    
    # Convert sale_date to ISO 8601 format with timezone
    # Set the time to noon on the sale date
    pixel_datetime = sale_date.replace(hour=12, minute=0, second=0)
    # Format as ISO 8601 with UTC timezone
    iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    
    # Debug: Print detailed timestamp information
    print(f"\nDebug Timestamp Info:")
    print(f"  Sale Date: {sale_date}")
    print(f"  Using DateTime: {pixel_datetime}")
    print(f"  ISO 8601 Format: {iso_datetime}")
    
    params = {
        'o': '32022',
        'e': '565',
        'f': 'pb',
        't': transaction_id,
        'pubid': '42865',
        'campid': campaign_id,
        'dt': iso_datetime
    }
    
    # Construct and print the complete URL
    url_with_params = requests.Request('GET', pixel_url, params=params).prepare().url
    print(f"  Complete URL: {url_with_params}")
    
    try:
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Log the exact request details
        logging.info(f"Pixel fired successfully:")
        logging.info(f"  Transaction ID: {transaction_id}")
        logging.info(f"  Install Method: {install_method}")
        logging.info(f"  Campaign ID: {campaign_id}")
        logging.info(f"  Sale Date: {sale_date}")
        logging.info(f"  ISO DateTime: {iso_datetime}")
        logging.info(f"  URL: {url_with_params}")
        logging.info(f"  Response Status: {response.status_code}")
        logging.info(f"  Response Headers: {dict(response.headers)}")
        logging.info(f"  Response Content: {response.text[:200]}...")  # Log first 200 chars of response
        
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error firing pixel:")
        logging.error(f"  Transaction ID: {transaction_id}")
        logging.error(f"  Install Method: {install_method}")
        logging.error(f"  Sale Date: {sale_date}")
        logging.error(f"  ISO DateTime: {iso_datetime}")
        logging.error(f"  URL: {url_with_params}")
        logging.error(f"  Error: {str(e)}")
        return False

def process_adt_report(file_path):
    """
    Main function to process the ADT report and fire pixels
    """
    try:
        # Read the CSV file with different encodings
        print("\n=== ADT Pixel Firing Process Started ===")
        print(f"Reading file: {file_path}")
        logging.info(f"Reading file: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        successful_encoding = None
        
        for encoding in encodings:
            try:
                print(f"\nTrying to read file with {encoding} encoding...")
                df = pd.read_csv(file_path, encoding=encoding)
                successful_encoding = encoding
                print(f"Successfully read file with {encoding} encoding!")
                break
            except UnicodeDecodeError:
                print(f"Failed to read with {encoding} encoding, trying next...")
                continue
        
        if df is None:
            raise Exception("Could not read the file with any of the supported encodings")
        
        # Clean and filter the data
        filtered_df = clean_data(df)
        
        # Count of sales to process
        total_sales = len(filtered_df)
        sale_date = filtered_df['Sale_Date'].iloc[0] if total_sales > 0 else None
        sale_date_str = sale_date.strftime('%Y-%m-%d') if sale_date is not None else None
        print(f"\nFound {total_sales} qualifying sales for date: {sale_date_str}")
        
        # Count by install method
        difm_count = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIFM', case=False, na=False)].shape[0]
        diy_count = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIY', case=False, na=False)].shape[0]
        
        print(f"DIFM Sales: {difm_count}")
        print(f"DIY Sales: {diy_count}")
        logging.info(f"Found {total_sales} sales to process (DIFM: {difm_count}, DIY: {diy_count})")
        
        if total_sales == 0:
            print("\nNo qualifying sales found to process. Check the log file for details.")
            return
        
        # Fire pixel for each sale
        print("\nFiring pixels...")
        successful_fires = {'DIFM': 0, 'DIY': 0}
        for idx, row in filtered_df.iterrows():
            # Generate a unique transaction ID using sale date and UUID
            sale_date_str = row['Sale_Date'].strftime('%Y%m%d')
            transaction_id = f"ADT_{sale_date_str}_{str(uuid.uuid4())[:8]}"
            install_method = row['INSTALL_METHOD']
            
            # Determine install method category
            category = 'DIFM' if 'DIFM' in str(install_method).upper() else 'DIY'
            current_count = successful_fires['DIFM'] + successful_fires['DIY'] + 1
            
            print(f"  Firing pixel {current_count} of {total_sales} ({category})...", end="")
            if fire_pixel(transaction_id, install_method, row['Sale_Date']):
                successful_fires[category] += 1
                print(" ✓")
            else:
                print(" ✗")
        
        # Log summary
        summary = f"\nProcessing complete!\n" \
                 f"Date processed: {sale_date_str}\n" \
                 f"Successfully fired DIFM pixels: {successful_fires['DIFM']} out of {difm_count}\n" \
                 f"Successfully fired DIY pixels: {successful_fires['DIY']} out of {diy_count}\n" \
                 f"Total pixels fired: {sum(successful_fires.values())} out of {total_sales}\n" \
                 f"Check the log file for detailed information: adt_pixel_firing_{datetime.now().strftime('%Y%m%d')}.log"
        
        print("\n=== Summary ===")
        print(summary)
        logging.info(summary)
        
    except Exception as e:
        error_msg = f"Error processing ADT report: {str(e)}"
        print(f"\n❌ Error: {error_msg}")
        print("Check the log file for more details.")
        logging.error(error_msg)
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nError: Missing report file argument")
        print("\nHow to use this script:")
        print("1. Make sure your ADT Athena report file is in the same folder as this script")
        print("2. The file should be named like: ADT_Athena_DLY_Lead_CallData_Direct_Agnts_[date].csv")
        print("3. Run the script with: python3 adt_pixel_firing.py your_report_file.csv")
        print("\nExample:")
        print("python3 adt_pixel_firing.py ADT_Athena_DLY_Lead_CallData_Direct_Agnts_20250326.csv")
        sys.exit(1)
    
    report_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(report_path):
        print(f"\nError: File '{report_path}' not found!")
        print("\nPlease check that:")
        print("1. The file name is correct")
        print("2. The file is in the same folder as this script")
        print("3. You have the correct file permissions")
        print("\nCurrent directory contents:")
        try:
            files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if files:
                print("\nAvailable CSV files:")
                for f in files:
                    print(f"- {f}")
            else:
                print("No CSV files found in current directory")
        except Exception as e:
            print(f"Could not list directory contents: {str(e)}")
        sys.exit(1)
    
    # Check if file is readable
    try:
        with open(report_path, 'r') as f:
            first_line = f.readline()
        if not first_line:
            print(f"\nError: File '{report_path}' is empty!")
            sys.exit(1)
    except Exception as e:
        print(f"\nError: Could not read file '{report_path}': {str(e)}")
        print("Please check file permissions and format")
        sys.exit(1)
    
    process_adt_report(report_path) 

    