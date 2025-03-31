import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import uuid

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
    3. Filter for all records from yesterday in the Sale_Date column
    4. Filter for WEB0021011 in Lead_DNIS
    5. Filter for order types containing 'New' or 'Resale'
    """
    try:
        # Convert Sale_Date to datetime if it's not already and remove any null values
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], errors='coerce')
        df = df.dropna(subset=['Sale_Date'])
        
        # Get all unique dates and sort them to find yesterday
        all_dates = sorted(df['Sale_Date'].dt.date.unique())
        if len(all_dates) >= 2:
            yesterday = all_dates[-2]  # Second to last date is yesterday
            logging.info(f"Found yesterday's date in data: {yesterday.strftime('%Y-%m-%d')}")
        else:
            raise ValueError("Not enough dates in the data to determine yesterday")
        
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
    
    # Convert sale date to Unix timestamp for yesterday at noon (to ensure it's during business hours)
    sale_datetime = pd.to_datetime(sale_date).replace(hour=12, minute=0, second=0)
    timestamp = int(sale_datetime.timestamp())
    
    # Debug: Print timestamp information
    print(f"\nDebug Timestamp Info:")
    print(f"  Sale Date: {sale_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Unix Timestamp: {timestamp}")
    
    params = {
        'o': '32022',
        'e': '565',
        'f': 'pb',
        't': transaction_id,
        'pubid': '42865',
        'campid': campaign_id,
        'timestamp': timestamp
    }
    
    # Construct and print the complete URL
    url_with_params = requests.Request('GET', pixel_url, params=params).prepare().url
    print(f"  Complete URL: {url_with_params}")
    
    try:
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        logging.info(f"Pixel fired successfully for transaction {transaction_id} (Install Method: {install_method}, Campaign: {campaign_id}, Timestamp: {sale_datetime})")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error firing pixel for transaction {transaction_id} (Install Method: {install_method}, Timestamp: {sale_datetime}): {str(e)}")
        return False

def process_adt_report(file_path):
    """
    Main function to process the ADT report and fire pixels
    """
    try:
        # Read the CSV file
        print("\n=== ADT Pixel Firing Process Started ===")
        print(f"Reading file: {file_path}")
        logging.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        
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
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python adt_pixel_firing.py <path_to_report>")
        sys.exit(1)
    
    report_path = sys.argv[1]
    process_adt_report(report_path) 

    