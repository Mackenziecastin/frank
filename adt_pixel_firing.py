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
    1. Remove Health from Ln_of_Busn
    2. Remove 'US: Health' from DNIS_BUSN_SEG_CD
    3. Filter for most recent date in Sale_Date column
    4. Filter for WEB0021011 in Lead_DNIS
    5. Filter for New/Resale in Ordr_Type
    """
    try:
        # Convert Sale_Date to datetime if it's not already
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
        
        # Get the most recent date from the Sale_Date column
        most_recent_date = df['Sale_Date'].max()
        logging.info(f"Most recent sale date in data: {most_recent_date.strftime('%Y-%m-%d')}")
        
        # Apply all filters
        filtered_df = df[
            # Exclude Health from Ln_of_Busn
            (df['Ln_of_Busn'] != 'Health') &
            # Exclude US: Health from DNIS_BUSN_SEG_CD
            (df['DNIS_BUSN_SEG_CD'] != 'US: Health') &
            # Filter for most recent date
            (df['Sale_Date'].dt.date == most_recent_date.date()) &
            # Filter for specific Lead_DNIS
            (df['Lead_DNIS'] == 'WEB0021011') &
            # Filter for New and Resale order types (case insensitive)
            (df['Ordr_Type'].str.upper().isin(['NEW', 'RESALE']))
        ]
        
        logging.info(f"Data cleaned successfully. Found {len(filtered_df)} qualifying sales for {most_recent_date.strftime('%Y-%m-%d')}.")
        return filtered_df
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

def fire_pixel(transaction_id):
    """
    Fire the pixel for a given transaction ID
    """
    pixel_url = "https://speedtrkzone.com/m.ashx"
    params = {
        'o': '32022',
        'e': '565',
        'f': 'pb',
        't': transaction_id,
        'pubid': '42865',
        'campid': '91149'
    }
    
    try:
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        logging.info(f"Pixel fired successfully for transaction {transaction_id}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error firing pixel for transaction {transaction_id}: {str(e)}")
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
        print(f"\nFound {total_sales} qualifying sales for date: {filtered_df['Sale_Date'].iloc[0].strftime('%Y-%m-%d')}")
        logging.info(f"Found {total_sales} sales to process")
        
        if total_sales == 0:
            print("\nNo qualifying sales found to process. Check the log file for details.")
            return
        
        # Fire pixel for each sale
        print("\nFiring pixels...")
        successful_fires = 0
        for idx, row in filtered_df.iterrows():
            # Generate a unique transaction ID using timestamp and UUID
            transaction_id = f"ADT_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"
            
            print(f"  Firing pixel {successful_fires + 1} of {total_sales}...", end="")
            if fire_pixel(transaction_id):
                successful_fires += 1
                print(" ✓")
            else:
                print(" ✗")
        
        # Log summary
        summary = f"\nProcessing complete!\n" \
                 f"Date processed: {filtered_df['Sale_Date'].iloc[0].strftime('%Y-%m-%d')}\n" \
                 f"Successfully fired: {successful_fires} out of {total_sales} pixels\n" \
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