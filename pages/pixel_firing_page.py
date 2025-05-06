import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import logging
import io
import uuid
import re
import subprocess

# Set page config first
st.set_page_config(page_title="ADT Pixel Firing", layout="wide")

# Function to install a package
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

# Try importing pandas, install if not available
try:
    import pandas as pd
    st.sidebar.success("✓ Successfully imported pandas")
except ImportError:
    st.warning("Pandas not found. Attempting to install...")
    if install_package("pandas==2.2.0"):
        try:
            import pandas as pd
            st.sidebar.success("✓ Successfully installed and imported pandas")
        except ImportError as e:
            st.error(f"Failed to import pandas even after installation: {str(e)}")
            st.stop()
    else:
        st.error("Failed to install pandas. Please contact support.")
        st.stop()

# Try importing requests, install if not available
try:
    import requests
    st.sidebar.success("✓ Successfully imported requests")
except ImportError:
    st.warning("Requests not found. Attempting to install...")
    if install_package("requests==2.31.0"):
        try:
            import requests
            st.sidebar.success("✓ Successfully installed and imported requests")
        except ImportError as e:
            st.error(f"Failed to import requests even after installation: {str(e)}")
            st.stop()
    else:
        st.error("Failed to install requests. Please contact support.")
        st.stop()

# Display Python environment information
st.sidebar.write("Environment Information:")
st.sidebar.code(f"""
Python Version: {sys.version}
Working Directory: {os.getcwd()}
Python Path: {sys.path}
Pandas Version: {pd.__version__ if 'pd' in locals() else 'Not installed'}
""")

def setup_logging():
    """Set up logging to capture output"""
    log_stream = io.StringIO()
    logging.basicConfig(
        stream=log_stream,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return log_stream

def clean_data(df, report_filename):
    """
    Clean and filter the data according to requirements
    """
    try:
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
        st.write(f"Starting with {total_records} total records")
        
        # Apply filters one by one and show counts
        # Remove health leads from Ln_of_Busn
        health_business_filter = ~df['Ln_of_Busn'].str.contains('Health', case=False, na=False)
        df_after_health_business = df[health_business_filter]
        st.write(f"After excluding Health from Ln_of_Busn: {len(df_after_health_business)} records")
        
        # Remove US: Health from DNIS_BUSN_SEG_CD
        health_dnis_filter = ~df_after_health_business['DNIS_BUSN_SEG_CD'].str.contains('US: Health', case=False, na=False)
        df_after_health_dnis = df_after_health_business[health_dnis_filter]
        st.write(f"After excluding US: Health from DNIS_BUSN_SEG_CD: {len(df_after_health_dnis)} records")
        
        # Filter for yesterday's date based on Sale_Date
        date_filter = (df_after_health_dnis['Sale_Date'].dt.date == yesterday)
        df_after_date = df_after_health_dnis[date_filter]
        st.write(f"After filtering for yesterday ({yesterday.strftime('%Y-%m-%d')}): {len(df_after_date)} records")
        
        dnis_filter = (df_after_date['Lead_DNIS'] == 'WEB0021011')
        df_after_lead_dnis = df_after_date[dnis_filter]
        st.write(f"After filtering for Lead_DNIS 'WEB0021011': {len(df_after_lead_dnis)} records")
        
        # Filter for New/Resale order types
        order_type_filter = (
            df_after_lead_dnis['Ordr_Type'].str.contains('New', case=False, na=False) |
            df_after_lead_dnis['Ordr_Type'].str.contains('Resale', case=False, na=False)
        )
        filtered_df = df_after_lead_dnis[order_type_filter]
        st.write(f"After filtering for New/Resale order types: {len(filtered_df)} records")
        
        # Separate DIFM and DIY records
        difm_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIFM', case=False, na=False)]
        diy_records = filtered_df[filtered_df['INSTALL_METHOD'].str.contains('DIY', case=False, na=False)]
        
        # Count DIFM and DIY records
        difm_count = len(difm_records)
        diy_count = len(diy_records)
        
        # Combine all records
        filtered_df = pd.concat([difm_records, diy_records])
        
        st.write("Final counts:")
        st.write(f"DIFM Sales: {difm_count}")
        st.write(f"DIY Sales: {diy_count}")
        
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
    campaign_id = "91149" if "DIFM" in str(install_method).upper() else "91162"
    
    # Set the time to noon on the sale date
    pixel_datetime = sale_date.replace(hour=12, minute=0, second=0)
    iso_datetime = pixel_datetime.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    
    params = {
        'o': '32022',
        'e': '565',
        'f': 'pb',
        't': transaction_id,
        'pubid': '42865',
        'campid': campaign_id,
        'dt': iso_datetime
    }
    
    try:
        response = requests.get(pixel_url, params=params)
        response.raise_for_status()
        logging.info(f"Pixel fired successfully for {transaction_id}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error firing pixel for {transaction_id}: {str(e)}")
        return False

def process_adt_report(uploaded_file):
    """
    Process the ADT report and fire pixels
    """
    try:
        # Set up logging
        log_stream = setup_logging()
        
        st.write("=== ADT Pixel Firing Process Started ===")
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                st.write(f"Trying to read file with {encoding} encoding...")
                df = pd.read_csv(uploaded_file, encoding=encoding)
                st.write(f"Successfully read file with {encoding} encoding!")
                break
            except UnicodeDecodeError:
                st.write(f"Failed to read with {encoding} encoding, trying next...")
                continue
        
        if df is None:
            raise Exception("Could not read the file with any of the supported encodings")
        
        # Clean and filter the data
        filtered_df = clean_data(df, uploaded_file.name)
        
        # Count of sales to process
        total_sales = len(filtered_df)
        if total_sales == 0:
            st.warning("No qualifying sales found to process.")
            return
        
        # Fire pixel for each sale
        st.write("Firing pixels...")
        successful_fires = {'DIFM': 0, 'DIY': 0}
        
        for idx, row in filtered_df.iterrows():
            # Generate transaction ID
            sale_date_str = row['Sale_Date'].strftime('%Y%m%d')
            transaction_id = f"ADT_{sale_date_str}_{str(uuid.uuid4())[:8]}"
            install_method = row['INSTALL_METHOD']
            
            # Determine category and fire pixel
            category = 'DIFM' if 'DIFM' in str(install_method).upper() else 'DIY'
            
            if fire_pixel(transaction_id, install_method, row['Sale_Date']):
                successful_fires[category] += 1
        
        # Show summary
        st.success("Processing complete!")
        st.write(f"Successfully fired DIFM pixels: {successful_fires['DIFM']}")
        st.write(f"Successfully fired DIY pixels: {successful_fires['DIY']}")
        st.write(f"Total pixels fired: {sum(successful_fires.values())} out of {total_sales}")
        
        # Show logs
        with st.expander("View Processing Logs"):
            st.text(log_stream.getvalue())
        
    except Exception as e:
        st.error(f"Error processing ADT report: {str(e)}")
        st.error("Check the logs below for more details")
        with st.expander("View Error Logs"):
            st.text(log_stream.getvalue())

def main():
    st.title("ADT Pixel Firing")
    
    st.write("""
    This tool processes ADT Athena reports and fires pixels for qualifying sales.
    Upload your ADT Athena report (CSV format) to begin.
    """)
    
    uploaded_file = st.file_uploader("Upload ADT Athena Report (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Process and Fire Pixels"):
            process_adt_report(uploaded_file)

if __name__ == "__main__":
    main() 