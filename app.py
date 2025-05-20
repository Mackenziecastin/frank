import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from datetime import datetime, timedelta
import logging
import sys
import os
import uuid
import requests
import io
import tempfile
from adt_pixel_firing import process_adt_report

st.set_page_config(page_title="Partner Optimization Report Generator", layout="wide")

def show_main_page():
    st.title("Partner Optimization Report Generator")

    st.write("""
    This tool processes your marketing data files and generates a comprehensive optimization report.
    Please upload the required files below.
    """)

    col1, col2 = st.columns(2)

    with col1:
        affiliate_file = st.file_uploader("Upload Affiliate Leads QA File (CSV)", type=['csv'])
    
    with col2:
        advanced_file = st.file_uploader("Upload Advanced Action Sheet (CSV)", type=['csv'])

    if affiliate_file and advanced_file:
        try:
            # Read files
            affiliate_df = pd.read_csv(affiliate_file)
            advanced_df = pd.read_csv(advanced_file)
            
            # Read partner list automatically
            try:
                partner_list_df = pd.read_csv('Full DA Performance Marketing Team Partner List - Sheet1.csv')
            except Exception as e:
                st.warning(f"Could not read partner list file: {str(e)}. VLOOKUP functionality will be disabled.")
                partner_list_df = None
            
            # Process both dataframes
            affiliate_df_processed = process_dataframe(affiliate_df, 'Click URL')
            if affiliate_df_processed is None:
                st.error("Failed to process Affiliate file. Please check if it contains a 'Click URL' column.")
                st.stop()
                
            advanced_df_processed = process_dataframe(advanced_df, 'Landing Page URL')
            if advanced_df_processed is None:
                st.error("Failed to process Advanced Action file. Please check if it contains a 'Landing Page URL' column.")
                st.stop()
            
            # Show preview of processed data
            st.subheader("Preview of Processed Affiliate Data")
            st.dataframe(affiliate_df_processed[['Click URL', 'PID', 'SUBID', 'partnerID']].head())
            
            st.subheader("Preview of Processed Advanced Action Data")
            st.dataframe(advanced_df_processed[['Landing Page URL', 'PID', 'SUBID', 'partnerID']].head())
            
            # Create maturation-adjusted dataframes
            if 'Created Date' in affiliate_df_processed.columns and 'Action Date' in advanced_df_processed.columns:
                # Convert dates to datetime
                affiliate_df_processed['Created Date'] = pd.to_datetime(affiliate_df_processed['Created Date'])
                advanced_df_processed['Action Date'] = pd.to_datetime(advanced_df_processed['Action Date'])
                
                # Get the date range from Advanced Action report
                full_end_date = advanced_df_processed['Action Date'].max()
                full_start_date = advanced_df_processed['Action Date'].min()
                
                # For matured report: exclude the last 7 days but keep same start date
                matured_end_date = full_end_date - pd.Timedelta(days=7)
                matured_start_date = full_start_date  # Same as full report start date
                
                # Debug information before filtering
                st.write("Debug - Before filtering:")
                st.write(f"Total records in affiliate data: {len(affiliate_df_processed)}")
                st.write(f"Total Transaction Count: {affiliate_df_processed['Transaction Count'].sum()}")
                
                # Create full report dataframes with date filtering
                # Filter both datasets to match exactly
                affiliate_df_full = affiliate_df_processed[
                    (affiliate_df_processed['Created Date'].dt.date >= full_start_date.date()) &
                    (affiliate_df_processed['Created Date'].dt.date <= full_end_date.date())
                ]
                advanced_df_full = advanced_df_processed[
                    (advanced_df_processed['Action Date'].dt.date >= full_start_date.date()) &
                    (advanced_df_processed['Action Date'].dt.date <= full_end_date.date())
                ]
                
                # Debug information after full report filtering
                st.write("\nDebug - After full report filtering:")
                st.write(f"Records in affiliate data: {len(affiliate_df_full)}")
                st.write(f"Transaction Count: {affiliate_df_full['Transaction Count'].sum()}")
                st.write(f"Date range: {affiliate_df_full['Created Date'].min()} to {affiliate_df_full['Created Date'].max()}")
                
                # Create matured report dataframes with date filtering
                affiliate_df_matured = affiliate_df_processed[
                    (affiliate_df_processed['Created Date'].dt.date >= matured_start_date.date()) &
                    (affiliate_df_processed['Created Date'].dt.date <= matured_end_date.date())
                ]
                advanced_df_matured = advanced_df_processed[
                    (advanced_df_processed['Action Date'].dt.date >= matured_start_date.date()) &
                    (advanced_df_processed['Action Date'].dt.date <= matured_end_date.date())
                ]
                
                # Debug information after matured report filtering
                st.write("\nDebug - After matured report filtering:")
                st.write(f"Records in affiliate data: {len(affiliate_df_matured)}")
                st.write(f"Transaction Count: {affiliate_df_matured['Transaction Count'].sum()}")
                st.write(f"Date range: {affiliate_df_matured['Created Date'].min()} to {affiliate_df_matured['Created Date'].max()}")
                
                # Show date ranges for both reports
                st.subheader("Date Ranges")
                st.write("Full Report Dates:")
                st.write(f"- Start Date: {full_start_date.strftime('%Y-%m-%d')} (Based on Advanced Action report)")
                st.write(f"- End Date: {full_end_date.strftime('%Y-%m-%d')} (Based on Advanced Action report)")
                
                st.write("\nMatured Report Dates (excluding last 7 days):")
                st.write(f"- Start Date: {matured_start_date.strftime('%Y-%m-%d')}")
                st.write(f"- End Date: {matured_end_date.strftime('%Y-%m-%d')}")
                
                # Add date range validation and warning
                if affiliate_df_processed['Created Date'].max() > full_end_date:
                    extra_days = (affiliate_df_processed['Created Date'].max() - full_end_date).days
                    st.warning(f"Note: Affiliate data contains {extra_days} additional day(s) beyond {full_end_date.strftime('%Y-%m-%d')}. These dates have been excluded for consistency with Advanced Action data.")
                
                # Create pivot tables and reports
                # Full report
                affiliate_pivot_full = create_affiliate_pivot(affiliate_df_full)
                advanced_pivot_full = create_advanced_pivot(advanced_df_full)
                optimization_report_full = create_optimization_report(
                    affiliate_pivot_full, advanced_pivot_full, partner_list_df
                )
                
                # Matured report
                affiliate_pivot_matured = create_affiliate_pivot(affiliate_df_matured)
                advanced_pivot_matured = create_advanced_pivot(advanced_df_matured)
                optimization_report_matured = create_optimization_report(
                    affiliate_pivot_matured, advanced_pivot_matured, partner_list_df
                )
                
                # Show preview of both reports
                st.subheader("Preview of Full Optimization Report")
                st.dataframe(optimization_report_full)
                
                st.subheader("Preview of Matured Optimization Report (Excluding Last 7 Days)")
                st.dataframe(optimization_report_matured)
                
                # Create download buttons for both reports
                col1, col2 = st.columns(2)
                
                with col1:
                    excel_data = to_excel_download(
                        affiliate_df_full, advanced_df_full, optimization_report_full
                    )
                    st.download_button(
                        label="Download Full Report",
                        data=excel_data,
                        file_name=f"partner_optimization_report_full_{full_end_date.strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    excel_data_matured = to_excel_download(
                        affiliate_df_matured, advanced_df_matured, optimization_report_matured
                    )
                    st.download_button(
                        label="Download Matured Report",
                        data=excel_data_matured,
                        file_name=f"partner_optimization_report_matured_{matured_end_date.strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                missing_columns = []
                if 'Created Date' not in affiliate_df_processed.columns:
                    missing_columns.append("'Created Date' in Affiliate Leads file")
                if 'Action Date' not in advanced_df_processed.columns:
                    missing_columns.append("'Action Date' in Advanced Action file")
                    
                st.error(f"Required date columns not found: {', '.join(missing_columns)}. Available columns are:")
                st.write("Affiliate Leads columns:", ", ".join(affiliate_df_processed.columns))
                st.write("Advanced Action columns:", ", ".join(advanced_df_processed.columns))
                
                # Show only the full report if date columns are missing
                st.subheader("Preview of Full Optimization Report (without date filtering)")
                st.dataframe(optimization_report_full)
                
                excel_data = to_excel_download(
                    affiliate_df_full, advanced_df_full, optimization_report_full
                )
                st.download_button(
                    label="Download Full Report",
                    data=excel_data,
                    file_name="partner_optimization_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
        except Exception as e:
            st.error(f"An error occurred while processing the files: {str(e)}")
            st.error("Please ensure your files contain all required columns and are in the correct format.")
            # Add more detailed error information
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Please upload both required files to generate the report.")

def extract_values_after_3d(url):
    """Extract all values after %3D in the URL."""
    try:
        if pd.isna(url):
            return ""
        
        # Find the part after %3D
        match = re.search(r'%3D(.*?)(?:$|&)', url)
        if match:
            return match.group(1)
        return ""
    except:
        return ""

def extract_pid_subid(after_3d_value):
    """Extract PID and SUBID from the string after %3D."""
    try:
        if not after_3d_value:
            return "", ""
        
        # Split by underscore
        parts = after_3d_value.split('_')
        
        # First part is PID
        pid = parts[0] if parts and parts[0].isdigit() else ""
        
        # Second part is SUBID (if it exists and contains only digits)
        subid = parts[1] if len(parts) > 1 and parts[1].isdigit() else ""
        
        return pid, subid
    except:
        return "", ""

def process_dataframe(df, url_column):
    """Process dataframe to add PID, SUBID, and partnerID columns."""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Check for variations of URL column names
    url_column_variations = {
        'Click URL': ['Click URL', 'ClickURL', 'Click_URL', 'click url', 'click_url'],
        'Landing Page URL': ['Landing Page URL', 'LandingPageURL', 'Landing_Page_URL', 'landing page url', 'landing_page_url', 'URL']
    }
    
    # Find the actual column name in the dataframe
    actual_column = None
    expected_type = 'Click URL' if url_column == 'Click URL' else 'Landing Page URL'
    
    for col in df.columns:
        if col in url_column_variations[expected_type]:
            actual_column = col
            break
    
    if actual_column is None:
        available_columns = ", ".join(df.columns)
        st.error(f"Could not find {expected_type} column. Available columns are: {available_columns}")
        return None
    
    # Filter out rows where URL contains "coolsculpting"
    coolsculpting_count = df[df[actual_column].str.contains('coolsculpting', case=False, na=False)].shape[0]
    if coolsculpting_count > 0:
        st.info(f"Filtered out {coolsculpting_count} rows containing 'coolsculpting' in the URL.")
    
    df = df[~df[actual_column].str.contains('coolsculpting', case=False, na=False)]
    
    # Create new columns
    df['After_3D'] = df[actual_column].apply(extract_values_after_3d)
    df['PID'] = ""
    df['SUBID'] = ""
    df['partnerID'] = ""
    
    # Process each row
    for idx, row in df.iterrows():
        pid, subid = extract_pid_subid(row['After_3D'])
        df.at[idx, 'PID'] = pid
        df.at[idx, 'SUBID'] = subid
        
        # Create partnerID
        if pid:
            if subid:
                df.at[idx, 'partnerID'] = f"{pid}_{subid}"
            else:
                df.at[idx, 'partnerID'] = f"{pid}_"
        else:
            df.at[idx, 'partnerID'] = "Unattributed"
    
    # Replace any partnerID that is just "_" with "Unattributed"
    df.loc[df['partnerID'] == "_", 'partnerID'] = "Unattributed"
    
    # Drop the temporary column
    df = df.drop('After_3D', axis=1)
    
    return df

def create_affiliate_pivot(df):
    """Create pivot table for Affiliate Leads QA data.
    
    As per instructions:
    1. Pull in the ClickURL_partnerID into the rows (index='partnerID')
    2. In the values, pull in Sum of Booked Count, Sum of Transaction Count & Sum of Net Sales Amount
    """
    # First verify Transaction Count column exists
    if 'Transaction Count' not in df.columns:
        st.error("Transaction Count column not found in affiliate data")
        return None
        
    # Convert Transaction Count to numeric, treating any non-numeric values as 0
    df['Transaction Count'] = pd.to_numeric(df['Transaction Count'], errors='coerce').fillna(0)
    
    # Ensure other numeric columns are properly converted if they exist
    numeric_cols = ['Booked Count', 'Net Sales Amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Log data for debugging
    st.write(f"Debug - Total Transaction Count before pivot: {df['Transaction Count'].sum()}")
    st.write(f"Debug - Date range in data: {df['Created Date'].min()} to {df['Created Date'].max()}")
    
    # Create pivot table with specific aggregation methods
    pivot = df.groupby('partnerID').agg({
        'Transaction Count': 'sum',
        'Booked Count': 'sum',
        'Net Sales Amount': 'sum'
    }).reset_index()
    
    # Log pivot results for debugging
    st.write(f"Debug - Total Transaction Count after pivot: {pivot['Transaction Count'].sum()}")
    
    return pivot

def create_advanced_pivot(df):
    """Create pivot table for Advanced Action data.
    
    As per instructions:
    1. Pull in the Landing Page URL_PartnerID into the rows (index='partnerID')
    2. In the values, pull in Count of Event Type but filter for ONLY the Lead Submissions
    3. Also pull in the Sum of Action Earnings
    """
    # Ensure numeric columns are properly converted
    df['Action Id'] = pd.to_numeric(df['Action Id'], errors='coerce').fillna(0)
    df['Action Earnings'] = pd.to_numeric(df['Action Earnings'], errors='coerce').fillna(0)
    
    # Filter for Lead Submissions
    lead_submissions = df[df['Event Type'] == 'Lead Submission']
    
    # Count the number of rows with Lead Submission per partnerID
    lead_counts = lead_submissions.groupby('partnerID').size().reset_index(name='Leads')
    
    # Sum the Action Earnings per partnerID
    earnings_sums = lead_submissions.groupby('partnerID')['Action Earnings'].sum().reset_index()
    
    # Merge the two dataframes
    pivot = pd.merge(lead_counts, earnings_sums, on='partnerID')
    
    # Rename columns for clarity
    pivot.columns = ['partnerID', 'Leads', 'Spend']
    
    return pivot

def create_optimization_report(affiliate_pivot, advanced_pivot, partner_list=None):
    """Create the final optimization report by combining pivot tables.
    
    As per instructions, columns should be:
    1. PartnerID = landing page URL_partnerID and clickURL_partnerID values
    2. Leads = count of event type from Cleaned_Advance_Action pivot table
    3. Spend = sum of action earnings from Cleaned_Advance_Action pivot table 
    4. Bookings = sum of booked count from Cleaned_Affliate_Leads_QA pivot table
    5. Sales = sum of transaction count from Cleaned_Affliate_Leads_QA pivot table
    6. Revenue = sum of net sales amount from Cleaned_Affliate_Leads_QA pivot table
    """
    # First, rename the affiliate pivot columns for clarity
    renamed_affiliate = affiliate_pivot.copy()
    if 'Transaction Count' in renamed_affiliate.columns:
        renamed_affiliate = renamed_affiliate.rename(columns={
            'Booked Count': 'Bookings',
            'Transaction Count': 'Sales',
            'Net Sales Amount': 'Revenue'
        })
    else:
        st.warning("Expected columns not found in affiliate data. Using default column names.")
        renamed_affiliate = renamed_affiliate.rename(columns={
            renamed_affiliate.columns[1]: 'Bookings',
            renamed_affiliate.columns[2]: 'Sales',
            renamed_affiliate.columns[3]: 'Revenue'
        })
    
    # Merge the pivot tables
    merged_df = pd.merge(
        advanced_pivot,
        renamed_affiliate,
        on='partnerID',
        how='outer'
    ).fillna(0)
    
    # Ensure all numeric columns are properly converted
    for col in ['Leads', 'Spend', 'Bookings', 'Sales', 'Revenue']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
    
    # Remove rows with all zeros
    merged_df = merged_df[~((merged_df['Leads'] == 0) & 
                           (merged_df['Spend'] == 0) & 
                           (merged_df['Bookings'] == 0) & 
                           (merged_df['Sales'] == 0) & 
                           (merged_df['Revenue'] == 0))]
    
    # Calculate additional metrics
    merged_df['Lead to Sale'] = merged_df['Sales'] / merged_df['Leads'].replace(0, np.inf)
    merged_df['ROAS'] = merged_df['Revenue'] / merged_df['Spend'].replace(0, np.inf)
    merged_df['eCPL at $1.50'] = (merged_df['Revenue'] / merged_df['Leads'].replace(0, np.inf)) / 1.5
    
    # Clean up infinity values
    merged_df = merged_df.replace([np.inf, -np.inf], 0)
    
    # Add VLOOKUP data if partner list is provided
    if partner_list is not None:
        try:
            # Extract affiliate ID from partnerID (part before underscore)
            merged_df['Affiliate ID'] = merged_df['partnerID'].apply(
                lambda x: x.split('_')[0] if x != "Unattributed" and '_' in x else x)
            
            # Ensure required columns exist in partner list
            required_cols = ['Affiliate ID', 'Affiliate Name', 'Account Manager Name']
            if not all(col in partner_list.columns for col in required_cols):
                st.warning("Partner list file missing required columns. Required columns are: Affiliate ID, Affiliate Name, Account Manager Name")
            else:
                # Convert Affiliate ID to string in both dataframes
                partner_list['Affiliate ID'] = partner_list['Affiliate ID'].astype(str)
                merged_df['Affiliate ID'] = merged_df['Affiliate ID'].astype(str)
                
                # Merge with partner list to get affiliate name and account manager
                merged_df = pd.merge(
                    merged_df,
                    partner_list[['Affiliate ID', 'Affiliate Name', 'Account Manager Name']],
                    on='Affiliate ID',
                    how='left'
                )
                
                # Fill NaN values with empty strings
                merged_df['Affiliate Name'] = merged_df['Affiliate Name'].fillna("")
                merged_df['Account Manager Name'] = merged_df['Account Manager Name'].fillna("")
                
                # Reorder columns to put VLOOKUP data first
                cols = ['partnerID', 'Affiliate Name', 'Account Manager Name'] + \
                    [col for col in merged_df.columns if col not in 
                        ['partnerID', 'Affiliate Name', 'Account Manager Name', 'Affiliate ID']]
                merged_df = merged_df[cols]
                
                # Drop the temporary Affiliate ID column
                merged_df = merged_df.drop('Affiliate ID', axis=1)
                
        except Exception as e:
            st.warning(f"Error in VLOOKUP processing: {str(e)}. Continuing without VLOOKUP data.")
    
    return merged_df

def to_excel_download(df_affiliate, df_advanced, df_optimization):
    """Convert dataframes to Excel file for download."""
    output = BytesIO()
    
    # Use xlsxwriter engine instead of openpyxl for formatting support
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write each dataframe to a different sheet
        df_affiliate.to_excel(writer, sheet_name='Cleaned Affiliate Data', index=False)
        df_advanced.to_excel(writer, sheet_name='Cleaned Advanced Action Data', index=False)
        df_optimization.to_excel(writer, sheet_name='Optimization Report', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Optimization Report']
        
        # Define formats
        money_format = workbook.add_format({'num_format': '$#,##0.00'})
        integer_format = workbook.add_format({'num_format': '0'})
        percent_format = workbook.add_format({'num_format': '0.0%'})
        
        # Apply formats to specific columns
        for col_idx, col_name in enumerate(df_optimization.columns):
            if col_name in ['Spend', 'Revenue', 'ROAS', 'eCPL at $1.50']:
                worksheet.set_column(col_idx, col_idx, 15, money_format)
            elif col_name in ['Leads', 'Bookings', 'Sales']:
                worksheet.set_column(col_idx, col_idx, 15, integer_format)
            elif col_name in ['Lead to Sale']:
                worksheet.set_column(col_idx, col_idx, 15, percent_format)
            else:
                worksheet.set_column(col_idx, col_idx, 15)  # Default width
    
    return output.getvalue()

def show_adt_pixel():
    st.title("ADT Pixel Firing")
    
    st.write("""
    This tool processes ADT Athena reports and fires pixels for qualifying sales.
    Upload your ADT Athena report (CSV format) to begin.
    """)
    
    uploaded_file = st.file_uploader("Upload ADT Athena Report (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Process and Fire Pixels"):
            try:
                # Save uploaded file temporarily with original filename
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Set up logging
                log_stream = setup_logging()
                
                # Process the report
                process_adt_report(temp_path)
                
                # Clean up temporary file and directory
                os.unlink(temp_path)
                os.rmdir(temp_dir)
                
                # Show success message
                st.success("Pixels fired successfully!")
                
                # Show logs
                with st.expander("View Processing Logs"):
                    st.text(log_stream.getvalue())
                    
            except Exception as e:
                st.error(f"Error processing ADT report: {str(e)}")
                with st.expander("View Error Logs"):
                    st.text(log_stream.getvalue())

def setup_logging():
    """Set up logging to capture output"""
    log_stream = io.StringIO()
    logging.basicConfig(
        stream=log_stream,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return log_stream

# At the bottom of the file, modify the main execution logic
def main():
    # Create the navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Frank (LaserAway)", "Bob (ADT)", "ADT Pixel Firing", "Brinks Optimization Report"])
    
    if page == "Frank (LaserAway)":
        show_main_page()
    elif page == "Bob (ADT)":
        # Import and show Bob's analysis page
        from pages.bob_analysis import show_bob_analysis
        show_bob_analysis()
    elif page == "ADT Pixel Firing":
        show_adt_pixel()
    elif page == "Brinks Optimization Report":
        # Import and show Brinks optimization page - use standard import
        from pages.brinks_optimization import show_brinks_optimization
        show_brinks_optimization()

if __name__ == "__main__":
    main() 