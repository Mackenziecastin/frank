import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

st.set_page_config(page_title="Partner Optimization Report Generator", layout="wide")

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
    
    # Ensure URL column exists
    if url_column not in df.columns:
        st.error(f"Column '{url_column}' not found in the uploaded file. Please check your file format.")
        return None
    
    # Create new columns
    df['After_3D'] = df[url_column].apply(extract_values_after_3d)
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
    """Create pivot table for Affiliate Leads QA data."""
    # Ensure numeric columns are properly converted
    numeric_cols = ['Booked Count', 'Transaction Count', 'Net Sales Amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    pivot = pd.pivot_table(
        df,
        index='partnerID',
        values=numeric_cols,
        aggfunc='sum'
    ).reset_index()
    
    return pivot

def create_advanced_pivot(df):
    """Create pivot table for Advanced Action data."""
    # Ensure numeric columns are properly converted
    df['Action Id'] = pd.to_numeric(df['Action Id'], errors='coerce').fillna(0)
    df['Action Earnings'] = pd.to_numeric(df['Action Earnings'], errors='coerce').fillna(0)
    
    # Filter for Lead Submissions
    lead_submissions = df[df['Event Type'] == 'Lead Submission']
    
    pivot = pd.pivot_table(
        lead_submissions,
        index='partnerID',
        values=['Action Id', 'Action Earnings'],
        aggfunc={'Action Id': 'count', 'Action Earnings': 'sum'}
    ).reset_index()
    
    # Rename columns for clarity
    pivot.columns = ['partnerID', 'Leads', 'Spend']
    
    return pivot

def create_optimization_report(affiliate_pivot, advanced_pivot, partner_list=None):
    """Create the final optimization report by combining pivot tables."""
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
                lambda x: x.split('_')[0] if x != "Unattributed" else "")
            
            # Ensure Affiliate ID column exists in partner list
            if 'Affiliate ID' not in partner_list.columns:
                st.warning("Partner list file does not contain 'Affiliate ID' column. VLOOKUP functionality disabled.")
            elif 'Affiliate Name' not in partner_list.columns or 'Account Manager' not in partner_list.columns:
                st.warning("Partner list file missing required columns. VLOOKUP functionality limited.")
            else:
                # Convert Affiliate ID to string in both dataframes
                partner_list['Affiliate ID'] = partner_list['Affiliate ID'].astype(str)
                
                # Create a dictionary for faster lookups
                affiliate_dict = dict(zip(partner_list['Affiliate ID'], 
                                        partner_list['Affiliate Name']))
                manager_dict = dict(zip(partner_list['Affiliate ID'], 
                                        partner_list['Account Manager']))
                
                # Apply the lookups
                merged_df['Affiliate Name'] = merged_df['Affiliate ID'].map(affiliate_dict).fillna("")
                merged_df['Account Manager'] = merged_df['Affiliate ID'].map(manager_dict).fillna("")
                
                # Reorder columns to put VLOOKUP data first
                cols = ['Affiliate ID', 'Affiliate Name', 'Account Manager', 'partnerID'] + \
                    [col for col in merged_df.columns if col not in 
                        ['Affiliate ID', 'Affiliate Name', 'Account Manager', 'partnerID']]
                merged_df = merged_df[cols]
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

# Main app
st.title("Partner Optimization Report Generator")

st.write("""
This tool processes your marketing data files and generates a comprehensive optimization report.
Please upload the required files below.
""")

col1, col2, col3 = st.columns(3)

with col1:
    affiliate_file = st.file_uploader("Upload Affiliate Leads QA File (CSV)", type=['csv'])
    
with col2:
    advanced_file = st.file_uploader("Upload Advanced Action Sheet (CSV)", type=['csv'])

with col3:
    partner_list_file = st.file_uploader("Upload Partner List (CSV, Optional)", type=['csv'])

if affiliate_file and advanced_file:
    try:
        # Read files
        affiliate_df = pd.read_csv(affiliate_file)
        advanced_df = pd.read_csv(advanced_file)
        
        # Display column names for debugging
        st.write("Affiliate file columns:", affiliate_df.columns.tolist())
        st.write("Advanced file columns:", advanced_df.columns.tolist())
        
        # Process both dataframes
        affiliate_df_processed = process_dataframe(affiliate_df, 'Click URL')
        if affiliate_df_processed is None:
            st.error("Failed to process Affiliate file. Please check if it contains a 'Click URL' column.")
            st.stop()
            
        advanced_df_processed = process_dataframe(advanced_df, 'Landing Page URL')
        if advanced_df_processed is None:
            st.error("Failed to process Advanced Action file. Please check if it contains a 'Landing Page URL' column.")
            st.stop()
        
        # Read partner list if provided
        partner_list_df = None
        if partner_list_file:
            partner_list_df = pd.read_csv(partner_list_file)
            st.write("Partner list columns:", partner_list_df.columns.tolist())
        
        # Show preview of processed data
        st.subheader("Preview of Processed Affiliate Data")
        st.dataframe(affiliate_df_processed[['Click URL', 'PID', 'SUBID', 'partnerID']].head())
        
        st.subheader("Preview of Processed Advanced Action Data")
        st.dataframe(advanced_df_processed[['Landing Page URL', 'PID', 'SUBID', 'partnerID']].head())
        
        # Create pivot tables
        affiliate_pivot = create_affiliate_pivot(affiliate_df_processed)
        st.subheader("Preview of Affiliate Pivot")
        st.dataframe(affiliate_pivot.head())
        
        advanced_pivot = create_advanced_pivot(advanced_df_processed)
        st.subheader("Preview of Advanced Action Pivot")
        st.dataframe(advanced_pivot.head())
        
        # Create optimization report
        optimization_report = create_optimization_report(affiliate_pivot, advanced_pivot, partner_list_df)
        
        # Show preview of results
        st.subheader("Preview of Optimization Report")
        st.dataframe(optimization_report)
        
        # Create download button
        excel_data = to_excel_download(affiliate_df_processed, advanced_df_processed, optimization_report)
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