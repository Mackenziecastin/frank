import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
import re
import os
import io

def show_brinks_optimization():
    """
    Main function to display the Brinks Optimization Report interface
    """
    st.title("Brinks Optimization Report")
    
    st.write("""
    This tool generates the Brinks Optimization Report by processing the Lead Source Sales and Conversion Report data.
    Please upload the required files below to get started.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lead_source_file = st.file_uploader("Upload Lead Source Sales Report (CSV)", type=["csv", "xlsx"])
    
    with col2:
        conversion_file = st.file_uploader("Upload Conversion Report (CSV)", type=["csv", "xlsx"])
    
    with col3:
        partner_list_file = st.file_uploader("Upload Internal Brinks Performance + TFNS (CSV)", type=["csv", "xlsx"])
    
    # Check if we have all required files
    if lead_source_file and conversion_file and partner_list_file:
        if st.button("Generate Optimization Report"):
            with st.spinner("Processing data and generating report..."):
                try:
                    # Load the data
                    lead_source_df = load_file(lead_source_file)
                    conversion_df = load_file(conversion_file)
                    partner_list_df = load_file(partner_list_file)
                    
                    # Process lead source sales data
                    st.subheader("Processing Lead Source Sales Data")
                    processed_sales_df = process_lead_source_sales(lead_source_df, partner_list_df)
                    
                    # Create sales pivot table
                    sales_pivot = create_sales_pivot(processed_sales_df)
                    
                    # Process conversion report data
                    st.subheader("Processing Conversion Report Data")
                    processed_conversion_df = process_conversion_report(conversion_df)
                    
                    # Create conversion pivot table
                    conversion_pivot = create_conversion_pivot(processed_conversion_df)
                    
                    # Merge the pivot tables and create final report
                    st.subheader("Creating Final Optimization Report")
                    final_report = create_final_report(sales_pivot, conversion_pivot, partner_list_df)
                    
                    # Display the final report
                    st.subheader("Brinks Optimization Report")
                    st.dataframe(final_report)
                    
                    # Download button for the final report
                    today = datetime.now().strftime("%m.%d")
                    excel_data = to_excel(processed_sales_df, processed_conversion_df, final_report)
                    st.download_button(
                        label="Download Optimization Report",
                        data=excel_data,
                        file_name=f"Brinks Optimization Report {today}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
    else:
        st.info("Please upload all three required files to generate the report.")
    
    # Add a section to explain the process
    with st.expander("How to use this tool"):
        st.markdown("""
        ### Instructions
        1. Upload the **Lead Source Sales Report** (CSV/Excel format)
        2. Upload the **Conversion Report** (CSV/Excel format)
        3. Upload the **Internal Brinks Performance + TFNS** file (CSV/Excel format)
        4. Click the **Generate Optimization Report** button
        5. Review the generated report and download it
        
        ### What this tool does
        - Processes the Lead Source Sales data and cleans the Pardot Partner IDs
        - Processes the Conversion Report data and creates the pid_subid column
        - Merges the data and calculates all required metrics
        - Generates a formatted Excel report with all necessary calculations
        """)

def load_file(file):
    """Simplified function to load files with fallback conversions"""
    try:
        # For Excel files
        if file.name.endswith(('.xlsx', '.xls')):
            try:
                # Try direct reading first
                return pd.read_excel(file)
            except Exception as excel_error:
                # If direct reading fails, try converting to CSV first
                st.warning(f"Direct Excel reading failed. Attempting conversion to CSV.")
                
                # Save file content
                file_content = file.getvalue()
                
                # Try using BytesIO to read the Excel file
                try:
                    # Use a memory buffer
                    buffer = BytesIO(file_content)
                    
                    # Try to read with openpyxl (newer Excel files)
                    df = pd.read_excel(buffer, engine='openpyxl')
                    return df
                except Exception as e:
                    # If that fails, try with xlrd (older Excel files)
                    try:
                        buffer = BytesIO(file_content)
                        df = pd.read_excel(buffer, engine='xlrd')
                        return df
                    except Exception as e:
                        st.error(f"All Excel reading methods failed: {str(e)}")
                        raise ValueError(f"Could not read Excel file: {file.name}")
            
        # For CSV files
        elif file.name.endswith('.csv'):
            try:
                # Try the most basic approach first
                return pd.read_csv(file)
            except UnicodeDecodeError:
                # Rewind the file and try with Latin-1 encoding
                file.seek(0)
                try:
                    return pd.read_csv(file, encoding='latin-1')
                except Exception as csv_error:
                    st.error(f"CSV reading failed: {str(csv_error)}")
                    raise ValueError(f"Could not read CSV file: {file.name}")
                
        else:
            st.error(f"Unsupported file format: {file.name}")
            raise ValueError(f"Unsupported file format: {file.name}")
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        raise

def clean_pardot_partner_id(pid):
    """Clean up the Pardot Partner ID according to the rules"""
    if pd.isna(pid) or pid == "":
        return pid
    
    pid_str = str(pid)
    
    # Special case for 41382
    if pid_str.startswith('41382'):
        return '41382_2'
    
    # Extract the first part (before underscore if present)
    if '_' in pid_str:
        base_pid = pid_str.split('_')[0]
        return f"{base_pid}_"
    
    # If no underscore, just add one
    return f"{pid_str}_"

def process_lead_source_sales(df, partner_list_df):
    """Process the Lead Source Sales report according to instructions"""
    st.write("Original Lead Source Sales data shape:", df.shape)
    
    # Sort by Pardot_Partner_ID
    df = df.sort_values(by="Pardot_Partner_ID")
    
    # Create a mapping from First ACD Call Keyword to Affiliate ID
    keyword_to_id_map = {}
    for _, row in partner_list_df.iterrows():
        if pd.notna(row['Name']) and pd.notna(row['Affiliate ID']):
            keyword_to_id_map[row['Name']] = row['Affiliate ID']
    
    # Fill empty Pardot_Partner_ID cells with matched Affiliate IDs from First ACD Call Keyword
    for i, row in df.iterrows():
        if pd.isna(row['Pardot_Partner_ID']) or row['Pardot_Partner_ID'] == "":
            keyword = row['First ACD Call Keyword']
            if pd.notna(keyword) and keyword in keyword_to_id_map:
                affiliate_id = keyword_to_id_map[keyword]
                df.at[i, 'Pardot_Partner_ID'] = affiliate_id
    
    # Clean up all Pardot_Partner_ID values
    df['Pardot_Partner_ID'] = df['Pardot_Partner_ID'].apply(clean_pardot_partner_id)
    
    st.write("Processed Lead Source Sales data shape:", df.shape)
    st.write("Sample of processed data:")
    st.dataframe(df[['Pardot_Partner_ID', 'First ACD Call Keyword']].head(10))
    
    return df

def create_sales_pivot(df):
    """Create a pivot table for sales data"""
    # Group by Pardot_Partner_ID and count the sales
    sales_pivot = df.groupby('Pardot_Partner_ID').size().reset_index(name='Sales')
    
    st.write("Sales pivot table created with shape:", sales_pivot.shape)
    st.write("Sample of sales pivot:")
    st.dataframe(sales_pivot.head(10))
    
    return sales_pivot

def process_conversion_report(df):
    """Process the Conversion Report according to instructions"""
    st.write("Original Conversion Report data shape:", df.shape)
    
    # Create pid_subid column by concatenating Advertiser ID and Sub ID
    df['pid_subid'] = df.apply(lambda row: f"{row['Advertiser ID']}_{row['Sub ID']}" if pd.notna(row['Sub ID']) else f"{row['Advertiser ID']}_", axis=1)
    
    # Clean up the pid_subid column using the same rules
    df['pid_subid'] = df['pid_subid'].apply(clean_pardot_partner_id)
    
    st.write("Processed Conversion Report data shape:", df.shape)
    st.write("Sample of processed data:")
    st.dataframe(df[['Advertiser ID', 'Sub ID', 'pid_subid']].head(10))
    
    return df

def create_conversion_pivot(df):
    """Create a pivot table for conversion data"""
    # Group by pid_subid and calculate metrics
    conversion_pivot = df.groupby('pid_subid').agg({
        'Lead ID': 'count',  # Count of leads
        'Paid': ['sum', 'mean'],  # Sum and average of Paid
        'Received': ['sum', 'mean']  # Sum and average of Received
    }).reset_index()
    
    # Flatten the column names
    conversion_pivot.columns = [
        'pid_subid',
        'Leads',
        'Cost',  # Sum of Paid
        'Pub CPL',  # Average of Paid
        'Revenue',  # Sum of Received
        'Client CPL'  # Average of Received
    ]
    
    st.write("Conversion pivot table created with shape:", conversion_pivot.shape)
    st.write("Sample of conversion pivot:")
    st.dataframe(conversion_pivot.head(10))
    
    return conversion_pivot

def create_final_report(sales_pivot, conversion_pivot, partner_list_df):
    """Create the final optimization report by merging the pivot tables"""
    # Merge the pivot tables on pid_subid = Pardot_Partner_ID
    final_report = pd.merge(
        conversion_pivot,
        sales_pivot,
        left_on='pid_subid',
        right_on='Pardot_Partner_ID',
        how='outer'
    ).fillna(0)
    
    # If pid_subid is missing, use Pardot_Partner_ID
    final_report['pid_subid'] = final_report.apply(
        lambda row: row['Pardot_Partner_ID'] if pd.isna(row['pid_subid']) or row['pid_subid'] == 0 else row['pid_subid'],
        axis=1
    )
    
    # Drop the Pardot_Partner_ID column if it exists (now redundant)
    if 'Pardot_Partner_ID' in final_report.columns:
        final_report = final_report.drop(columns=['Pardot_Partner_ID'])
    
    # Calculate additional metrics
    final_report['CPS'] = final_report.apply(
        lambda row: row['Revenue'] / row['Sales'] if row['Sales'] > 0 else 0,
        axis=1
    )
    
    final_report['Lead to Sale'] = final_report.apply(
        lambda row: row['Sales'] / row['Leads'] if row['Leads'] > 0 else 0,
        axis=1
    )
    
    final_report['Rate at $700'] = final_report.apply(
        lambda row: (row['Sales'] * 700) / row['Leads'] if row['Leads'] > 0 else 0,
        axis=1
    )
    
    final_report['Rate at 20%'] = final_report['Rate at $700'] * 0.8
    
    # Add Partner Name and Media Buyer columns
    partner_name_map = {}
    media_buyer_map = {}
    
    for _, row in partner_list_df.iterrows():
        if pd.notna(row['Affiliate ID']):
            affiliate_id = str(row['Affiliate ID'])
            # Add underscore to match pid_subid format
            if not affiliate_id.endswith('_'):
                affiliate_id = f"{affiliate_id}_"
            
            # Special case for 41382
            if affiliate_id.startswith('41382'):
                affiliate_id = '41382_2'
            
            partner_name_map[affiliate_id] = row['Affiliate Name'] if pd.notna(row['Affiliate Name']) else ""
            media_buyer_map[affiliate_id] = row['Account Manager Name'] if pd.notna(row['Account Manager Name']) else ""
    
    final_report['Partner Name'] = final_report['pid_subid'].map(partner_name_map).fillna("")
    final_report['Media Buyer'] = final_report['pid_subid'].map(media_buyer_map).fillna("")
    
    # Reorder columns to put Partner Name and Media Buyer first, followed by pid_subid
    column_order = [
        'Partner Name', 'Media Buyer', 'pid_subid',
        'Leads', 'Cost', 'Pub CPL', 'Revenue', 'Client CPL', 
        'Sales', 'CPS', 'Lead to Sale', 'Rate at $700', 'Rate at 20%'
    ]
    
    final_report = final_report[column_order]
    
    # Rename pid_subid to Partner ID for clarity
    final_report = final_report.rename(columns={'pid_subid': 'Partner ID'})
    
    st.write("Final report created with shape:", final_report.shape)
    
    return final_report

def to_excel(sales_df, conversion_df, final_report):
    """Convert the dataframes to an Excel file for download"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write each dataframe to a different sheet
        final_report.to_excel(writer, sheet_name='Optimization Report', index=False)
        sales_df.to_excel(writer, sheet_name='Processed Sales Data', index=False)
        conversion_df.to_excel(writer, sheet_name='Processed Conversion Data', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Optimization Report']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        money_format = workbook.add_format({'num_format': '$#,##0.00'})
        integer_format = workbook.add_format({'num_format': '0'})
        percent_format = workbook.add_format({'num_format': '0.0%'})
        
        # Apply header format
        for col_num, value in enumerate(final_report.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Set column widths and formats
        worksheet.set_column('A:B', 20)  # Partner Name, Media Buyer
        worksheet.set_column('C:C', 15)  # Partner ID
        worksheet.set_column('D:D', 10, integer_format)  # Leads
        worksheet.set_column('E:E', 12, money_format)  # Cost
        worksheet.set_column('F:F', 12, money_format)  # Pub CPL
        worksheet.set_column('G:G', 12, money_format)  # Revenue
        worksheet.set_column('H:H', 12, money_format)  # Client CPL
        worksheet.set_column('I:I', 10, integer_format)  # Sales
        worksheet.set_column('J:J', 12, money_format)  # CPS
        worksheet.set_column('K:K', 12, percent_format)  # Lead to Sale
        worksheet.set_column('L:L', 12, money_format)  # Rate at $700
        worksheet.set_column('M:M', 12, money_format)  # Rate at 20%
    
    return output.getvalue()

# Helper functions will be added here based on your specific requirements 