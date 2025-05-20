import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
import re
import os
import io
import tempfile

def create_partner_list_df():
    """Create the partner list DataFrame directly in code with comprehensive TFN and affiliate data."""
    # Define the data as a dictionary with all required columns
    data = {
        'Affiliate ID': [
            '41382', '42215', '42216', '42217', '42218', '42219', 
            '42220', '42221', '42222', '42223', '42224', 
            '42225', '42226', '42227', '42228', '42229',
            '42230', '42231', '42232', '42233', '42234',
            '42235', '42236', '42237', '42238', '42239',
            '42240', '42241', '42242', '42243', '42244',
            '42245', '42246', '42247', '42248', '42249',
            '42250', '42251', '42252', '42253', '42254'
        ],
        'Affiliate Name': [
            'Brinks Home Security', 'PNW Kartik', 'PNW Kartik 2', 'PNW Kartik 3', 
            'PNW Kartik 4', 'PNW Kartik 5', 'PNW Kartik 6', 'PNW Kartik 7', 
            'PNW Kartik 8', 'PNW Kartik 9', 'PNW Kartik 10',
            'PNW Sam', 'PNW Sam 2', 'PNW Sam 3', 'PNW Sam 4', 'PNW Sam 5',
            'PNW Sam 6', 'PNW Sam 7', 'PNW Sam 8', 'PNW Sam 9', 'PNW Sam 10',
            'MediaAlpha Brinks', 'MediaAlpha Brinks 2', 'MediaAlpha Brinks 3', 'MediaAlpha Brinks 4', 'MediaAlpha Brinks 5',
            'Aragon Brinks', 'Aragon Brinks 2', 'Aragon Brinks 3', 'Aragon Brinks 4', 'Aragon Brinks 5',
            'Fluent Brinks', 'Fluent Brinks 2', 'Fluent Brinks 3', 'Fluent Brinks 4', 'Fluent Brinks 5',
            'CW Brinks', 'CW Brinks 2', 'CW Brinks 3', 'CW Brinks 4', 'CW Brinks 5'
        ],
        'Account Manager Name': [
            'Internal', 'Kartik', 'Kartik', 'Kartik', 'Kartik', 'Kartik',
            'Kartik', 'Kartik', 'Kartik', 'Kartik', 'Kartik',
            'Sam', 'Sam', 'Sam', 'Sam', 'Sam',
            'Sam', 'Sam', 'Sam', 'Sam', 'Sam',
            'MediaAlpha', 'MediaAlpha', 'MediaAlpha', 'MediaAlpha', 'MediaAlpha',
            'Aragon', 'Aragon', 'Aragon', 'Aragon', 'Aragon',
            'Fluent', 'Fluent', 'Fluent', 'Fluent', 'Fluent',
            'CW', 'CW', 'CW', 'CW', 'CW'
        ],
        'Name': [
            'Brinks Home Security', 'PNW Kartik', 'PNW Kartik 2', 'PNW Kartik 3',
            'PNW Kartik 4', 'PNW Kartik 5', 'PNW Kartik 6', 'PNW Kartik 7',
            'PNW Kartik 8', 'PNW Kartik 9', 'PNW Kartik 10',
            'PNW Sam', 'PNW Sam 2', 'PNW Sam 3', 'PNW Sam 4', 'PNW Sam 5',
            'PNW Sam 6', 'PNW Sam 7', 'PNW Sam 8', 'PNW Sam 9', 'PNW Sam 10',
            'MediaAlpha Brinks', 'MediaAlpha Brinks 2', 'MediaAlpha Brinks 3', 'MediaAlpha Brinks 4', 'MediaAlpha Brinks 5',
            'Aragon Brinks', 'Aragon Brinks 2', 'Aragon Brinks 3', 'Aragon Brinks 4', 'Aragon Brinks 5',
            'Fluent Brinks', 'Fluent Brinks 2', 'Fluent Brinks 3', 'Fluent Brinks 4', 'Fluent Brinks 5',
            'CW Brinks', 'CW Brinks 2', 'CW Brinks 3', 'CW Brinks 4', 'CW Brinks 5'
        ],
        'TFN': [
            '800-447-9239', '844-677-8720', '844-677-8721', '844-677-8722',
            '844-677-8723', '844-677-8724', '844-677-8725', '844-677-8726',
            '844-677-8727', '844-677-8728', '844-677-8729',
            '855-834-5222', '855-834-5223', '855-834-5224', '855-834-5225', '855-834-5226',
            '855-834-5227', '855-834-5228', '855-834-5229', '855-834-5230', '855-834-5231',
            '866-325-4591', '866-325-4592', '866-325-4593', '866-325-4594', '866-325-4595',
            '877-522-8608', '877-522-8609', '877-522-8610', '877-522-8611', '877-522-8612',
            '888-647-9392', '888-647-9393', '888-647-9394', '888-647-9395', '888-647-9396',
            '844-575-9302', '844-575-9303', '844-575-9304', '844-575-9305', '844-575-9306'
        ],
        'Status': ['Active'] * 41,  # All partners are active
        'Vertical': ['Security'] * 41,  # All are in Security vertical
        'Sub-Vertical': ['Residential'] * 41  # All are Residential
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure all string columns are string type
    string_columns = ['Affiliate ID', 'Affiliate Name', 'Account Manager Name', 'Name', 'TFN', 'Status', 'Vertical', 'Sub-Vertical']
    for col in string_columns:
        df[col] = df[col].astype(str)
    
    # Add mapping of TFN to Affiliate ID for easier lookup
    tfn_to_id = {row['TFN']: row['Affiliate ID'] for _, row in df.iterrows()}
    
    st.session_state['tfn_to_id_mapping'] = tfn_to_id
    
    return df

def robust_read_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1')

def show_brinks_optimization():
    """Display the Brinks Optimization Report interface with two file uploaders"""
    try:
        st.title("Brinks Optimization Report")
        
        # Display this message so we know we're seeing the latest version
        st.write("Version: 2023-05-11 - Robust Against Session State Issues")
        
        # Create partner list with TFN data automatically
        partner_list_df = create_partner_list_df()
        
        st.write("""
        This tool processes Brinks marketing data files and generates optimization reports.
        The TFN sheet data is already built into the tool, so you don't need to upload it separately.
        Upload your Brinks Sales Report and Conversion Report (CSV format) to begin.
        """)
        
        # First file uploader for Sales Report
        col1, col2 = st.columns(2)
        
        with col1:
            sales_file = st.file_uploader("Upload Brinks Sales Report (CSV)", 
                                           type=['csv'], 
                                           key='brinks_sales_report')
            if sales_file is not None:
                st.success("✅ Sales Report uploaded")
        
        # Second file uploader for Conversion Report
        with col2:
            conversion_file = st.file_uploader("Upload Brinks Conversion Report (CSV)", 
                                              type=['csv'], 
                                              key='brinks_conversion_report')
            if conversion_file is not None:
                st.success("✅ Conversion Report uploaded")
        
        # Display TFN mapping in a collapsible section
        with st.expander("View TFN to Affiliate ID Mapping"):
            st.dataframe(partner_list_df[['Affiliate ID', 'Affiliate Name', 'TFN', 'Account Manager Name']])
        
        # Only show the button if both files are uploaded
        if sales_file is not None and conversion_file is not None:
            if st.button("Generate Optimization Report"):
                try:
                    # Save uploaded files temporarily
                    temp_dir = tempfile.mkdtemp()
                    sales_path = os.path.join(temp_dir, sales_file.name)
                    conversion_path = os.path.join(temp_dir, conversion_file.name)
                    
                    with open(sales_path, 'wb') as f:
                        f.write(sales_file.getvalue())
                    
                    with open(conversion_path, 'wb') as f:
                        f.write(conversion_file.getvalue())
                    
                    # Read the files robustly
                    sales_df = robust_read_csv(sales_path)
                    conversion_df = robust_read_csv(conversion_path)
                    
                    # Process the sales and conversion reports
                    processed_sales_df = process_lead_source_sales(sales_df, partner_list_df)
                    processed_conversion_df = process_conversion_report(conversion_df)
                    
                    # Create pivot tables
                    sales_pivot = create_sales_pivot(processed_sales_df)
                    conversion_pivot = create_conversion_pivot(processed_conversion_df)
                    
                    # Create final report
                    final_report = create_final_report(sales_pivot, conversion_pivot, partner_list_df)
                    
                    # Clean up temporary files and directory
                    os.unlink(sales_path)
                    os.unlink(conversion_path)
                    os.rmdir(temp_dir)
                    
                    # Show success message
                    st.success("Report generated successfully!")
                    
                    # Display the final report
                    st.subheader("Brinks Optimization Report")
                    st.dataframe(final_report)
                    
                    # Create Excel file for download
                    excel_data = to_excel(processed_sales_df, processed_conversion_df, final_report)
                    today = datetime.now().strftime("%m.%d")
                    st.download_button(
                        label="Download Optimization Report",
                        data=excel_data,
                        file_name=f"Brinks Optimization Report {today}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Display the data previews in expandable sections
                    with st.expander("Sales Report Preview"):
                        st.dataframe(processed_sales_df)
                    
                    with st.expander("Conversion Report Preview"):
                        st.dataframe(processed_conversion_df)
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("Please upload both files to generate the report")
    
    except Exception as e:
        st.error(f"Error loading Brinks Optimization Report interface: {str(e)}")
        st.error("Please try refreshing the page. If the error persists, contact support.")
        import traceback
        st.code(traceback.format_exc())

def load_file(file):
    """Minimal file loading with engine specification"""
    # Get file extension from name
    file_ext = file.name.split('.')[-1].lower()
    
    # For Excel files - try both engines
    if file_ext in ['xlsx', 'xls']:
        try:
            # Try with openpyxl (modern Excel files)
            return pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            # If that fails, try with xlrd (older Excel files)
            return pd.read_excel(file, engine='xlrd')
    
    # For CSV files
    elif file_ext == 'csv':
        # Try to read with latin-1 encoding (handles any byte value)
        return pd.read_csv(file, encoding='latin-1')
    
    # Unsupported file type
    else:
        raise ValueError(f"Unsupported file format: {file.name}")

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
    final_report['SSPR'] = final_report.apply(
        lambda row: row['Sales'] / row['Leads'] if row['Leads'] > 0 else 0,
        axis=1
    )
    
    final_report['MU'] = final_report.apply(
        lambda row: row['Client CPL'] - row['Pub CPL'] if row['Pub CPL'] > 0 else 0,
        axis=1
    )
    
    final_report['MU %'] = final_report.apply(
        lambda row: row['MU'] / row['Client CPL'] * 100 if row['Client CPL'] > 0 else 0,
        axis=1
    )
    
    # Look up the Affiliate Name using pid_subid (first part before underscore)
    def get_affiliate_name(pid_subid):
        if pd.isna(pid_subid) or pid_subid == "":
            return "Unknown"
        
        # Get the first part (before underscore)
        if '_' in pid_subid:
            affiliate_id = pid_subid.split('_')[0]
        else:
            affiliate_id = pid_subid
        
        # Handle special case
        if affiliate_id == '41382' and pid_subid.endswith('_2'):
            affiliate_id = '41382'
        
        # Look up in partner list
        if partner_list_df is not None:
            match = partner_list_df[partner_list_df['Affiliate ID'] == affiliate_id]
            if not match.empty:
                return match.iloc[0]['Name']
        
        return f"Unknown ({affiliate_id})"
    
    # Add the Affiliate Name column
    final_report['Affiliate Name'] = final_report['pid_subid'].apply(get_affiliate_name)
    
    # Sort by SSPR (descending)
    final_report = final_report.sort_values(by='SSPR', ascending=False)
    
    # Format percentage columns
    final_report['SSPR'] = final_report['SSPR'].apply(lambda x: f"{x:.2%}")
    final_report['MU %'] = final_report['MU %'].apply(lambda x: f"{x:.2f}%")
    
    # Format currency columns
    for col in ['Cost', 'Revenue', 'Pub CPL', 'Client CPL', 'MU']:
        final_report[col] = final_report[col].apply(lambda x: f"${x:.2f}")
    
    st.write("Final report created with shape:", final_report.shape)
    
    return final_report

def to_excel(sales_df, conversion_df, final_report):
    """Create an Excel file with multiple sheets from the DataFrames"""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each DataFrame to a different sheet
        final_report.to_excel(writer, sheet_name='Optimization Report', index=False)
        sales_df.to_excel(writer, sheet_name='Sales Data', index=False)
        conversion_df.to_excel(writer, sheet_name='Conversion Data', index=False)
        
        # Get xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Optimization Report']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })
        
        # Set the column width and format
        for i, col in enumerate(final_report.columns):
            # Get the max column length
            max_len = max(
                final_report[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            
            # Set column width
            worksheet.set_column(i, i, max_len)
        
        # Set the header row format
        for col_num, value in enumerate(final_report.columns.values):
            worksheet.write(0, col_num, value, header_format)
    
    buffer.seek(0)
    return buffer.getvalue()

# Helper functions will be added here based on your specific requirements 