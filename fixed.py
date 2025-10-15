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
    """Enhanced CSV file reader that handles problematic files with inconsistent delimiters or quotes"""
    # Try multiple approaches to read the CSV file
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            # First attempt: standard reading with automatic quoting detection
            return pd.read_csv(path, encoding=encoding)
        except Exception as e1:
            try:
                # Second attempt: force quoted CSV format
                return pd.read_csv(path, encoding=encoding, quoting=pd.io.common.csv.QUOTE_ALL)
            except Exception as e2:
                try:
                    # Third attempt: with quotechar as double-quote and escapechar
                    return pd.read_csv(path, encoding=encoding, quotechar='"', escapechar='\\')
                except Exception as e3:
                    try:
                        # Fourth attempt: with quotechar as double-quote and allowing line breaks inside quotes
                        try:
                            # Try with old parameter names first
                            return pd.read_csv(path, encoding=encoding, quotechar='"', doublequote=True, 
                                              error_bad_lines=False, warn_bad_lines=True)
                        except Exception as param_error:
                            # If it fails due to deprecated parameters, try with newer names
                            if "error_bad_lines" in str(param_error) or "warn_bad_lines" in str(param_error):
                                return pd.read_csv(path, encoding=encoding, quotechar='"', doublequote=True, 
                                                  on_bad_lines='skip')
                            else:
                                # Re-raise if it's not a parameter issue
                                raise
                    except Exception as e4:
                        try:
                            # Fifth attempt: Python's built-in CSV reader with custom processing
                            import csv
                            import io
                            
                            st.warning("Using fallback CSV parser due to formatting issues")
                            
                            # Read file content
                            with open(path, 'r', encoding=encoding) as file:
                                content = file.read()
                            
                            # Parse using Python's CSV module which is more forgiving
                            lines = []
                            reader = csv.reader(io.StringIO(content))
                            header = next(reader)  # Get header
                            
                            for row in reader:
                                # Pad or truncate rows to match header length
                                if len(row) < len(header):
                                    row.extend([''] * (len(header) - len(row)))
                                elif len(row) > len(header):
                                    row = row[:len(header)]
                                lines.append(row)
                            
                            # Create DataFrame from processed lines
                            return pd.DataFrame(lines, columns=header)
                        except Exception as e5:
                            if encoding == 'cp1252':  # This was our last attempt
                                st.error(f"All CSV parsing methods failed. Error details: {str(e5)}")
                                # Re-raise the most informative error
                                raise Exception(f"Could not parse CSV file: {str(e1)}\nTried multiple approaches but all failed.")
                            # Otherwise continue to next encoding
                            continue
    
    # This should never be reached because the last encoding attempt will either return or raise
    raise Exception("Failed to read CSV file with all available methods")

def show_brinks_optimization():
    """Display the Brinks Optimization Report interface with two file uploaders"""
    try:
        st.title("Brinks Optimization Report")
        
        # Display this message so we know we're seeing the latest version
        st.write("Version: 2023-05-15 - With Custom TFN Mapping Upload Option")
        
        st.write("""
        This tool processes Brinks marketing data files and generates optimization reports.
        Upload your Brinks Sales Report, Conversion Report, and optionally your own TFN mapping file to begin.
        """)
        
        # Option to use built-in TFN mapping or upload custom
        use_custom_tfn = st.checkbox("Upload my own TFN mapping file", value=False)
        
        if use_custom_tfn:
            tfn_file = st.file_uploader("Upload TFN Mapping File (CSV)", 
                                       type=['csv'], 
                                       key='tfn_mapping_file',
                                       help="CSV file with Affiliate ID and TFN columns")
            if tfn_file is not None:
                try:
                    # Read the custom TFN mapping file
                    tfn_df = pd.read_csv(tfn_file)
                    # Validate columns
                    required_cols = ['Affiliate ID', 'TFN']
                    if all(col in tfn_df.columns for col in required_cols):
                        st.success("✅ Custom TFN mapping file uploaded successfully")
                        partner_list_df = tfn_df
                    else:
                        st.error(f"The TFN mapping file must contain these columns: {', '.join(required_cols)}")
                        st.info("Using built-in TFN mapping as fallback")
                        partner_list_df = create_partner_list_df()
                except Exception as e:
                    st.error(f"Error reading TFN mapping file: {str(e)}")
                    st.info("Using built-in TFN mapping as fallback")
                    partner_list_df = create_partner_list_df()
            else:
                st.info("Please upload a TFN mapping file or uncheck the box to use the built-in mapping")
                partner_list_df = create_partner_list_df()
        else:
            # Use the built-in TFN data
            partner_list_df = create_partner_list_df()
            
        # Display TFN mapping in a collapsible section
        with st.expander("View Current TFN to Affiliate ID Mapping"):
            st.dataframe(partner_list_df[['Affiliate ID', 'Affiliate Name', 'TFN', 'Account Manager Name']] 
                         if 'Affiliate Name' in partner_list_df.columns and 'Account Manager Name' in partner_list_df.columns
                         else partner_list_df[['Affiliate ID', 'TFN']])
        
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
                    
                    # Display file info for debugging
                    st.info("Analyzing uploaded files...")
                    
                    # Read the first few lines of each file to diagnose issues
                    def analyze_csv_structure(file_path, file_type):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                lines = [f.readline().strip() for _ in range(5)]  # Read first 5 lines
                            
                            st.write(f"### {file_type} File Structure Analysis")
                            for i, line in enumerate(lines):
                                field_count = len(line.split(','))
                                st.write(f"Line {i+1}: {field_count} fields")
                                if i < 3:  # Show first 3 lines content
                                    st.code(line, language="csv")
                                    
                            # Check for quotes in file that might indicate CSV format issues
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read(5000)  # Read first 5000 chars
                                has_quotes = '"' in content
                                has_commas_in_quotes = re.search(r'"[^"]*,[^"]*"', content) is not None
                                
                            if has_quotes and has_commas_in_quotes:
                                st.warning("⚠️ Found commas inside quoted fields, which might cause parsing issues.")
                                
                        except Exception as e:
                            st.error(f"Error analyzing file structure: {str(e)}")
                    
                    # Analyze both files
                    with st.expander("CSV Structure Analysis"):
                        analyze_csv_structure(sales_path, "Sales Report")
                        analyze_csv_structure(conversion_path, "Conversion Report")
                    
                    try:
                        # Read the files robustly
                        st.info("Attempting to parse Sales Report...")
                        sales_df = robust_read_csv(sales_path)
                        st.success(f"✅ Sales Report parsed successfully with {sales_df.shape[0]} rows and {sales_df.shape[1]} columns")
                        
                        st.info("Attempting to parse Conversion Report...")
                        conversion_df = robust_read_csv(conversion_path)
                        st.success(f"✅ Conversion Report parsed successfully with {conversion_df.shape[0]} rows and {conversion_df.shape[1]} columns")
                        
                        # Show column names from both files for verification
                        with st.expander("File Columns"):
                            st.write("Sales Report Columns:", list(sales_df.columns))
                            st.write("Conversion Report Columns:", list(conversion_df.columns))
                        
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
                        
                        # Provide more helpful context about CSV errors
                        if "Error tokenizing data" in str(e):
                            st.error("""
                            CSV parsing error detected. This typically happens when:
                            1. A field contains a comma but isn't properly quoted
                            2. The file has inconsistent number of columns across rows
                            3. There are special characters or line breaks within fields
                            
                            Try the following:
                            - Open the CSV in Excel or Google Sheets and re-save it
                            - Make sure all text fields with commas are properly quoted
                            - Ensure consistent column count across all rows
                            """)
                            
                            # Show the lines around the problematic area if possible
                            error_line_match = re.search(r'line (\d+)', str(e))
                            if error_line_match:
                                try:
                                    error_line = int(error_line_match.group(1))
                                    st.write(f"Issue detected around line {error_line}. Showing nearby rows:")
                                    
                                    with open(sales_path if "sales" in str(e).lower() else conversion_path, 'r', encoding='utf-8', errors='replace') as f:
                                        lines = f.readlines()
                                        
                                    start_line = max(0, error_line - 2)
                                    end_line = min(len(lines), error_line + 2)
                                    
                                    for i in range(start_line, end_line):
                                        st.code(f"Line {i+1}: {lines[i].strip()}")
                                except Exception as parse_error:
                                    st.error(f"Error displaying problematic lines: {str(parse_error)}")
                        
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.info("Please upload both Sales and Conversion files to generate the report")
    
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
    
    # Check if required columns exist
    required_columns = ["Pardot_Partner_ID", "First ACD Call Keyword"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in Sales Report: {', '.join(missing_columns)}")
        st.write("Available columns:", list(df.columns))
        
        # Try to identify potential substitute columns
        for missing_col in missing_columns:
            if missing_col == "Pardot_Partner_ID":
                potential_substitutes = [col for col in df.columns if "partner" in col.lower() or "id" in col.lower()]
                if potential_substitutes:
                    st.warning(f"Potential substitutes for 'Pardot_Partner_ID': {potential_substitutes}")
                    # Use the first potential substitute
                    df = df.rename(columns={potential_substitutes[0]: "Pardot_Partner_ID"})
                    st.info(f"Using '{potential_substitutes[0]}' as 'Pardot_Partner_ID'")
                else:
                    # Create empty column if no substitute found
                    df["Pardot_Partner_ID"] = ""
                    st.warning("Created empty 'Pardot_Partner_ID' column - results may be limited")
                    
            elif missing_col == "First ACD Call Keyword":
                potential_substitutes = [col for col in df.columns if "keyword" in col.lower() or "call" in col.lower()]
                if potential_substitutes:
                    st.warning(f"Potential substitutes for 'First ACD Call Keyword': {potential_substitutes}")
                    # Use the first potential substitute
                    df = df.rename(columns={potential_substitutes[0]: "First ACD Call Keyword"})
                    st.info(f"Using '{potential_substitutes[0]}' as 'First ACD Call Keyword'")
                else:
                    # Create empty column if no substitute found
                    df["First ACD Call Keyword"] = ""
                    st.warning("Created empty 'First ACD Call Keyword' column - results may be limited")
    
    # Sort by Pardot_Partner_ID (handle cases where it might be empty)
    if "Pardot_Partner_ID" in df.columns:
        df = df.sort_values(by="Pardot_Partner_ID", na_position='last')
    
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
    
    # Analyze post-processing results
    empty_partner_ids = df['Pardot_Partner_ID'].isna().sum()
    if empty_partner_ids > 0:
        st.warning(f"Found {empty_partner_ids} rows with empty Partner IDs after processing")
    
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
    
    # Check if required columns exist
    required_columns = ["Advertiser ID", "Sub ID"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in Conversion Report: {', '.join(missing_columns)}")
        st.write("Available columns:", list(df.columns))
        
        # Try to identify potential substitute columns
        for missing_col in missing_columns:
            if missing_col == "Advertiser ID":
                potential_substitutes = [col for col in df.columns if "advertiser" in col.lower() or "id" in col.lower()]
                if potential_substitutes:
                    st.warning(f"Potential substitutes for 'Advertiser ID': {potential_substitutes}")
                    # Use the first potential substitute
                    df = df.rename(columns={potential_substitutes[0]: "Advertiser ID"})
                    st.info(f"Using '{potential_substitutes[0]}' as 'Advertiser ID'")
                else:
                    # Create empty column if no substitute found
                    df["Advertiser ID"] = ""
                    st.warning("Created empty 'Advertiser ID' column - results may be limited")
                    
            elif missing_col == "Sub ID":
                potential_substitutes = [col for col in df.columns if "sub" in col.lower() or "id" in col.lower()]
                if potential_substitutes:
                    st.warning(f"Potential substitutes for 'Sub ID': {potential_substitutes}")
                    # Use the first potential substitute
                    df = df.rename(columns={potential_substitutes[0]: "Sub ID"})
                    st.info(f"Using '{potential_substitutes[0]}' as 'Sub ID'")
                else:
                    # Create empty column if no substitute found
                    df["Sub ID"] = ""
                    st.warning("Created empty 'Sub ID' column - results may be limited")
    
    # Check for required numeric columns
    required_numeric_columns = ["Lead ID", "Paid", "Received"]
    missing_numeric_columns = [col for col in required_numeric_columns if col not in df.columns]
    
    if missing_numeric_columns:
        st.error(f"Missing required numeric columns in Conversion Report: {', '.join(missing_numeric_columns)}")
        
        # Create missing columns with zeros
        for col in missing_numeric_columns:
            df[col] = 0
            st.warning(f"Created '{col}' column with zeros - results may be limited")
    
    # Ensure numeric columns are numeric
    for col in required_numeric_columns:
        if col in df.columns:
            # Try to convert string values (remove currency symbols, commas, etc.)
            if df[col].dtype == object:  # If it's a string or mixed type
                df[col] = df[col].astype(str).str.replace('$', '', regex=False)
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = df[col].astype(str).str.replace('"', '', regex=False)
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Create pid_subid column by concatenating Advertiser ID and Sub ID
    df['pid_subid'] = df.apply(
        lambda row: f"{row['Advertiser ID']}_{row['Sub ID']}" if pd.notna(row['Sub ID']) and str(row['Sub ID']).strip() != '' 
        else f"{row['Advertiser ID']}_", 
        axis=1
    )
    
    # Clean up the pid_subid column using the same rules
    df['pid_subid'] = df['pid_subid'].apply(clean_pardot_partner_id)
    
    st.write("Processed Conversion Report data shape:", df.shape)
    st.write("Sample of processed data:")
    
    display_columns = ['Advertiser ID', 'Sub ID', 'pid_subid']
    display_columns = [col for col in display_columns if col in df.columns]
    st.dataframe(df[display_columns].head(10))
    
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