import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

st.set_page_config(page_title="Partner Optimization Report Generator", layout="wide")

def extract_pid_subid(url):
    """Extract PID and SUBID from URL."""
    try:
        # Find all numbers after %3D
        matches = re.findall(r'%3D(\d+)_?([0-9]+)?', url)
        if matches:
            pid = matches[0][0]
            # Check if SUBID exists and contains only numbers
            subid = matches[0][1] if matches[0][1] and matches[0][1].isdigit() else ''
            return pid, subid
        return None, None
    except:
        return None, None

def process_dataframe(df):
    """Process dataframe to add PID, SUBID, and partnerID columns."""
    # Create new columns
    df['PID'] = ''
    df['SUBID'] = ''
    df['partnerID'] = ''
    
    # Process each row
    for idx, row in df.iterrows():
        if 'URL' in df.columns and pd.notna(row['URL']):
            pid, subid = extract_pid_subid(row['URL'])
            df.at[idx, 'PID'] = pid if pid else ''
            df.at[idx, 'SUBID'] = subid if subid else ''
            # Create partnerID
            if pid:
                df.at[idx, 'partnerID'] = f"{pid}_{subid}" if subid else f"{pid}_"
    
    return df

def analyze_data(affiliate_df, advanced_df):
    """Perform analysis on the processed dataframes."""
    # Group by partnerID
    affiliate_grouped = affiliate_df.groupby('partnerID').agg({
        'Leads': 'sum',
        'Spend': 'sum'
    }).reset_index()
    
    advanced_grouped = advanced_df.groupby('partnerID').agg({
        'Net Sales Amount': 'sum',
        'Order ID': 'count'
    }).reset_index()
    
    # Merge the dataframes
    merged_df = pd.merge(affiliate_grouped, advanced_grouped, 
                        on='partnerID', how='outer').fillna(0)
    
    # Rename columns
    merged_df.columns = ['partnerID', 'Leads', 'Spend', 'Revenue', 'Sales']
    
    # Calculate metrics
    analysis_df = pd.DataFrame()
    analysis_df['partnerID'] = merged_df['partnerID']
    analysis_df['Leads to Sale'] = merged_df['Sales'] / merged_df['Leads']
    analysis_df['ROAS'] = merged_df['Revenue'] / merged_df['Spend']
    analysis_df['Current Rate'] = merged_df['Spend'] / merged_df['Leads']
    analysis_df['ECPL at $1.50'] = 1.50 * merged_df['Leads']
    
    return merged_df, analysis_df

def to_excel_download(df1, df2):
    """Convert dataframes to Excel file for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Main Metrics', index=False)
        df2.to_excel(writer, sheet_name='Analysis', index=False)
    return output.getvalue()

# Main app
st.title("Partner Optimization Report Generator")

st.write("""
This tool processes your marketing data files and generates a comprehensive analysis report.
Please upload your affiliate leads QA file and advanced action sheet below.
""")

col1, col2 = st.columns(2)

with col1:
    affiliate_file = st.file_uploader("Upload Affiliate Leads QA File", type=['csv'])
    
with col2:
    advanced_file = st.file_uploader("Upload Advanced Action Sheet", type=['csv'])

if affiliate_file and advanced_file:
    try:
        # Read and process files
        affiliate_df = pd.read_csv(affiliate_file)
        advanced_df = pd.read_csv(advanced_file)
        
        # Process both dataframes
        affiliate_df_processed = process_dataframe(affiliate_df)
        advanced_df_processed = process_dataframe(advanced_df)
        
        # Perform analysis
        main_metrics_df, analysis_df = analyze_data(affiliate_df_processed, advanced_df_processed)
        
        # Show preview of results
        st.subheader("Preview of Main Metrics")
        st.dataframe(main_metrics_df.head())
        
        st.subheader("Preview of Analysis")
        st.dataframe(analysis_df.head())
        
        # Create download button
        excel_data = to_excel_download(main_metrics_df, analysis_df)
        st.download_button(
            label="Download Full Report",
            data=excel_data,
            file_name="partner_optimization_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"An error occurred while processing the files: {str(e)}")
else:
    st.info("Please upload both files to generate the report.") 