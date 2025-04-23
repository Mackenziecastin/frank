import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# -------------------------------
# Constants
# -------------------------------

TFN_SHEET_URL = "https://docs.google.com/spreadsheets/d/10BHN_-Wz_ZPmi7rezNtqiDPTguHOoNzmkXzovFOTbaU/edit#gid=1629976834"

# -------------------------------
# Helper Functions
# -------------------------------

def clean_affiliate_code(code):
    if pd.isna(code): return ''
    
    # Split into parts
    parts = code.split('_')
    if len(parts) < 2: return ''  # Need at least OfferID_PID
    
    # Always keep the PID (second part)
    pid = parts[1]
    
    # If there's a subID (third part), check if it's numeric
    if len(parts) > 2:
        subid = parts[2]
        # Only include subID if it's purely numeric
        if subid.isdigit():
            return f"{pid}_{subid}"
    
    # Default case: just return PID_
    return f"{pid}_"

def proportional_allocation(row, web_val, total_web_val, total_phone_val):
    """Allocate phone metrics proportionally based on web totals."""
    try:
        # Convert all values to float
        web_value = float(row[web_val])
        total_web = float(row[total_web_val])
        total_phone = float(total_phone_val)
        
        if total_web == 0 or pd.isna(total_web): return 0
        return round(total_phone * (web_value / total_web))
    except (ValueError, TypeError):
        return 0

def calculate_projected_installs(row):
    pct = 0.5 if str(row['Concatenated']).startswith('4790') else 0.7
    return int(round(row['Total DIFM Sales'] * pct))

def get_current_rates(conversion_df):
    # Convert date and clean IDs
    conversion_df['Conversion Date'] = pd.to_datetime(conversion_df['Conversion Date'], errors='coerce')
    conversion_df['Sub ID'] = conversion_df['Sub ID'].astype(str).apply(lambda x: x if x.isdigit() else '')
    conversion_df['Affiliate ID'] = conversion_df['Affiliate ID'].astype(str)
    
    # Create composite key for matching
    conversion_df['Composite Key'] = conversion_df['Affiliate ID'] + '_' + conversion_df['Sub ID']
    
    # Sort by date descending and get most recent rate for each composite key
    current_rates = (
        conversion_df.sort_values('Conversion Date', ascending=False)
        .dropna(subset=['Paid'])
        .groupby('Composite Key')['Paid']
        .first()
        .reset_index()
        .rename(columns={'Paid': 'Current Rate'})
    )
    
    # Debug output
    st.write("\n### Current Rate Calculation Debug")
    st.write("Sample of current rates:")
    st.write(current_rates.head())
    st.write(f"Total unique rates found: {len(current_rates)}")
    
    return current_rates

def load_combined_resi_tfn_data(sheet_url):
    base_url = sheet_url.split("/edit")[0]
    
    def sheet_csv_url(sheet_name):
        url = f"{base_url}/export?format=csv&gid="
        # RESI TFN Sheet gid
        if sheet_name == "RESI TFN Sheet":
            url += "1629976834"
        # Display TFN Sheet gid
        elif sheet_name == "Display TFN Sheet":
            url += "383243987"  # Correct GID for Display TFN Sheet
        return url
    
    try:
        # Load RESI sheet with first row as headers
        st.write("\nLoading RESI TFN Sheet...")
        resi_df = pd.read_csv(sheet_csv_url("RESI TFN Sheet"), header=0, na_values=['', 'nan', 'NaN', 'None'])
        
        # Convert PID to integer type immediately after loading
        resi_df['PID'] = pd.to_numeric(resi_df['PID'], errors='coerce').fillna('').astype(str).replace('\.0$', '', regex=True)
        
        # Clean the dataframe - replace NaN with empty string
        resi_df = resi_df.fillna('')
        
        st.write("RESI Sheet Columns:", [col for col in resi_df.columns.tolist() if col])
        st.write("RESI Sheet first few rows (non-empty rows only):")
        
        # Debug PID format right after loading
        st.write("\nPID format check after initial load:")
        st.write(resi_df[['Partner Name', 'PID', 'Code', 'Phone #']].head().to_dict('records'))
        
        # Filter rows that have data
        non_empty_mask = (resi_df['PID'].astype(str).str.strip().ne('')) & (resi_df['Phone #'].astype(str).str.strip().ne(''))
        sample_rows = resi_df[non_empty_mask].head()
        
        # Convert to records and clean up the output
        clean_records = []
        for record in sample_rows[['Partner Name', 'PID', 'Code', 'Phone #']].to_dict('records'):
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value) or value == '' or value == 'nan':
                    clean_record[key] = ''
                else:
                    clean_record[key] = str(value).strip().rstrip('.0')  # Ensure we remove any trailing .0
            clean_records.append(clean_record)
        st.write(clean_records)
        
        # Load Display sheet with first row as headers
        st.write("\nLoading Display TFN Sheet...")
        st.write("URL being used:", sheet_csv_url("Display TFN Sheet"))  # Debug URL
        display_df = pd.read_csv(sheet_csv_url("Display TFN Sheet"), header=0, na_values=['', 'nan', 'NaN', 'None'])
        
        # Debug raw column names
        st.write("Raw Display Sheet column names:", display_df.columns.tolist())
        
        # Try to find PID and TFN columns - they might have different names
        pid_column = None
        tfn_column = None
        
        # Common variations of column names
        pid_variations = ['PID', 'Pid', 'pid', 'Partner ID', 'PartnerID', 'ID']
        tfn_variations = ['TFN', 'Tfn', 'tfn', 'Phone Number', 'Phone #', 'PhoneNumber', 'Phone']
        
        for col in display_df.columns:
            # Debug each column name and its first few values
            st.write(f"Column '{col}' first few values:", display_df[col].head().tolist())
            
            # Check if this column might be PID or TFN
            if any(pid_var in col for pid_var in pid_variations):
                pid_column = col
            if any(tfn_var in col for tfn_var in tfn_variations):
                tfn_column = col
        
        st.write(f"Found PID column: {pid_column}")
        st.write(f"Found TFN column: {tfn_column}")
        
        if pid_column is None or tfn_column is None:
            st.error("Could not find required columns in Display TFN Sheet")
            st.error(f"Available columns: {display_df.columns.tolist()}")
            raise ValueError(f"Missing required columns in Display TFN Sheet. Need PID and TFN, found: {display_df.columns.tolist()}")
        
        # Rename columns to standard names
        display_df = display_df.rename(columns={
            pid_column: 'PID',
            tfn_column: 'TFN'
        })
        
        # Now proceed with the cleaning
        display_df['PID'] = pd.to_numeric(display_df['PID'], errors='coerce').fillna('').astype(str).replace('\.0$', '', regex=True)
        display_df = display_df.fillna('')
        
        st.write("Display Sheet Columns:", [col for col in display_df.columns.tolist() if col])
        st.write("Display Sheet first few rows (non-empty rows only):")
        non_empty_mask = (display_df['PID'].astype(str).str.strip().ne('')) & (display_df['TFN'].astype(str).str.strip().ne(''))
        sample_rows = display_df[non_empty_mask].head()
        
        # Convert to records and clean up the output
        clean_records = []
        for record in sample_rows[['PID', 'TFN']].to_dict('records'):
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value) or value == '' or value == 'nan':
                    clean_record[key] = ''
                else:
                    clean_record[key] = str(value).strip()
            clean_records.append(clean_record)
        st.write(clean_records)
        
        # Search for the number in RESI sheet
        st.write("\nSearching for 8446778720 in RESI sheet:")
        matches = resi_df[resi_df['Phone #'].astype(str).str.contains('8446778720', na=False)]
        if not matches.empty:
            clean_records = []
            for record in matches[['Partner Name', 'PID', 'Code', 'Phone #']].to_dict('records'):
                clean_record = {}
                for key, value in record.items():
                    if pd.isna(value) or value == '' or value == 'nan':
                        clean_record[key] = ''
                    else:
                        clean_record[key] = str(value).strip()
                clean_records.append(clean_record)
            st.write(clean_records)
        
        # Get the actual column names for TFN and PID
        tfn_col = 'Phone #'  # From the actual header row
        pid_col = 'PID'  # From the actual header row
        
        st.write("\nIdentified columns:")
        st.write(f"TFN column: {tfn_col}")
        st.write(f"PID column: {pid_col}")
        
        # Filter out rows with empty PIDs or phone numbers before combining
        resi_df = resi_df[
            resi_df[pid_col].astype(str).str.strip().ne('') & 
            resi_df[tfn_col].astype(str).str.strip().ne('')
        ]
        display_df = display_df[
            display_df['PID'].astype(str).str.strip().ne('') & 
            display_df['TFN'].astype(str).str.strip().ne('')
        ]
        
        # Combine sheets with correct column names
        combined_df = pd.concat([
            resi_df[[pid_col, tfn_col]].rename(columns={pid_col: "PID", tfn_col: "TFN"}),
            display_df[['PID', 'TFN']]
        ], ignore_index=True)
        
        # Clean TFNs - keep empty values as blank strings
        combined_df['Clean_TFN'] = combined_df['TFN'].fillna('').astype(str)
        combined_df.loc[combined_df['Clean_TFN'].str.strip() != '', 'Clean_TFN'] = combined_df.loc[combined_df['Clean_TFN'].str.strip() != '', 'Clean_TFN'].str.replace(r'[^0-9]', '', regex=True)
        
        # Clean PIDs - handle NaN and float formatting
        def clean_pid(x):
            try:
                if pd.isna(x) or str(x).strip() == '':
                    return ''
                # Remove any decimal points and trailing zeros
                numeric_str = str(x).strip()
                if '.' in numeric_str:
                    numeric_str = numeric_str.split('.')[0]
                return numeric_str if numeric_str.isdigit() else ''
            except (ValueError, TypeError):
                return ''
        
        combined_df['PID'] = combined_df['PID'].apply(clean_pid)
        
        # Debug PID cleaning
        st.write("\nPID Cleaning Example:")
        sample_pids = resi_df['PID'].head()
        st.write("Before cleaning:", [str(x) for x in sample_pids.tolist()])
        st.write("After cleaning:", [clean_pid(x) for x in sample_pids])
        
        # Remove any rows where either Clean_TFN or PID is empty after cleaning
        combined_df = combined_df[
            combined_df['Clean_TFN'].str.strip().ne('') & 
            combined_df['PID'].str.strip().ne('')
        ]
        
        # Debug final mapping
        st.write("\nFinal TFN mapping check:")
        st.write("Total records in mapping:", len(combined_df))
        st.write("Sample of final mapping:")
        clean_records = []
        for record in combined_df[['Clean_TFN', 'PID']].head(20).to_dict('records'):
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value) or value == '' or value == 'nan':
                    clean_record[key] = ''
                else:
                    clean_record[key] = str(value).strip()
            clean_records.append(clean_record)
        st.write(clean_records)
        
        st.write("\nChecking for 8446778720:")
        matches = combined_df[combined_df['Clean_TFN'] == '8446778720']
        if not matches.empty:
            clean_records = []
            for record in matches[['Clean_TFN', 'PID']].to_dict('records'):
                clean_record = {}
                for key, value in record.items():
                    if pd.isna(value) or value == '' or value == 'nan':
                        clean_record[key] = ''
                    else:
                        clean_record[key] = str(value).strip()
                clean_records.append(clean_record)
            st.write(clean_records)
        
        st.write("\nAll unique PIDs in mapping:")
        st.write(sorted([str(pid).strip() for pid in combined_df['PID'].unique().tolist() if str(pid).strip()]))
        
        return combined_df[['Clean_TFN', 'PID']]
        
    except Exception as e:
        st.error(f"Error loading TFN data: {str(e)}")
        st.error("Full error details:")
        import traceback
        st.error(traceback.format_exc())
        raise

def clean_athena(athena_df, tfn_df, leads_df, start_date, end_date):
    # Debug original Affiliate_Codes
    st.write("\n### Affiliate Code Cleaning Debug")
    st.write("Sample of original Affiliate_Codes:")
    sample_codes = athena_df['Affiliate_Code'].head(20).tolist()
    st.write(sample_codes)
    
    # Filter Athena data
    athena_df['Lead_Creation_Date'] = pd.to_datetime(athena_df['Lead_Creation_Date'], errors='coerce')
    athena_df = athena_df[
        (athena_df['Lead_Creation_Date'] >= start_date) &
        (athena_df['Lead_Creation_Date'] <= end_date)
    ]
    athena_df = athena_df[
        (athena_df['Ln_of_Busn'].str.lower() != 'health') &
        (athena_df['DNIS_BUSN_SEG_CD'].str.lower() != 'us: health') &
        (athena_df['Sale_Date'].notna()) &
        (athena_df['Ordr_Type'].str.upper().isin(['NEW', 'RESALE']))
    ]
    
    # Clean Affiliate_Code and show examples
    athena_df['Original_Code'] = athena_df['Affiliate_Code']  # Keep original for reference
    athena_df['Affiliate_Code'] = athena_df['Affiliate_Code'].apply(clean_affiliate_code)
    
    # Show cleaning results with examples of different cases
    st.write("\nAffiliate Code Cleaning Results:")
    cleaning_examples = pd.DataFrame({
        'Original': athena_df['Original_Code'],
        'Cleaned': athena_df['Affiliate_Code']
    })
    
    # Show examples of different cases
    st.write("Case 1 - OfferID_PID_NumericSubID (should keep PID and subID):")
    st.write(cleaning_examples[
        cleaning_examples['Original'].str.contains('_\d+_\d+$', na=False)
    ].head())
    
    st.write("\nCase 2 - OfferID_PID_NonNumericSubID (should keep only PID_):")
    st.write(cleaning_examples[
        cleaning_examples['Original'].str.contains('_\d+_[^\d]+', na=False)
    ].head())
    
    st.write("\nCase 3 - OfferID_PID (should keep PID_):")
    st.write(cleaning_examples[
        cleaning_examples['Original'].str.count('_') == 1
    ].head())
    
    # Handle Lead_DNIS and PID matching
    athena_df['Lead_DNIS'] = athena_df['Lead_DNIS'].fillna('').astype(str)
    
    # Debug TFN mapping
    st.write("\n### TFN Matching Debug")
    st.write("TFN mapping from sheet:")
    st.write(tfn_df[['Clean_TFN', 'PID']].head(20))
    
    # Create TFN mapping dictionary
    tfn_map = dict(zip(tfn_df['Clean_TFN'], tfn_df['PID']))
    
    # Debug specific mapping
    st.write("\nChecking specific TFN in mapping:")
    st.write(f"Is '8446778720' in TFN map? {'8446778720' in tfn_map}")
    if '8446778720' in tfn_map:
        st.write(f"Mapped PID for 8446778720: {tfn_map['8446778720']}")
    
    # Get non-WEB records for debugging
    non_web_records = athena_df[~athena_df['Lead_DNIS'].str.contains("WEB", na=False)]
    st.write("\nSample of non-WEB Lead_DNIS values to match:")
    st.write(non_web_records['Lead_DNIS'].head(10))
    
    # Debug specific DNIS
    st.write("\nLooking for specific DNIS in Athena data:")
    matching_records = athena_df[athena_df['Lead_DNIS'].str.contains('8446778720', na=False)]
    st.write("Found records with 8446778720:", len(matching_records))
    st.write("Sample of these records (Lead_DNIS only):")
    st.write(matching_records['Lead_DNIS'].head())
    
    # Match PIDs for non-WEB records
    def match_pid(row):
        dnis = row['Lead_DNIS']
        if 'WEB' not in dnis:
            # Extract only numeric characters for matching
            numeric_dnis = ''.join(c for c in dnis if c.isdigit())
            matched_pid = tfn_map.get(numeric_dnis, '')
            # Debug specific number
            if numeric_dnis == '8446778720':
                st.write(f"Found target DNIS: {dnis}")
                st.write(f"Numeric version: {numeric_dnis}")
                st.write(f"Matched PID: {matched_pid}")
            return matched_pid
        return None
    
    # Apply PID matching
    athena_df['PID'] = athena_df.apply(match_pid, axis=1)
    
    # Show matching results
    st.write("\nMatching Results:")
    st.write(f"Total non-WEB records: {len(non_web_records)}")
    matched_records = athena_df[athena_df['PID'].notna() & (athena_df['PID'] != '')]
    st.write(f"Successfully matched records: {len(matched_records)}")
    
    # Process leads data
    leads_df.columns = [col.lower() for col in leads_df.columns]
    leads_df['subid'] = leads_df['subid'].apply(lambda x: str(x) if str(x).isdigit() else '')
    leads_df['pid'] = leads_df['pid'].astype(str)
    leads_df['Concatenated'] = leads_df.apply(lambda r: f"{r['pid']}_{r['subid']}" if r['subid'] else f"{r['pid']}_", axis=1)
    leads_df['phone'] = leads_df['phone'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    phone_map = dict(zip(leads_df['phone'], leads_df['Concatenated']))
    
    # Match phone numbers for WEB leads
    athena_df['Primary_Phone_Customer_ANI'] = athena_df['Primary_Phone_Customer_ANI'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    def fill_code(row):
        if row['Affiliate_Code'] == '' and 'WEB' in row['Lead_DNIS']:
            return phone_map.get(row['Primary_Phone_Customer_ANI'], '')
        return row['Affiliate_Code']
    athena_df['Affiliate_Code'] = athena_df.apply(fill_code, axis=1)
    
    return athena_df

def generate_pivots(athena_df):
    # Split web and phone data
    web_df = athena_df[athena_df['Lead_DNIS'].str.contains("WEB", na=False)]
    phone_df = athena_df[~athena_df['Lead_DNIS'].str.contains("WEB", na=False) & athena_df['PID'].notna()]
    
    # Debug info
    st.write("\n### Data Summary")
    st.write(f"Web records: {len(web_df)}")
    st.write(f"Phone records: {len(phone_df)}")
    
    # Create web pivot if we have data
    if len(web_df) > 0:
        web_pivot = pd.pivot_table(
            web_df, 
            index='Affiliate_Code', 
            values=['Sale_Date', 'Install_Date'], 
            columns='INSTALL_METHOD', 
            aggfunc='count', 
            fill_value=0
        )
        # Rename columns to match expected format
        web_pivot.columns = [f"{col}_{val}" for col, val in web_pivot.columns]
        web_pivot = web_pivot.reset_index()
        
        # Rename columns to match expected names
        column_mapping = {
            'Sale_Date_DIFM': 'Web DIFM Sales',
            'Sale_Date_DIY': 'Web DIY Sales',
            'Install_Date_DIFM': 'DIFM Web Installs',
            'Install_Date_DIY': 'DIY Web Installs'
        }
        web_pivot = web_pivot.rename(columns=column_mapping)
    else:
        web_pivot = pd.DataFrame({
            'Affiliate_Code': [],
            'Web DIFM Sales': [],
            'Web DIY Sales': [],
            'DIFM Web Installs': [],
            'DIY Web Installs': []
        })
    
    # Create phone pivot if we have data
    if len(phone_df) > 0:
        phone_pivot = pd.pivot_table(
            phone_df, 
            index='PID', 
            values=['Sale_Date', 'Install_Date'], 
            columns='INSTALL_METHOD', 
            aggfunc='count', 
            fill_value=0
        )
        # Rename columns to match expected format
        phone_pivot.columns = [f"{col}_{val}" for col, val in phone_pivot.columns]
        phone_pivot = phone_pivot.reset_index()
        
        # Rename columns to match expected names
        column_mapping = {
            'Sale_Date_DIFM': 'Phone DIFM Sales',
            'Sale_Date_DIY': 'Phone DIY Sales',
            'Install_Date_DIFM': 'DIFM Phone Installs',
            'Install_Date_DIY': 'DIY Phone Installs'
        }
        phone_pivot = phone_pivot.rename(columns=column_mapping)
    else:
        phone_pivot = pd.DataFrame({
            'PID': [],
            'Phone DIFM Sales': [],
            'Phone DIY Sales': [],
            'DIFM Phone Installs': [],
            'DIY Phone Installs': []
        })
    
    # Ensure all required columns exist with 0s
    for df, cols in [
        (web_pivot, ['Web DIFM Sales', 'Web DIY Sales', 'DIFM Web Installs', 'DIY Web Installs']),
        (phone_pivot, ['Phone DIFM Sales', 'Phone DIY Sales', 'DIFM Phone Installs', 'DIY Phone Installs'])
    ]:
        for col in cols:
            if col not in df.columns:
                df[col] = 0
    
    # Display Web Pivot Table
    st.write("\n### Web Pivot Table")
    if not web_pivot.empty:
        st.write("Full Web Pivot Data:")
        # Convert numeric columns to integers for display
        numeric_cols = ['Web DIFM Sales', 'Web DIY Sales', 'DIFM Web Installs', 'DIY Web Installs']
        for col in numeric_cols:
            if col in web_pivot.columns:
                web_pivot[col] = web_pivot[col].fillna(0).astype(int)
        
        # Sort by total sales for better visualization
        web_pivot['Total Sales'] = web_pivot['Web DIFM Sales'] + web_pivot['Web DIY Sales']
        web_pivot = web_pivot.sort_values('Total Sales', ascending=False)
        web_pivot = web_pivot.drop('Total Sales', axis=1)
        
        st.dataframe(web_pivot)
        
        # Create bar chart for web data
        web_metrics = web_pivot.melt(
            id_vars=['Affiliate_Code'],
            value_vars=['Web DIFM Sales', 'Web DIY Sales', 'DIFM Web Installs', 'DIY Web Installs']
        )
        
        # Create a more readable bar chart
        fig_web = px.bar(
            web_metrics,
            x='Affiliate_Code',
            y='value',
            color='variable',
            title='Web Channel Metrics by Affiliate',
            labels={'value': 'Count', 'variable': 'Metric'},
            barmode='group'
        )
        
        # Rotate x-axis labels for better readability
        fig_web.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Affiliate Code (PID_SubID)",
            height=600  # Make the chart taller
        )
        
        st.plotly_chart(fig_web)
        
        # Show summary statistics
        st.write("\nWeb Channel Summary:")
        summary = {
            'Total DIFM Sales': web_pivot['Web DIFM Sales'].sum(),
            'Total DIY Sales': web_pivot['Web DIY Sales'].sum(),
            'Total DIFM Installs': web_pivot['DIFM Web Installs'].sum(),
            'Total DIY Installs': web_pivot['DIY Web Installs'].sum(),
            'Unique Affiliate Codes': len(web_pivot)
        }
        st.write(summary)
    else:
        st.write("No web data available")
    
    # Display Phone Pivot Table
    st.write("\n### Phone Pivot Table")
    if not phone_pivot.empty:
        st.write("Full Phone Pivot Data:")
        # Convert numeric columns to integers for display
        numeric_cols = ['Phone DIFM Sales', 'Phone DIY Sales', 'DIFM Phone Installs', 'DIY Phone Installs']
        for col in numeric_cols:
            if col in phone_pivot.columns:
                phone_pivot[col] = phone_pivot[col].fillna(0).astype(int)
        
        # Sort by total sales for better visualization
        phone_pivot['Total Sales'] = phone_pivot['Phone DIFM Sales'] + phone_pivot['Phone DIY Sales']
        phone_pivot = phone_pivot.sort_values('Total Sales', ascending=False)
        phone_pivot = phone_pivot.drop('Total Sales', axis=1)
        
        st.dataframe(phone_pivot)
        
        # Create bar chart for phone data
        phone_metrics = phone_pivot.melt(
            id_vars=['PID'],
            value_vars=['Phone DIFM Sales', 'Phone DIY Sales', 'DIFM Phone Installs', 'DIY Phone Installs']
        )
        
        # Create a more readable bar chart
        fig_phone = px.bar(
            phone_metrics,
            x='PID',
            y='value',
            color='variable',
            title='Phone Channel Metrics by PID',
            labels={'value': 'Count', 'variable': 'Metric'},
            barmode='group'
        )
        
        # Rotate x-axis labels for better readability
        fig_phone.update_layout(
            xaxis_tickangle=-45,
            height=600  # Make the chart taller
        )
        
        st.plotly_chart(fig_phone)
        
        # Show summary statistics
        st.write("\nPhone Channel Summary:")
        summary = {
            'Total DIFM Sales': phone_pivot['Phone DIFM Sales'].sum(),
            'Total DIY Sales': phone_pivot['Phone DIY Sales'].sum(),
            'Total DIFM Installs': phone_pivot['DIFM Phone Installs'].sum(),
            'Total DIY Installs': phone_pivot['DIY Phone Installs'].sum(),
            'Unique PIDs': len(phone_pivot)
        }
        st.write(summary)
    else:
        st.write("No phone data available")
    
    # Debug pivot tables
    st.write("\nWeb pivot columns:", web_pivot.columns.tolist())
    st.write("Phone pivot columns:", phone_pivot.columns.tolist())
    
    return web_pivot, phone_pivot

def clean_conversion(conversion_df):
    # Filter out specific offer IDs
    conversion_df = conversion_df[~conversion_df['Offer ID'].isin([31908, 31989])]
    
    # Clean Sub ID and create Concatenated
    conversion_df['Sub ID'] = conversion_df['Sub ID'].apply(lambda x: str(x) if str(x).isdigit() else '')
    conversion_df['Affiliate ID'] = conversion_df['Affiliate ID'].astype(str)
    conversion_df['Concatenated'] = conversion_df.apply(lambda r: f"{r['Affiliate ID']}_{r['Sub ID']}" if r['Sub ID'] else f"{r['Affiliate ID']}_", axis=1)
    
    # Create Cake pivot
    cake = conversion_df.groupby('Concatenated').agg({
        'Affiliate ID': lambda x: str(x.iloc[0]),
        'Paid': 'sum',
        'Concatenated': 'count'
    }).rename(columns={'Affiliate ID': 'PID', 'Paid': 'Cost', 'Concatenated': 'Leads'}).reset_index()
    
    return cake

def allocate_phone_metrics(cake_df):
    """Allocate phone metrics to subIDs based on web activity."""
    st.write("\n### Phone Attribution Debug")
    
    # Convert phone metrics to integers
    phone_metrics = ['Phone DIFM Sales', 'Phone DIY Sales', 'DIFM Phone Installs']
    for metric in phone_metrics:
        cake_df[metric] = cake_df[metric].fillna(0).astype(int)
    
    # Group by PID to handle each partner's phone metrics
    for pid in cake_df['PID'].unique():
        if pd.isna(pid) or pid == '': continue
        
        pid_mask = cake_df['PID'] == pid
        pid_rows = cake_df[pid_mask]
        
        # Skip if no phone metrics
        if pid_rows['Phone DIFM Sales'].sum() == 0 and pid_rows['Phone DIY Sales'].sum() == 0:
            continue
            
        st.write(f"\nProcessing PID: {pid}")
        st.write("Phone metrics to allocate:", {
            'DIFM Sales': int(pid_rows['Phone DIFM Sales'].sum()),
            'DIY Sales': int(pid_rows['Phone DIY Sales'].sum()),
            'DIFM Installs': int(pid_rows['DIFM Phone Installs'].sum())
        })
        
        # Step 1: Proportional Allocation
        total_web_difm = float(pid_rows['Web DIFM Sales'].sum())
        total_web_diy = float(pid_rows['Web DIY Sales'].sum())
        total_web_installs = float(pid_rows['DIFM Web Installs'].sum())
        
        if total_web_difm > 0 or total_web_diy > 0 or total_web_installs > 0:
            # Allocate proportionally
            for idx in pid_rows.index:
                row = pid_rows.loc[idx]
                
                # Allocate DIFM Sales
                if total_web_difm > 0:
                    allocated = int(proportional_allocation(
                        row, 'Web DIFM Sales', 'Web DIFM Sales',
                        float(pid_rows['Phone DIFM Sales'].sum())
                    ))
                    cake_df.loc[idx, 'Phone DIFM Sales'] = allocated
                
                # Allocate DIY Sales
                if total_web_diy > 0:
                    allocated = int(proportional_allocation(
                        row, 'Web DIY Sales', 'Web DIY Sales',
                        float(pid_rows['Phone DIY Sales'].sum())
                    ))
                    cake_df.loc[idx, 'Phone DIY Sales'] = allocated
                
                # Allocate DIFM Installs
                if total_web_installs > 0:
                    allocated = int(proportional_allocation(
                        row, 'DIFM Web Installs', 'DIFM Web Installs',
                        float(pid_rows['DIFM Phone Installs'].sum())
                    ))
                    cake_df.loc[idx, 'DIFM Phone Installs'] = allocated
        
        # Step 2: Fix Under-Allocated Totals
        for metric in phone_metrics:
            total_allocated = int(cake_df.loc[pid_mask, metric].sum())
            total_available = int(pid_rows[metric].sum())
            
            if total_allocated < total_available:
                remaining = total_available - total_allocated
                st.write(f"Under-allocated {metric}: {remaining} units remaining")
                
                # Sort by Leads or Web DIFM Sales
                sorted_idx = pid_rows.sort_values('Leads', ascending=False).index
                for idx in sorted_idx[:remaining]:
                    cake_df.loc[idx, metric] += 1
        
        # Step 3: Catch-All Attribution (No Web Activity)
        if total_web_difm == 0 and total_web_diy == 0 and total_web_installs == 0:
            st.write("No web activity found - using catch-all attribution")
            
            # Find row with highest Leads
            max_leads_idx = pid_rows['Leads'].idxmax()
            
            # Assign all phone metrics to this row
            for metric in phone_metrics:
                total = int(pid_rows[metric].sum())
                cake_df.loc[max_leads_idx, metric] = total
                # Set other rows to 0
                other_idx = pid_rows[pid_rows.index != max_leads_idx].index
                cake_df.loc[other_idx, metric] = 0
    
    # Ensure all phone metrics are integers
    for metric in phone_metrics:
        cake_df[metric] = cake_df[metric].fillna(0).astype(int)
    
    return cake_df

def merge_and_compute(cake, web, phone):
    # Debug info
    st.write("\n### Merge Debug Info")
    st.write("Cake columns:", cake.columns.tolist())
    st.write("Web columns:", web.columns.tolist())
    st.write("Phone columns:", phone.columns.tolist())
    
    # Prepare for merge
    cake = cake.copy()
    web = web.copy() if not web.empty else pd.DataFrame()
    phone = phone.copy() if not phone.empty else pd.DataFrame()
    
    # Ensure required columns exist in web DataFrame
    web_cols = ['Web DIFM Sales', 'Web DIY Sales', 'DIFM Web Installs', 'DIY Web Installs']
    phone_cols = ['Phone DIFM Sales', 'Phone DIY Sales', 'DIFM Phone Installs', 'DIY Phone Installs']
    
    if not web.empty:
        web = web.set_index('Affiliate_Code')
        for col in web_cols:
            if col not in web.columns:
                web[col] = 0
    else:
        web = pd.DataFrame(columns=['Affiliate_Code'] + web_cols)
        web = web.set_index('Affiliate_Code')
    
    if not phone.empty:
        phone = phone.set_index('PID')
        for col in phone_cols:
            if col not in phone.columns:
                phone[col] = 0
    else:
        phone = pd.DataFrame(columns=['PID'] + phone_cols)
        phone = phone.set_index('PID')
    
    # Debug before merge
    st.write("\nBefore merge:")
    st.write("Web DataFrame:")
    st.write(web.head().to_dict('records'))
    st.write("Phone DataFrame:")
    st.write(phone.head().to_dict('records'))
    
    # Merge data
    if not web.empty:
        cake = cake.merge(web, how='left', left_on='Concatenated', right_index=True)
    if not phone.empty:
        cake = cake.merge(phone, how='left', left_on='PID', right_index=True)
    
    # Fill NaN values with 0
    cake = cake.fillna(0)
    
    # Debug after merge
    st.write("\nAfter merge:")
    st.write("Cake columns:", cake.columns.tolist())
    st.write("Sample of merged data:")
    st.write(cake.head().to_dict('records'))
    
    # After merging and before calculating totals, allocate phone metrics
    cake = allocate_phone_metrics(cake)
    
    # Calculate totals
    cake['Total DIFM Sales'] = cake['Web DIFM Sales'] + cake['Phone DIFM Sales']
    cake['Total DIY Sales'] = cake['Web DIY Sales'] + cake['Phone DIY Sales']
    cake['Total DIFM Installs'] = cake['DIFM Web Installs'] + cake['DIFM Phone Installs']
    
    # Calculate revenue metrics
    cake['Revenue'] = 1080 * cake['Total DIFM Installs'] + 300 * cake['Total DIY Sales']
    cake['Profit/Loss'] = cake['Revenue'] - cake['Cost']
    cake['Projected Installs'] = cake.apply(calculate_projected_installs, axis=1)
    cake['Projected Revenue'] = 1080 * cake['Projected Installs'] + 300 * cake['Total DIY Sales']
    cake['Projected Profit/Loss'] = cake['Projected Revenue'] - cake['Cost']
    cake['Projected Margin'] = np.where(cake['Projected Revenue'] == 0, -1, cake['Projected Profit/Loss'] / cake['Projected Revenue'])
    cake['eCPL'] = np.where(cake['Leads'] == 0, 0, cake['Projected Revenue'] / cake['Leads'])
    
    # Format numeric columns
    cake['Revenue'] = cake['Revenue'].apply(lambda x: f"${x:,.2f}")
    cake['Profit/Loss'] = cake['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
    cake['Projected Revenue'] = cake['Projected Revenue'].apply(lambda x: f"${x:,.2f}")
    cake['Projected Profit/Loss'] = cake['Projected Profit/Loss'].apply(lambda x: f"${x:,.2f}")
    cake['Cost'] = cake['Cost'].apply(lambda x: f"${x:,.2f}")
    cake['eCPL'] = cake['eCPL'].apply(lambda x: f"${x:,.2f}")
    cake['Projected Margin'] = cake['Projected Margin'].apply(lambda x: f"{x:.2%}" if x != -1 else "-")
    
    # Add Current Rate column if not present
    if 'Current Rate' not in cake.columns:
        cake['Current Rate'] = 0
    
    # Reorder columns
    columns = [
        'Concatenated', 'PID', 'Leads', 'Cost', 'Current Rate',
        'Web DIFM Sales', 'Phone DIFM Sales', 'Total DIFM Sales',
        'DIFM Web Installs', 'DIFM Phone Installs', 'Total DIFM Installs',
        'Web DIY Sales', 'Phone DIY Sales', 'Total DIY Sales',
        'Revenue', 'Profit/Loss',
        'Projected Installs', 'Projected Revenue', 'Projected Profit/Loss',
        'Projected Margin', 'eCPL'
    ]
    cake = cake[columns]
    
    # Debug final metrics
    st.write("\nFinal metrics check:")
    st.write("Sample of final metrics:")
    st.write(cake[columns].head().to_dict('records'))
    
    return cake

def compare_with_reference(computed_df):
    try:
        # Load reference report
        reference_df = pd.read_csv('Final_Formatted_Optimization_Report.csv')
        
        # Remove formatting from computed df for numeric comparison
        numeric_df = computed_df.copy()
        for col in ['Cost', 'Revenue', 'Profit/Loss', 'Projected Revenue', 'Projected Profit/Loss', 'eCPL']:
            numeric_df[col] = numeric_df[col].str.replace('$', '').str.replace(',', '').astype(float)
        numeric_df['Projected Margin'] = numeric_df['Projected Margin'].replace('-', float('nan')).str.rstrip('%').astype(float) / 100
        
        # Remove formatting from reference df
        ref_numeric = reference_df.copy()
        for col in ['Cost', 'Revenue', 'Profit/Loss', 'Projected Revenue', 'Projected Profit/Loss', 'eCPL']:
            ref_numeric[col] = ref_numeric[col].str.replace('$', '').str.replace(',', '').astype(float)
        ref_numeric['Projected Margin'] = ref_numeric['Projected Margin'].replace('-', float('nan')).str.rstrip('%').astype(float) / 100
        
        # Compare web sales row by row
        st.write("### Detailed Web Sales Comparison")
        st.write("Comparing web sales numbers for each Concatenated value...")
        
        # Merge the dataframes on Concatenated to compare row by row
        comparison = pd.merge(
            numeric_df[['Concatenated', 'Web DIFM Sales', 'DIFM Web Installs', 'DIY Web Sales']],
            ref_numeric[['Concatenated', 'Web DIFM Sales', 'DIFM Web Installs', 'DIY Web Sales']],
            on='Concatenated',
            how='outer',
            suffixes=('_computed', '_reference')
        )
        
        # Find rows with differences
        differences = comparison[
            (comparison['Web DIFM Sales_computed'] != comparison['Web DIFM Sales_reference']) |
            (comparison['DIFM Web Installs_computed'] != comparison['DIFM Web Installs_reference']) |
            (comparison['DIY Web Sales_computed'] != comparison['DIY Web Sales_reference'])
        ]
        
        if len(differences) > 0:
            st.write(f"Found {len(differences)} rows with differences in web sales/installs")
            st.write("Sample of differences (first 10 rows):")
            st.dataframe(differences.head(10))
            
            # Show summary of differences
            st.write("Summary of differences:")
            st.write(f"Total Web DIFM Sales - Computed: {numeric_df['Web DIFM Sales'].sum():,}")
            st.write(f"Total Web DIFM Sales - Reference: {ref_numeric['Web DIFM Sales'].sum():,}")
            st.write(f"Total DIFM Web Installs - Computed: {numeric_df['DIFM Web Installs'].sum():,}")
            st.write(f"Total DIFM Web Installs - Reference: {ref_numeric['DIFM Web Installs'].sum():,}")
            st.write(f"Total DIY Web Sales - Computed: {numeric_df['DIY Web Sales'].sum():,}")
            st.write(f"Total DIY Web Sales - Reference: {ref_numeric['DIY Web Sales'].sum():,}")
        else:
            st.success("No differences found in web sales/installs numbers!")
        
        # Continue with other comparisons...
        metrics = [
            'Leads', 'Phone DIFM Sales', 'Total DIFM Sales',
            'DIFM Phone Installs', 'Total DIFM Installs',
            'DIY Phone Sales', 'Total DIY Sales',
            'Projected Installs'
        ]
        
        st.write("### Comparison with Reference Report")
        st.write("Checking other key metrics for differences...")
        
        for metric in metrics:
            computed_sum = numeric_df[metric].sum()
            reference_sum = ref_numeric[metric].sum()
            if abs(computed_sum - reference_sum) > 0.01:  # Allow for small floating point differences
                st.error(f"Discrepancy in {metric}:")
                st.error(f"Our calculation: {computed_sum:,.0f}")
                st.error(f"Reference: {reference_sum:,.0f}")
                st.error(f"Difference: {computed_sum - reference_sum:,.0f}")
        
        # Compare monetary metrics
        monetary_metrics = ['Cost', 'Revenue', 'Profit/Loss', 'Projected Revenue', 'Projected Profit/Loss', 'eCPL']
        for metric in monetary_metrics:
            computed_sum = numeric_df[metric].sum()
            reference_sum = ref_numeric[metric].sum()
            if abs(computed_sum - reference_sum) > 0.01:  # Allow for small floating point differences
                st.error(f"Discrepancy in {metric}:")
                st.error(f"Our calculation: ${computed_sum:,.2f}")
                st.error(f"Reference: ${reference_sum:,.2f}")
                st.error(f"Difference: ${computed_sum - reference_sum:,.2f}")
        
        # Check for any missing or extra rows
        if len(numeric_df) != len(ref_numeric):
            st.error(f"Row count mismatch:")
            st.error(f"Our rows: {len(numeric_df)}")
            st.error(f"Reference rows: {len(ref_numeric)}")
        
        # If no discrepancies found
        if not st.session_state.get('has_error', False):
            st.success("All numbers match the reference report!")
            
    except Exception as e:
        st.error(f"Error comparing reports: {str(e)}")
        st.error("Full error details:")
        import traceback
        st.error(traceback.format_exc())

def show_bob_analysis():
    st.title("ADT Partner Optimization Analysis")
    
    st.write("""
    This tool analyzes ADT partner performance data to generate optimization reports.
    Please select the lead creation date range and upload the required files below.
    """)
    
    # Date Range Selection
    st.subheader("Select Lead Creation Date Range")
    col_date1, col_date2 = st.columns(2)
    
    with col_date1:
        start_date = st.date_input(
            "Start Date (Lead Creation)",
            value=pd.Timestamp.now().replace(day=1),  # First day of current month
            help="Select the start date for lead creation"
        )
    
    with col_date2:
        end_date = st.date_input(
            "End Date (Lead Creation)",
            value=pd.Timestamp.now(),  # Current date
            help="Select the end date for lead creation"
        )
    
    if start_date > end_date:
        st.error("Error: End date must be after start date")
        return
    
    # Convert dates to datetime
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # File uploaders in three columns
    st.subheader("Upload Files")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        athena_file = st.file_uploader(
            "1. Athena Report (CSV)", 
            type=['csv'],
            help="Upload the Athena report containing sales and installation data"
        )
    
    with col2:
        cake_conversion_file = st.file_uploader(
            "2. Cake Conversion Report (CSV)", 
            type=['csv'],
            help="Upload the Cake Conversion report for costs and lead counts"
        )
    
    with col3:
        database_leads_file = st.file_uploader(
            "3. Database Leads (CSV)", 
            type=['csv'],
            help="Upload the Database Leads file for affiliate code matchback. Required columns: Subid, PID, Phone"
        )
    
    if athena_file and cake_conversion_file and database_leads_file:
        try:
            # Load and process data with debugging
            st.write("DEBUG: Starting data loading...")
            
            try:
                athena_df = pd.read_csv(athena_file)
                st.write("DEBUG: Successfully loaded Athena file")
                st.write("Athena columns:", athena_df.columns.tolist())
            except Exception as e:
                st.error(f"Error loading Athena file: {str(e)}")
                return
            
            try:
                conversion_df = pd.read_csv(cake_conversion_file)
                st.write("DEBUG: Successfully loaded Conversion file")
                st.write("Conversion columns:", conversion_df.columns.tolist())
            except Exception as e:
                st.error(f"Error loading Conversion file: {str(e)}")
                return
            
            try:
                leads_df = pd.read_csv(database_leads_file)
                st.write("DEBUG: Successfully loaded Database Leads file")
                st.write("Database Leads columns:", leads_df.columns.tolist())
                st.write("Database Leads data types:")
                st.write(leads_df.dtypes)
                st.write("First few rows of Database Leads:")
                st.write(leads_df.head())
            except Exception as e:
                st.error(f"Error loading Database Leads file: {str(e)}")
                return
            
            st.write("DEBUG: Loading TFN data...")
            try:
                tfn_df = load_combined_resi_tfn_data(TFN_SHEET_URL)
                st.write("DEBUG: Successfully loaded TFN data")
            except Exception as e:
                st.error(f"Error loading TFN data: {str(e)}")
                return
            
            # Display date range being analyzed
            st.info(f"Analyzing leads created from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Step 1: Clean Athena + Leads Report
            st.write("DEBUG: Starting clean_athena function...")
            try:
                athena_df = clean_athena(athena_df, tfn_df, leads_df, start_date, end_date)
                st.write("DEBUG: Successfully cleaned Athena data")
            except Exception as e:
                st.error("Error in clean_athena function")
                st.error(f"Error details: {str(e)}")
                import traceback
                st.error(f"Full traceback:\n{traceback.format_exc()}")
                return
            
            # Display record counts
            total_leads = len(athena_df)
            if total_leads == 0:
                st.warning("No leads found in the selected creation date range. Please adjust the dates and try again.")
                return
            
            st.write(f"Total leads created in date range: {total_leads:,}")
            
            # Step 2: Generate Web + Phone Pivots
            st.write("DEBUG: Generating pivots...")
            try:
                web_pivot, phone_pivot = generate_pivots(athena_df)
                st.write("DEBUG: Successfully generated pivots")
            except Exception as e:
                st.error(f"Error generating pivots: {str(e)}")
                return
            
            # Step 3: Clean Conversion Report
            st.write("DEBUG: Cleaning conversion report...")
            try:
                cake_df = clean_conversion(conversion_df)
                st.write("DEBUG: Successfully cleaned conversion report")
            except Exception as e:
                st.error(f"Error cleaning conversion report: {str(e)}")
                return
            
            # Step 4-5: Merge and Compute Final Metrics
            st.write("DEBUG: Merging and computing metrics...")
            try:
                final_df = merge_and_compute(cake_df, web_pivot, phone_pivot)
                st.write("DEBUG: Successfully computed metrics")
            except Exception as e:
                st.error(f"Error computing metrics: {str(e)}")
                return
            
            # Display optimization report
            st.subheader("Partner Optimization Report")
            
            # Select columns for display
            display_columns = [
                'Concatenated', 'PID', 'Leads', 'Cost', 'Current Rate',
                'Web DIFM Sales', 'Phone DIFM Sales', 'Total DIFM Sales',
                'DIFM Web Installs', 'DIFM Phone Installs', 'Total DIFM Installs',
                'Web DIY Sales', 'Phone DIY Sales', 'Total DIY Sales',
                'Revenue', 'Profit/Loss',
                'Projected Installs', 'Projected Revenue', 'Projected Profit/Loss',
                'Projected Margin', 'eCPL'
            ]
            
            # Format the dataframe
            display_df = final_df[display_columns].copy()
            
            # Display the table
            st.dataframe(
                display_df.sort_values('Projected Revenue', ascending=False),
                use_container_width=True
            )
            
            # Compare with reference report
            compare_with_reference(final_df)
            
            # Export functionality
            st.subheader("Export Report")
            if st.button("Export to Excel"):
                try:
                    # Create a BytesIO object to hold the Excel file in memory
                    output = io.BytesIO()
                    
                    # Create Excel writer object
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        final_df.to_excel(writer, sheet_name='Partner Performance', index=False)
                    
                    # Seek to the beginning of the BytesIO object
                    output.seek(0)
                    
                    # Create download button
                    st.download_button(
                        label="Download Excel Report",
                        data=output,
                        file_name=f"adt_optimization_report_leads_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error exporting report: {str(e)}")
                    st.error("Full error details:")
                    import traceback
                    st.error(traceback.format_exc())
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Please check your input files and try again.")
            st.error("Required columns in Database Leads file: Subid, PID, Phone")
            columns_found = leads_df.columns.tolist() if 'leads_df' in locals() else ["No file loaded"]
            st.error(f"Current columns found: {', '.join(str(col) for col in columns_found)}")
            import traceback
            st.error(f"Full traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    show_bob_analysis() 