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
    parts = code.split('_', 1)  # Split on first underscore only
    if len(parts) < 2: return ''
    second = parts[1]
    if '_' in second:
        seg = second.split('_')[0]
        return f"{seg}_" if seg.isdigit() else ''
    return f"{second}_" if second.isdigit() else ''

def proportional_allocation(row, web_val, total_web_val, total_phone_val):
    if row[total_web_val] == 0 or pd.isna(row[total_web_val]): return 0
    return round(row[total_phone_val] * (row[web_val] / row[total_web_val]))

def calculate_projected_installs(row):
    pct = 0.5 if str(row['Concatenated']).startswith('4790') else 0.7
    return int(round(row['Total DIFM Sales'] * pct))

def get_current_rates(conversion_df):
    conversion_df['Conversion Date'] = pd.to_datetime(conversion_df['Conversion Date'], errors='coerce')
    conversion_df['Sub ID'] = conversion_df['Sub ID'].astype(str).apply(lambda x: x if x.isdigit() else '')
    conversion_df['Affiliate ID'] = conversion_df['Affiliate ID'].astype(str)
    conversion_df['Composite Key'] = conversion_df['Affiliate ID'] + '_' + conversion_df['Sub ID']
    return (
        conversion_df.sort_values(by='Conversion Date', ascending=False)
        .dropna(subset=['Paid'])
        .groupby('Composite Key')['Paid']
        .first()
        .reset_index()
        .rename(columns={'Paid': 'Current Rate'})
    )

def load_combined_resi_tfn_data(sheet_url):
    base_url = sheet_url.split("/edit")[0]
    
    # Debug URLs
    st.write("\nAccessing Google Sheets:")
    st.write("Base URL:", base_url)
    
    def sheet_csv_url(sheet_name):
        url = f"{base_url}/export?format=csv&gid="
        # RESI TFN Sheet gid
        if sheet_name == "RESI TFN Sheet":
            url += "1629976834"
        # Display TFN Sheet gid
        elif sheet_name == "Display TFN Sheet":
            url += "0"  # default first sheet
        st.write(f"\nURL for {sheet_name}:", url)
        return url
    
    try:
        # Load RESI sheet
        st.write("\nAttempting to load RESI TFN Sheet...")
        resi_df = pd.read_csv(sheet_csv_url("RESI TFN Sheet"))
        st.write("Successfully loaded RESI TFN Sheet")
        st.write("RESI Sheet Columns:", resi_df.columns.tolist())
        st.write("RESI Sheet first few rows to check column names and data:")
        st.write(resi_df.head())
        
        # Load Display sheet
        st.write("\nAttempting to load Display TFN Sheet...")
        display_df = pd.read_csv(sheet_csv_url("Display TFN Sheet"))
        st.write("Successfully loaded Display TFN Sheet")
        st.write("Display Sheet Columns:", display_df.columns.tolist())
        st.write("Display Sheet first few rows:")
        st.write(display_df.head())
        
        # Search for the number in any column of RESI sheet
        st.write("\nSearching for 8446778720 in any column of RESI sheet:")
        for col in resi_df.columns:
            matches = resi_df[resi_df[col].astype(str).str.contains('8446778720', na=False)]
            if not matches.empty:
                st.write(f"Found matches in column '{col}':")
                st.write(matches)
        
        # Get the actual column names for TFN and PID
        tfn_col = next((col for col in resi_df.columns if 'tfn' in col.lower()), None)
        pid_col = next((col for col in resi_df.columns if 'pid' in col.lower()), None)
        
        st.write("\nIdentified columns:")
        st.write(f"TFN column: {tfn_col}")
        st.write(f"PID column: {pid_col}")
        
        if not tfn_col or not pid_col:
            raise ValueError(f"Could not find TFN or PID columns. Available columns: {resi_df.columns.tolist()}")
        
        # Combine sheets with correct column names
        combined_df = pd.concat([
            resi_df.rename(columns={pid_col: "PID", tfn_col: "TFN"}),
            display_df.rename(columns={"Partner ID": "PID", "TFN": "TFN"})
        ], ignore_index=True)
        
        # Clean TFNs - keep empty values as blank strings
        combined_df['Clean_TFN'] = combined_df['TFN'].fillna('').astype(str)
        combined_df.loc[combined_df['Clean_TFN'].str.strip() != '', 'Clean_TFN'] = combined_df.loc[combined_df['Clean_TFN'].str.strip() != '', 'Clean_TFN'].str.replace(r'[^0-9]', '', regex=True)
        
        # Clean PIDs - handle NaN and float formatting
        combined_df['PID'] = combined_df['PID'].apply(
            lambda x: str(int(float(x))) if pd.notnull(x) and str(x).strip() != '' else ''
        )
        
        # Debug final mapping
        st.write("\nFinal TFN mapping check:")
        st.write("Total records in mapping:", len(combined_df))
        st.write("Sample of final mapping:")
        st.write(combined_df[['Clean_TFN', 'PID']].head(20))
        st.write("\nChecking for 8446778720:")
        st.write(combined_df[combined_df['Clean_TFN'] == '8446778720'])
        
        return combined_df[['Clean_TFN', 'PID']]
        
    except Exception as e:
        st.error(f"Error loading TFN data: {str(e)}")
        st.error("Full error details:")
        import traceback
        st.error(traceback.format_exc())
        raise

def clean_athena(athena_df, tfn_df, leads_df, start_date, end_date):
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
    
    # Clean Affiliate_Code
    athena_df['Affiliate_Code'] = athena_df['Affiliate_Code'].apply(clean_affiliate_code)
    
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
    st.write("\nSample of phone records:")
    if len(phone_df) > 0:
        st.write(phone_df[['Lead_DNIS', 'PID', 'INSTALL_METHOD']].head())
    
    # Initialize empty DataFrames
    web_pivot = pd.DataFrame(columns=['Affiliate_Code'])
    phone_pivot = pd.DataFrame(columns=['PID'])
    
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
        web_pivot.columns = [f"{val} {col}" for col, val in web_pivot.columns]
        web_pivot = web_pivot.reset_index()
    
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
        phone_pivot.columns = [f"{val} {col}" for col, val in phone_pivot.columns]
        phone_pivot = phone_pivot.reset_index()
    
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

def merge_and_compute(cake, web, phone):
    # Debug info
    st.write("\n### Merge Debug Info")
    st.write("Cake columns:", cake.columns.tolist())
    st.write("Web columns:", web.columns.tolist())
    st.write("Phone columns:", phone.columns.tolist())
    
    # Prepare for merge
    cake = cake.copy()
    web = web.set_index('Affiliate_Code') if not web.empty else pd.DataFrame()
    phone = phone.set_index('PID') if not phone.empty else pd.DataFrame()
    
    # Merge data
    if not web.empty:
        cake = cake.merge(web, how='left', left_on='Concatenated', right_index=True)
    if not phone.empty:
        cake = cake.merge(phone, how='left', left_on='PID', right_index=True)
    cake.fillna(0, inplace=True)
    
    # Extract metrics - handle both empty and non-empty cases
    cake['Web DIFM Sales'] = cake.get('DIFM Sale_Date_x', 0).astype(int)
    cake['Phone DIFM Sales'] = cake.get('DIFM Sale_Date_y', 0).astype(int)
    cake['Web DIY Sales'] = cake.get('DIY Sale_Date_x', 0).astype(int)
    cake['Phone DIY Sales'] = cake.get('DIY Sale_Date_y', 0).astype(int)
    cake['DIFM Web Installs'] = cake.get('DIFM Install_Date_x', 0).astype(int)
    cake['DIFM Phone Installs'] = cake.get('DIFM Install_Date_y', 0).astype(int)
    
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
        'Concatenated', 'PID', 'Leads', 'Cost',
        'Web DIFM Sales', 'Phone DIFM Sales', 'Total DIFM Sales',
        'DIFM Web Installs', 'DIFM Phone Installs', 'Total DIFM Installs',
        'DIY Web Sales', 'DIY Phone Sales', 'Total DIY Sales',
        'Revenue', 'Profit/Loss',
        'Projected Installs', 'Projected Revenue', 'Projected Profit/Loss',
        'Projected Margin', 'Current Rate', 'eCPL'
    ]
    cake = cake[columns]
    
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
                'Concatenated', 'Leads', 'Total DIFM Sales', 'Total DIY Sales',
                'Total DIFM Installs', 'Revenue', 'Cost', 'Projected Installs',
                'Projected Revenue', 'Projected Margin', 'Profit/Loss', 'eCPL'
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