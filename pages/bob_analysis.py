import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Constants
# -------------------------------

TFN_SHEET_URL = "https://docs.google.com/spreadsheets/d/10BHN_-Wz_ZPmi7rezNtqiDPTguHOoNzmkXzovFOTbaU/edit#gid=1629976834"

# -------------------------------
# Helper Functions
# -------------------------------

def clean_affiliate_code(code):
    if pd.isna(code): return ''
    parts = code.split('_', 1)
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
    def sheet_csv_url(sheet_name):
        return f"{base_url}/gviz/tq?tqx=out:csv&sheet={sheet_name.replace(' ', '%20')}"
    resi_df = pd.read_csv(sheet_csv_url("RESI TFN Sheet"))
    display_df = pd.read_csv(sheet_csv_url("Display TFN Sheet"))
    combined_df = pd.concat([
        resi_df.rename(columns={"PID": "PID", "TFN": "TFN"}),
        display_df.rename(columns={"Partner ID": "PID", "TFN": "TFN"})
    ], ignore_index=True)
    combined_df['Clean_TFN'] = combined_df['TFN'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    combined_df['PID'] = combined_df['PID'].astype(str)
    return combined_df[['Clean_TFN', 'PID']]

def clean_athena(athena_df, tfn_df, leads_df, start_date, end_date):
    try:
        # 1. Process Athena data first
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
        
        # 2. Process TFN data
        athena_df['Affiliate_Code'] = athena_df['Affiliate_Code'].apply(clean_affiliate_code)
        athena_df['Lead_DNIS'] = athena_df['Lead_DNIS'].fillna('').astype(str)
        tfn_map = dict(zip(tfn_df['Clean_TFN'], tfn_df['PID']))
        athena_df['PID'] = athena_df['Lead_DNIS'].apply(lambda x: tfn_map.get(x) if x.isdigit() else None)
        
        # 3. Process leads data - print debugging info
        st.write("Leads DataFrame Info:")
        st.write(leads_df.info())
        st.write("\nLeads DataFrame Head:")
        st.write(leads_df.head())
        
        # Convert column names to lowercase and strip whitespace
        leads_df.columns = [col.strip().lower() for col in leads_df.columns]
        
        # Find required columns
        required_columns = {
            'subid': None,
            'pid': None,
            'phone': None
        }
        
        # Map actual column names to required names
        for col in leads_df.columns:
            col_lower = col.lower()
            if col_lower == 'subid':
                required_columns['subid'] = col
            elif col_lower == 'pid':
                required_columns['pid'] = col
            elif col_lower == 'phone':
                required_columns['phone'] = col
        
        # Check if we found all required columns
        missing_columns = [name for name, col in required_columns.items() if col is None]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}. Found columns: {', '.join(leads_df.columns)}")
        
        # Create working copy of leads DataFrame
        leads_working = leads_df.copy()
        
        # Handle each column separately with explicit error handling
        try:
            # Handle SubID
            leads_working[required_columns['subid']] = leads_working[required_columns['subid']].fillna('')
            leads_working[required_columns['subid']] = leads_working[required_columns['subid']].astype(str)
            
            # Handle PID
            leads_working[required_columns['pid']] = leads_working[required_columns['pid']].fillna('')
            leads_working[required_columns['pid']] = leads_working[required_columns['pid']].astype(str)
            
            # Handle Phone
            leads_working[required_columns['phone']] = leads_working[required_columns['phone']].fillna('')
            leads_working[required_columns['phone']] = leads_working[required_columns['phone']].astype(str)
            leads_working[required_columns['phone']] = leads_working[required_columns['phone']].str.replace(r'[^0-9]', '', regex=True)
            
            # Create concatenated field
            leads_working['Concatenated'] = leads_working.apply(
                lambda r: f"{r[required_columns['pid']]}_{r[required_columns['subid']]}" 
                if str(r[required_columns['subid']]).strip() 
                else f"{r[required_columns['pid']]}_",
                axis=1
            )
            
            # Create phone mapping
            phone_map = dict(zip(
                leads_working[required_columns['phone']],
                leads_working['Concatenated']
            ))
            
            # Clean Athena phone numbers
            athena_df['Primary_Phone_Customer_ANI'] = athena_df['Primary_Phone_Customer_ANI'].fillna('')
            athena_df['Primary_Phone_Customer_ANI'] = athena_df['Primary_Phone_Customer_ANI'].astype(str)
            athena_df['Primary_Phone_Customer_ANI'] = athena_df['Primary_Phone_Customer_ANI'].str.replace(r'[^0-9]', '', regex=True)
            
            # Apply phone mapping
            def fill_code(row):
                if row['Affiliate_Code'] == '' and 'WEB' in str(row['Lead_DNIS']):
                    return phone_map.get(row['Primary_Phone_Customer_ANI'], '')
                return row['Affiliate_Code']
            
            athena_df['Affiliate_Code'] = athena_df.apply(fill_code, axis=1)
            
            return athena_df
            
        except Exception as e:
            st.error(f"Error processing columns: {str(e)}")
            st.write("Column types:")
            st.write(leads_working.dtypes)
            raise
            
    except Exception as e:
        st.error(f"Error in clean_athena: {str(e)}")
        st.error("Full error details:")
        import traceback
        st.error(traceback.format_exc())
        raise

def generate_pivots(athena_df):
    web_df = athena_df[athena_df['Lead_DNIS'].str.contains("WEB", na=False)]
    phone_df = athena_df[~athena_df['Lead_DNIS'].str.contains("WEB", na=False) & athena_df['PID'].notna()]
    web_pivot = pd.pivot_table(web_df, index='Affiliate_Code', values=['Sale_Date', 'Install_Date'], columns='INSTALL_METHOD', aggfunc='count', fill_value=0)
    phone_pivot = pd.pivot_table(phone_df, index='PID', values=['Sale_Date', 'Install_Date'], columns='INSTALL_METHOD', aggfunc='count', fill_value=0)
    web_pivot.columns = [f"{val} {col}" for col, val in web_pivot.columns]
    phone_pivot.columns = [f"{val} {col}" for col, val in phone_pivot.columns]
    return web_pivot.reset_index(), phone_pivot.reset_index()

def clean_conversion(conversion_df):
    conversion_df = conversion_df[~conversion_df['Offer ID'].isin([31908, 31989])]
    conversion_df['Sub ID'] = conversion_df['Sub ID'].apply(lambda x: str(x) if str(x).isdigit() else '')
    conversion_df['Affiliate ID'] = conversion_df['Affiliate ID'].astype(str)
    conversion_df['Concatenated'] = conversion_df.apply(lambda r: f"{r['Affiliate ID']}_{r['Sub ID']}" if r['Sub ID'] else f"{r['Affiliate ID']}_", axis=1)
    cake = conversion_df.groupby('Concatenated').agg({
        'Affiliate ID': lambda x: str(x.iloc[0]),
        'Paid': 'sum',
        'Concatenated': 'count'
    }).rename(columns={'Affiliate ID': 'PID', 'Paid': 'Cost', 'Concatenated': 'Leads'}).reset_index()
    return cake

def merge_and_compute(cake, web, phone):
    cake = cake.copy()
    web = web.set_index('Affiliate_Code')
    phone = phone.set_index('PID')
    cake = cake.merge(web, how='left', left_on='Concatenated', right_index=True)
    cake = cake.merge(phone, how='left', left_on='PID', right_index=True)
    cake.fillna(0, inplace=True)

    cake['Web DIFM Sales'] = cake.get('DIFM Sale_Date_x', 0).astype(int)
    cake['Phone DIFM Sales'] = cake.get('DIFM Sale_Date_y', 0).astype(int)
    cake['Web DIY Sales'] = cake.get('DIY Sale_Date_x', 0).astype(int)
    cake['Phone DIY Sales'] = cake.get('DIY Sale_Date_y', 0).astype(int)
    cake['DIFM Web Installs'] = cake.get('DIFM Install_Date_x', 0).astype(int)
    cake['DIFM Phone Installs'] = cake.get('DIFM Install_Date_y', 0).astype(int)

    cake['Total DIFM Sales'] = cake['Web DIFM Sales'] + cake['Phone DIFM Sales']
    cake['Total DIY Sales'] = cake['Web DIY Sales'] + cake['Phone DIY Sales']
    cake['Total DIFM Installs'] = cake['DIFM Web Installs'] + cake['DIFM Phone Installs']

    cake['Revenue'] = 1080 * cake['Total DIFM Installs'] + 300 * cake['Total DIY Sales']
    cake['Profit/Loss'] = cake['Revenue'] - cake['Cost']
    cake['Projected Installs'] = cake.apply(calculate_projected_installs, axis=1)
    cake['Projected Revenue'] = 1080 * cake['Projected Installs'] + 300 * cake['Total DIY Sales']
    cake['Projected Profit/Loss'] = cake['Projected Revenue'] - cake['Cost']
    cake['Projected Margin'] = np.where(cake['Projected Revenue'] == 0, -1, cake['Projected Profit/Loss'] / cake['Projected Revenue'])
    cake['eCPL'] = np.where(cake['Leads'] == 0, 0, cake['Projected Revenue'] / cake['Leads'])
    return cake

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
            display_df['Projected Margin'] = display_df['Projected Margin'].apply(lambda x: f"{x:.2%}" if x != -1 else "N/A")
            display_df['eCPL'] = display_df['eCPL'].apply(lambda x: f"${x:,.2f}")
            display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"${x:,.2f}")
            display_df['Cost'] = display_df['Cost'].apply(lambda x: f"${x:,.2f}")
            display_df['Projected Revenue'] = display_df['Projected Revenue'].apply(lambda x: f"${x:,.2f}")
            display_df['Profit/Loss'] = display_df['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
            
            # Display the table
            st.dataframe(
                display_df.sort_values('Projected Revenue', ascending=False),
                use_container_width=True
            )
            
            # Export functionality
            st.subheader("Export Report")
            if st.button("Export to Excel"):
                try:
                    # Create Excel file
                    output = pd.ExcelWriter('adt_optimization_report.xlsx', engine='xlsxwriter')
                    final_df.to_excel(output, sheet_name='Partner Performance', index=False)
                    output.save()
                    
                    # Create download button
                    with open('adt_optimization_report.xlsx', 'rb') as f:
                        st.download_button(
                            label="Download Excel Report",
                            data=f,
                            file_name=f"adt_optimization_report_leads_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error exporting report: {str(e)}")
        
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