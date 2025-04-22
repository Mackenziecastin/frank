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

def clean_athena(athena_df, tfn_df, leads_df):
    athena_df['Lead_Creation_Date'] = pd.to_datetime(athena_df['Lead_Creation_Date'], errors='coerce')
    athena_df = athena_df[athena_df['Lead_Creation_Date'].dt.month == 4]
    athena_df = athena_df[
        (athena_df['Ln_of_Busn'].str.lower() != 'health') &
        (athena_df['DNIS_BUSN_SEG_CD'].str.lower() != 'us: health') &
        (athena_df['Sale_Date'].notna()) &
        (athena_df['Ordr_Type'].str.upper().isin(['NEW', 'RESALE']))
    ]
    athena_df['Affiliate_Code'] = athena_df['Affiliate_Code'].apply(clean_affiliate_code)
    athena_df['Lead_DNIS'] = athena_df['Lead_DNIS'].astype(str)
    tfn_map = dict(zip(tfn_df['Clean_TFN'], tfn_df['PID']))
    athena_df['PID'] = athena_df['Lead_DNIS'].apply(lambda x: tfn_map.get(x) if x.isdigit() else None)
    
    # Leads report processing - handle different column name formats
    # Check for column name variations
    sub_id_col = next((col for col in leads_df.columns if col in ['Subid', 'subID', 'sub_id', 'sub id', 'SubID']), None)
    pid_col = next((col for col in leads_df.columns if col in ['PID', 'pid', 'partner_id', 'partner id', 'PartnerID']), None)
    phone_col = next((col for col in leads_df.columns if col in ['Phone', 'phone', 'phone_number', 'phone number']), None)
    
    if not all([sub_id_col, pid_col, phone_col]):
        missing_cols = []
        if not sub_id_col:
            missing_cols.append("Subid/SubID")
        if not pid_col:
            missing_cols.append("PID/PartnerID")
        if not phone_col:
            missing_cols.append("Phone")
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(leads_df.columns)}")
    
    # Clean and process the data with the correct column names
    leads_df[sub_id_col] = leads_df[sub_id_col].apply(lambda x: str(x) if str(x).isdigit() else '')
    leads_df[pid_col] = leads_df[pid_col].astype(str)
    leads_df['Concatenated'] = leads_df.apply(
        lambda r: f"{r[pid_col]}_{r[sub_id_col]}" if r[sub_id_col] else f"{r[pid_col]}_", 
        axis=1
    )
    
    # Create phone mapping using the correct column name
    phone_map = dict(zip(leads_df[phone_col].astype(str), leads_df['Concatenated']))
    athena_df['Primary_Phone_Customer_ANI'] = athena_df['Primary_Phone_Customer_ANI'].astype(str)
    
    def fill_code(row):
        if row['Affiliate_Code'] == '' and 'WEB' in row['Lead_DNIS']:
            return phone_map.get(row['Primary_Phone_Customer_ANI'], '')
        return row['Affiliate_Code']
    
    athena_df['Affiliate_Code'] = athena_df.apply(fill_code, axis=1)
    return athena_df

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
    Please upload the required files below.
    """)
    
    # File uploaders in three columns
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
            # Load and process data
            athena_df = pd.read_csv(athena_file)
            conversion_df = pd.read_csv(cake_conversion_file)
            leads_df = pd.read_csv(database_leads_file)
            
            # Print column names for debugging
            st.write("Database Leads columns found:", leads_df.columns.tolist())
            
            tfn_df = load_combined_resi_tfn_data(TFN_SHEET_URL)
            
            # Step 1: Clean Athena + Leads Report
            athena_df = clean_athena(athena_df, tfn_df, leads_df)
            
            # Step 2: Generate Web + Phone Pivots
            web_pivot, phone_pivot = generate_pivots(athena_df)
            
            # Step 3: Clean Conversion Report
            cake_df = clean_conversion(conversion_df)
            
            # Step 4-5: Merge and Compute Final Metrics
            final_df = merge_and_compute(cake_df, web_pivot, phone_pivot)
            
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
                            file_name="adt_optimization_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error exporting report: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Please check your input files and try again.")
            # Add more detailed error information
            st.error("Required columns in Database Leads file: Subid, PID, Phone")
            st.error("Current columns found:", leads_df.columns.tolist() if 'leads_df' in locals() else "No file loaded")

if __name__ == "__main__":
    show_bob_analysis() 