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
    if pd.isna(code):
        return ''
    parts = code.split('_', 1)
    if len(parts) < 2:
        return ''
    second = parts[1]
    if '_' in second:
        seg = second.split('_')[0]
        return f"{seg}_" if seg.isdigit() else ''
    return f"{second}_" if second.isdigit() else ''

def proportional_allocation(row, web_val, total_web_val, total_phone_val):
    if row[total_web_val] == 0 or pd.isna(row[total_web_val]):
        return 0
    proportion = row[web_val] / row[total_web_val]
    return round(row[total_phone_val] * proportion)

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

def calculate_projected_installs(row):
    if str(row['Concatenated']).startswith('4790'):
        return int(round(row['Total DIFM Sales'] * 0.5))
    return int(round(row['Total DIFM Sales'] * 0.7))

def load_combined_resi_tfn_data(sheet_url):
    base_url = sheet_url.split("/edit")[0]

    def sheet_csv_url(sheet_name):
        return f"{base_url}/gviz/tq?tqx=out:csv&sheet={sheet_name.replace(' ', '%20')}"

    # Load data from specific tabs
    resi_df = pd.read_csv(sheet_csv_url("RESI TFN Sheet"))
    display_df = pd.read_csv(sheet_csv_url("Display TFN Sheet"))

    # Select only required columns
    resi_df = resi_df[['PID', 'TFN']]
    display_df = display_df[['PID', 'TFN']]

    # Combine the data
    combined_df = pd.concat([resi_df, display_df], ignore_index=True)

    # Clean the data
    combined_df['Clean_TFN'] = combined_df['TFN'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    combined_df['PID'] = combined_df['PID'].astype(str)

    return combined_df[['Clean_TFN', 'PID']]

def clean_athena(athena_df, tfn_df):
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
    return athena_df

def generate_pivots(athena_df):
    web_df = athena_df[athena_df['Lead_DNIS'].str.contains("WEB", na=False)]
    phone_df = athena_df[~athena_df['Lead_DNIS'].str.contains("WEB", na=False) & athena_df['PID'].notna()]

    web_pivot = pd.pivot_table(
        web_df,
        index='Affiliate_Code',
        values=['Sale_Date', 'Install_Date'],
        columns='INSTALL_METHOD',
        aggfunc='count',
        fill_value=0
    )
    phone_pivot = pd.pivot_table(
        phone_df,
        index='PID',
        values=['Sale_Date', 'Install_Date'],
        columns='INSTALL_METHOD',
        aggfunc='count',
        fill_value=0
    )
    web_pivot.columns = [f"{val} {col}" for col, val in web_pivot.columns]
    phone_pivot.columns = [f"{val} {col}" for col, val in phone_pivot.columns]
    return web_pivot.reset_index(), phone_pivot.reset_index()

def clean_conversion(conversion_df):
    conversion_df = conversion_df[~conversion_df['Offer ID'].isin([31908, 31989])]
    conversion_df['Sub ID'] = conversion_df['Sub ID'].apply(lambda x: str(x) if str(x).isdigit() else '')
    conversion_df['Affiliate ID'] = conversion_df['Affiliate ID'].astype(str)
    conversion_df['Sub ID'] = conversion_df['Sub ID'].astype(str)
    conversion_df['Concatenated'] = conversion_df.apply(
        lambda row: f"{row['Affiliate ID']}_{row['Sub ID']}" if row['Sub ID'] else f"{row['Affiliate ID']}_", axis=1)

    cake_pivot = conversion_df.groupby('Concatenated').agg({
        'Affiliate ID': lambda x: str(x.iloc[0]),
        'Paid': 'sum',
        'Concatenated': 'count'
    }).rename(columns={'Affiliate ID': 'PID', 'Paid': 'Cost', 'Concatenated': 'Leads'}).reset_index()
    return cake_pivot

def merge_web_phone_into_cake(cake_df, web_pivot, phone_pivot):
    web = web_pivot.set_index('Affiliate_Code')
    phone = phone_pivot.set_index('PID')
    
    cake_df = cake_df.copy()
    cake_df = cake_df.merge(web, how='left', left_on='Concatenated', right_index=True)
    cake_df = cake_df.merge(phone, how='left', left_on='PID', right_index=True)
    cake_df.fillna(0, inplace=True)
    return cake_df

def compute_final_metrics(df):
    df = df.copy()
    df['Web DIFM Sales'] = df.get('DIFM Sale_Date_x', 0).astype(int)
    df['Phone DIFM Sales'] = df.get('DIFM Sale_Date_y', 0).astype(int)
    df['Web DIY Sales'] = df.get('DIY Sale_Date_x', 0).astype(int)
    df['Phone DIY Sales'] = df.get('DIY Sale_Date_y', 0).astype(int)
    df['DIFM Web Installs'] = df.get('DIFM Install_Date_x', 0).astype(int)
    df['DIFM Phone Installs'] = df.get('DIFM Install_Date_y', 0).astype(int)

    df['Total DIFM Sales'] = df['Web DIFM Sales'] + df['Phone DIFM Sales']
    df['Total DIY Sales'] = df['Web DIY Sales'] + df['Phone DIY Sales']
    df['Total DIFM Installs'] = df['DIFM Web Installs'] + df['DIFM Phone Installs']

    df['Revenue'] = 1080 * df['Total DIFM Installs'] + 300 * df['Total DIY Sales']
    df['Profit/Loss'] = df['Revenue'] - df['Cost']
    df['Projected Installs'] = df.apply(calculate_projected_installs, axis=1)
    df['Projected Revenue'] = 1080 * df['Projected Installs'] + 300 * df['Total DIY Sales']
    df['Projected Profit/Loss'] = df['Projected Revenue'] - df['Cost']
    df['Projected Margin'] = np.where(
        df['Projected Revenue'] == 0, -1, df['Projected Profit/Loss'] / df['Projected Revenue']
    )
    df['eCPL'] = np.where(
        df['Leads'] == 0, 0, df['Projected Revenue'] / df['Leads']
    )
    return df

def show_bob_analysis():
    st.title("ADT Partner Optimization Analysis")
    
    st.write("""
    This tool analyzes ADT partner performance data to generate optimization reports.
    Please upload the required files below.
    """)
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        athena_file = st.file_uploader("Upload Athena Data (CSV)", type=['csv'])
    
    with col2:
        conversion_file = st.file_uploader("Upload Conversion Data (CSV)", type=['csv'])
    
    if athena_file and conversion_file:
        try:
            # Load and process data
            athena_df = pd.read_csv(athena_file)
            conversion_df = pd.read_csv(conversion_file)
            tfn_df = load_combined_resi_tfn_data(TFN_SHEET_URL)
            
            # Process data
            athena_df = clean_athena(athena_df, tfn_df)
            web_pivot, phone_pivot = generate_pivots(athena_df)
            cake_df = clean_conversion(conversion_df)
            merged_df = merge_web_phone_into_cake(cake_df, web_pivot, phone_pivot)
            final_df = compute_final_metrics(merged_df)
            
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

if __name__ == "__main__":
    show_bob_analysis() 