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
    """Get the most recent rate for each Affiliate ID + Sub ID combination."""
    # Convert date and clean IDs
    conversion_df['Conversion Date'] = pd.to_datetime(conversion_df['Conversion Date'], errors='coerce')
    
    # Clean Sub ID - keep only if numeric
    conversion_df['Sub ID'] = conversion_df['Sub ID'].astype(str)
    conversion_df['Sub ID'] = conversion_df['Sub ID'].apply(lambda x: x if x.isdigit() else '')
    
    # Clean Affiliate ID
    conversion_df['Affiliate ID'] = conversion_df['Affiliate ID'].astype(str)
    
    # Create Concatenated key in same format as main report
    conversion_df['Concatenated'] = conversion_df.apply(
        lambda r: f"{r['Affiliate ID']}_{r['Sub ID']}" if r['Sub ID'] else f"{r['Affiliate ID']}_",
        axis=1
    )
    
    # Sort by date descending and get most recent rate
    current_rates = (
        conversion_df
        .sort_values('Conversion Date', ascending=False)
        .groupby('Concatenated', as_index=False)
        .agg({
            'Paid': 'first',
            'Conversion Date': 'first'
        })
        .rename(columns={'Paid': 'Current Rate'})
    )
    
    # Debug output
    st.write("\n### Current Rate Calculation Debug")
    st.write("Sample of current rates (first 10 rows):")
    st.write(current_rates.head(10))
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
        
        # Search for critical phone numbers in RESI sheet
        critical_numbers = [
            '8446778720', '8005717438', '8009734275', '8442069696', '8442342126',
            '8444399581', '8444399582', '8445524846', '8445862465', '8445986623',
            '8446102586', '8446253379', '8446451022', '8556943664'
        ]
        
        st.write("\n### Checking for Important Phone Numbers in RESI Sheet")
        for number in critical_numbers:
            # Try multiple formats of the number
            matches = resi_df[
                resi_df['Phone #'].astype(str).str.contains(number, na=False) | 
                resi_df['Phone #'].astype(str).str.contains(f"\\+1{number}", na=False) |
                resi_df['Phone #'].astype(str).str.contains(f"1{number}", na=False) |
                resi_df['Phone #'].astype(str).str.contains(f"{number[:3]}-{number[3:6]}-{number[6:]}", na=False)
            ]
            
            if not matches.empty:
                st.write(f"✓ Found {number} in RESI sheet with PID: {matches['PID'].iloc[0]}")
                st.write(f"  Original format: {matches['Phone #'].iloc[0]}")
            else:
                st.write(f"⚠️ Warning: {number} NOT found in RESI sheet")
        
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
        
        # Store original TFN values for reference
        combined_df['Original_TFN'] = combined_df['TFN']
        
        # Clean TFNs - keep empty values as blank strings
        combined_df['Clean_TFN'] = combined_df['TFN'].fillna('').astype(str)
        
        # Try multiple cleaning approaches to ensure all formats are covered
        cleaned_tfns = []
        for tfn in combined_df['Clean_TFN']:
            # Standard cleaning - remove non-digits
            standard_clean = ''.join(c for c in str(tfn) if c.isdigit())
            
            # If it has country code (1) at the beginning and is 11 digits, create a 10-digit version too
            alt_clean = None
            if len(standard_clean) == 11 and standard_clean.startswith('1'):
                alt_clean = standard_clean[1:]
            
            cleaned_tfns.append((standard_clean, alt_clean))
        
        # Create expanded dataframe with multiple TFN formats
        expanded_rows = []
        for idx, row in combined_df.iterrows():
            std_clean, alt_clean = cleaned_tfns[idx]
            
            # Add row with standard cleaning
            if std_clean:
                expanded_rows.append({
                    'PID': row['PID'],
                    'TFN': row['TFN'],
                    'Original_TFN': row['Original_TFN'],
                    'Clean_TFN': std_clean
                })
            
            # Add row with alternate cleaning if it exists
            if alt_clean:
                expanded_rows.append({
                    'PID': row['PID'],
                    'TFN': row['TFN'],
                    'Original_TFN': row['Original_TFN'],
                    'Clean_TFN': alt_clean
                })
        
        # Replace combined_df with expanded version
        if expanded_rows:
            combined_df = pd.DataFrame(expanded_rows)
            st.write(f"Expanded TFN mappings to include alternate formats: {len(combined_df)} total mappings")
        
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
        st.write(combined_df[['Clean_TFN', 'PID', 'Original_TFN']].head(20))
        
        # Verify critical numbers are in the mapping
        st.write("\n### Verifying critical phone numbers exist in final mapping")
        for number in critical_numbers:
            # Check both with and without country code
            matches = combined_df[
                (combined_df['Clean_TFN'] == number) | 
                (combined_df['Clean_TFN'] == f"1{number}")
            ]
            
            if not matches.empty:
                st.write(f"✓ Found {number} in mapping with PID: {matches['PID'].iloc[0]}")
                st.write(f"  Original format: {matches['Original_TFN'].iloc[0]}")
            else:
                # Try more flexible matching on original
                original_matches = combined_df[combined_df['Original_TFN'].str.contains(number, na=False)]
                if not original_matches.empty:
                    st.write(f"⚠️ Found {number} in original TFN but not in cleaned version. PID: {original_matches['PID'].iloc[0]}")
                    st.write(f"  Original format: {original_matches['Original_TFN'].iloc[0]}")
                    st.write(f"  Cleaned to: {original_matches['Clean_TFN'].iloc[0]}")
                else:
                    st.write(f"✗ {number} NOT found in mapping")
        
        st.write("\nAll unique PIDs in mapping:")
        st.write(sorted([str(pid).strip() for pid in combined_df['PID'].unique().tolist() if str(pid).strip()]))
        
        # Return only the Clean_TFN and PID columns for mapping
        return combined_df[['Clean_TFN', 'PID']]
        
    except Exception as e:
        st.error(f"Error loading TFN data: {str(e)}")
        st.error("Full error details:")
        import traceback
        st.error(traceback.format_exc())
        raise

def clean_phone(phone_str):
    """Clean a phone number by extracting digits and handling special cases."""
    phone_str = str(phone_str).strip()
    # Extract only digits
    digits = ''.join(c for c in phone_str if c.isdigit())
    
    # Handle the common case of an extra digit at the end (as seen in the data)
    # If it's 11 digits and doesn't start with 1, it likely has an extra digit at the end
    if len(digits) == 11 and not digits.startswith('1'):
        return digits[:-1]  # Remove the last digit
    
    # If it has country code (1) at the beginning, remove it if the result would be 10 digits
    if len(digits) == 11 and digits.startswith('1'):
        return digits[1:]   # Remove the country code
    
    # If it has 10 digits, return as is
    if len(digits) == 10:
        return digits
    
    # Otherwise just return all digits
    return digits

def clean_athena():
    """Clean athena data."""
    st.session_state['athena_is_loaded'] = False
    
    # Retrieve data from S3 or local cache
    athena_df = fetch_athena_data()
    
    # Debug: Print column names before processing
    st.write("Original columns:", athena_df.columns.tolist())
    
    # Show shape before cleaning
    original_shape = athena_df.shape
    
    # Debug information - Business line distribution before filtering
    st.write("\n### Business Line Distribution Before Filtering")
    if 'Ln_of_Busn' in athena_df.columns:
        st.write("Unique business lines:", athena_df['Ln_of_Busn'].unique())
        st.write("Counts by business line:")
        st.write(athena_df['Ln_of_Busn'].value_counts())
    
    # Clean affiliate code
    if 'SRC_SUBD_ID' in athena_df.columns:
        athena_df['Affiliate_Code'] = athena_df['SRC_SUBD_ID'].apply(clean_affiliate_code)
    
    # Debug: Print sample rows to see column values
    st.write("Sample rows after processing SRC_SUBD_ID:")
    st.write(athena_df[['SRC_SUBD_ID', 'Affiliate_Code']].head())
    
    # Filter records
    athena_df = athena_df[
        # Keep only main residential records (non-health)
        (~athena_df['Ln_of_Busn'].str.contains('health', case=False, na=False)) &
        (~athena_df['DNIS_BUSN_SEG_CD'].str.contains('health', case=False, na=False)) &
        # Must have sale date
        (athena_df['Sale_Date'].notna()) &
        # Must be NEW or RESALE
        ((athena_df['Ordr_Type'] == 'NEW') | (athena_df['Ordr_Type'] == 'RESALE'))
    ]
    
    # Debug information - Business line distribution after filtering
    st.write("\n### Business Line Distribution After Filtering")
    if 'Ln_of_Busn' in athena_df.columns:
        st.write("Remaining business lines:", athena_df['Ln_of_Busn'].unique())
        st.write("Filtered counts by business line:")
        st.write(athena_df['Ln_of_Busn'].value_counts())
    
    # Show shape after filtering
    filtered_shape = athena_df.shape
    st.write(f"Records before filtering: {original_shape[0]}, after: {filtered_shape[0]}")
    
    # Match PIDs for non-web records
    st.write("\n### PID Matching for Non-Web Records")
    phones = athena_df[~athena_df['Lead_DNIS'].str.contains("WEB", na=False)]
    web = athena_df[athena_df['Lead_DNIS'].str.contains("WEB", na=False)]
    
    # Get counts before PID matching
    st.write(f"Non-web records before PID matching: {len(phones)}")
    
    # Create a mapping dict for TFN -> PID
    combined_df = load_combined_resi_tfn_data(TFN_SHEET_URL)
    
    # Debug TFN mapping dataframe
    st.write("\n### TFN Mapping Sample")
    st.write(combined_df[['PID', 'TFN', 'Clean_TFN']].head())
    
    # Apply PID matching for non-web records
    phones['PID'] = phones['Lead_DNIS'].apply(lambda dnis: match_pid(dnis, combined_df))
    
    # Debug how many matched
    matched = phones[phones['PID'].notna()]
    st.write(f"Non-web records with matched PID: {len(matched)} ({len(matched)/len(phones):.2%})")
    
    # Show distribution of matched PIDs
    st.write("Top 10 matched PIDs:")
    st.write(phones['PID'].value_counts().head(10))
    
    # Combine web and matched phone records
    athena_df = pd.concat([web, phones])
    
    # Analyze records by PID
    st.write("\n### Records by PID")
    pid_counts = analyze_records_by_pid(athena_df)
    
    # Set loaded flag
    st.session_state['athena_is_loaded'] = True
    
    # Store athena_df in session state for use in other functions
    st.session_state['athena_df'] = athena_df
    
    return athena_df

def analyze_pre_matchback_phone_metrics(athena_df, tfn_df=None):
    """
    Analyze phone records directly from Athena data before PID matchback.
    This helps identify if the issue is with the raw data or with the matchback process.
    
    Args:
        athena_df: The full Athena DataFrame
        tfn_df: TFN mapping DataFrame (optional)
    
    Returns:
        dict: Pre-matchback phone metrics
    """
    # First get all non-WEB records directly from Athena - but check for any case variations
    raw_phone_records = athena_df[~athena_df['Lead_DNIS'].str.contains("WEB", case=True, na=False)].copy()
    
    # Store raw phone records in session state for debugging
    st.session_state.raw_phone_records = raw_phone_records
    
    # Add export button for raw phone records
    st.subheader("Export Raw Phone Records")
    if st.button("Export Raw Phone Records to CSV"):
        try:
            # Create a BytesIO object to hold the CSV file in memory
            output = io.BytesIO()
            
            # Write the DataFrame to CSV
            raw_phone_records.to_csv(output, index=False)
            
            # Seek to the beginning of the BytesIO object
            output.seek(0)
            
            # Create download button
            st.download_button(
                label="Download Raw Phone Records",
                data=output,
                file_name="raw_phone_records.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error exporting raw phone records: {str(e)}")
            st.error("Full error details:")
            import traceback
            st.error(traceback.format_exc())
    
    # Check if there are any records with null Lead_DNIS (these would be missed by str.contains)
    null_dnis_records = athena_df[athena_df['Lead_DNIS'].isna()].copy()
    if not null_dnis_records.empty:
        st.write(f"Found {len(null_dnis_records)} records with null Lead_DNIS values")
        st.write("Sample of records with null Lead_DNIS:")
        st.write(null_dnis_records.head())
    
    # Check for non-WEB records with different case variations
    possible_phone_records = len(athena_df[~athena_df['Lead_DNIS'].str.contains("WEB", case=False, na=False)])
    if possible_phone_records != len(raw_phone_records):
        st.write(f"⚠️ Found discrepancy in phone record count when using case-insensitive search")
        st.write(f"Case-sensitive non-WEB count: {len(raw_phone_records)}")
        st.write(f"Case-insensitive non-WEB count: {possible_phone_records}")
        
        # Find the records that differ between the two methods
        case_sensitive_mask = ~athena_df['Lead_DNIS'].str.contains("WEB", case=True, na=False)
        case_insensitive_mask = ~athena_df['Lead_DNIS'].str.contains("WEB", case=False, na=False)
        different_records = athena_df[case_insensitive_mask & ~case_sensitive_mask]
        
        st.write(f"Found {len(different_records)} records with Lead_DNIS containing 'web', 'Web', etc. but not exactly 'WEB'")
        if not different_records.empty:
            st.write("Sample of these records:")
            st.write(different_records[['Lead_DNIS', 'INSTALL_METHOD', 'Sale_Date', 'Install_Date']])
    
    # Check for any records that contain "web" in different formats
    st.write("\n### Checking for records with variations of 'WEB' in Lead_DNIS")
    for variation in ["web", "Web", "WEb", "WeB"]:
        if variation == "WEB": continue  # Skip the exact match we already checked
        count = len(athena_df[athena_df['Lead_DNIS'].str.contains(variation, na=False)])
        if count > 0:
            st.write(f"Found {count} records containing '{variation}'")
            sample = athena_df[athena_df['Lead_DNIS'].str.contains(variation, na=False)].head()
            st.write(f"Sample Lead_DNIS values: {sample['Lead_DNIS'].tolist()}")
    
    # Now use more comprehensive search for non-phone records
    comprehensive_phone_records = athena_df[~athena_df['Lead_DNIS'].str.contains("WEB|web|Web", case=False, na=False)].copy()
    
    st.write(f"\n### Pre-Matchback Phone Analysis")
    st.write(f"Total non-WEB records with basic filter: {len(raw_phone_records)}")
    st.write(f"Total non-WEB records with comprehensive filter: {len(comprehensive_phone_records)}")
    
    # Use the comprehensive filter for further analysis
    raw_phone_records = comprehensive_phone_records
    
    # Check for potentially non-phone records that might be miscounted
    suspect_records = raw_phone_records[raw_phone_records['Lead_DNIS'].str.contains(r'\D', na=False)]
    st.write(f"\nFound {len(suspect_records)} non-WEB records with non-numeric characters in Lead_DNIS")
    if not suspect_records.empty:
        st.write("Sample of these records:")
        st.write(suspect_records[['Lead_DNIS', 'INSTALL_METHOD', 'Sale_Date', 'Install_Date']])
    
    # Check how many have Lead_DNIS values (should be all of them)
    with_dnis = raw_phone_records['Lead_DNIS'].notna().sum()
    st.write(f"Records with Lead_DNIS: {with_dnis}")
    
    # Count all Lead_DNIS values directly
    st.write("\n### Direct Lead_DNIS Analysis")
    st.write("Analyzing all unique Lead_DNIS values:")
    
    all_dnis_values = athena_df['Lead_DNIS'].value_counts().to_dict()
    st.write(f"Total unique Lead_DNIS values in dataset: {len(all_dnis_values)}")
    
    web_values = {dnis: count for dnis, count in all_dnis_values.items() 
                 if isinstance(dnis, str) and 'WEB' in dnis.upper()}
    non_web_values = {dnis: count for dnis, count in all_dnis_values.items() 
                     if not (isinstance(dnis, str) and 'WEB' in dnis.upper())}
    
    st.write(f"Lead_DNIS values containing 'WEB' (any case): {len(web_values)}")
    st.write(f"Lead_DNIS values not containing 'WEB': {len(non_web_values)}")
    
    total_web_records = sum(web_values.values())
    total_non_web_records = sum(non_web_values.values())
    
    st.write(f"Total records with WEB in Lead_DNIS: {total_web_records}")
    st.write(f"Total records without WEB in Lead_DNIS: {total_non_web_records}")
    st.write(f"Total records: {total_web_records + total_non_web_records}")
    st.write(f"Dataset total: {len(athena_df)}")
    
    # Verify if all records are accounted for
    if len(athena_df) != (total_web_records + total_non_web_records):
        st.write("⚠️ Some records are not being counted properly!")
        st.write(f"Missing records: {len(athena_df) - (total_web_records + total_non_web_records)}")
        
        # Look for null values
        null_dnis = athena_df[athena_df['Lead_DNIS'].isna()]
        st.write(f"Records with null Lead_DNIS: {len(null_dnis)}")
        if len(null_dnis) > 0:
            st.write("Sample of records with null Lead_DNIS:")
            st.write(null_dnis.head())
    
    # Show unique Lead_DNIS values (first 20)
    unique_dnis = raw_phone_records['Lead_DNIS'].unique()
    st.write(f"\nFound {len(unique_dnis)} unique DNIS values. First 20:")
    for dnis in unique_dnis[:20]:
        st.write(f"- '{dnis}'")
    
    # Get the pre-matchback counts by install method
    pre_match_stats = {
        'by_install_method': raw_phone_records.groupby('INSTALL_METHOD').size().to_dict(),
        'sales_by_method': {},
        'installs_by_method': {}
    }
    
    # Count sales and installs by install method
    for method in raw_phone_records['INSTALL_METHOD'].unique():
        method_df = raw_phone_records[raw_phone_records['INSTALL_METHOD'] == method]
        pre_match_stats['sales_by_method'][method] = method_df['Sale_Date'].notna().sum()
        pre_match_stats['installs_by_method'][method] = method_df['Install_Date'].notna().sum()
    
    # Show summary of pre-matchback counts
    st.write("\n#### Phone Record Counts by INSTALL_METHOD (Pre-Matchback)")
    method_counts = pd.DataFrame({
        'INSTALL_METHOD': list(pre_match_stats['by_install_method'].keys()),
        'Record Count': list(pre_match_stats['by_install_method'].values()),
    })
    st.dataframe(method_counts)
    
    # Show detailed sales and installs by method
    st.write("\n#### Sales and Installs by INSTALL_METHOD (Pre-Matchback)")
    sales_installs = pd.DataFrame({
        'INSTALL_METHOD': list(pre_match_stats['sales_by_method'].keys()),
        'Sales': list(pre_match_stats['sales_by_method'].values()),
        'Installs': list(pre_match_stats['installs_by_method'].values())
    })
    st.dataframe(sales_installs)
    
    # Use regex to find any Lead_DNIS with exact numeric pattern - these would be phone numbers
    likely_phone_mask = raw_phone_records['Lead_DNIS'].str.match(r'^\d+$', na=False)
    likely_phone_records = raw_phone_records[likely_phone_mask]
    
    st.write(f"\nRecords with purely numeric Lead_DNIS (likely true phone numbers): {len(likely_phone_records)}")
    
    # Calculate totals
    total_phone_sales = sum(pre_match_stats['sales_by_method'].values())
    total_phone_installs = sum(pre_match_stats['installs_by_method'].values())
    
    st.write(f"\n### Total Pre-Matchback Phone Metrics")
    st.write(f"Total Phone Sales: {total_phone_sales}")
    st.write(f"Total Phone Installs: {total_phone_installs}")
    
    # Create a bar chart for pre-matchback phone metrics
    fig = px.bar(
        sales_installs,
        x='INSTALL_METHOD',
        y=['Sales', 'Installs'],
        title='Phone Sales and Installs by INSTALL_METHOD (Pre-Matchback)',
        barmode='group'
    )
    st.plotly_chart(fig)
    
    # Now analyze the TFN mapping and potential matchback issues
    tfn_map = {}
    if tfn_df is not None:
        tfn_map = dict(zip(tfn_df['Clean_TFN'], tfn_df['PID']))
    
    st.write("\n#### Lead_DNIS to TFN Mapping Check")
    st.write(f"TFN mapping size: {len(tfn_map)}")
    
    # Check how many Lead_DNIS values match TFNs in the mapping
    matching_counts = 0
    dnis_to_check = min(100, len(raw_phone_records))
    checked_dnis = []
    
    for _, row in raw_phone_records.head(dnis_to_check).iterrows():
        dnis = row['Lead_DNIS']
        numeric_dnis = ''.join(c for c in str(dnis) if c.isdigit())
        matches_tfn = numeric_dnis in tfn_map
        checked_dnis.append({
            'Lead_DNIS': dnis,
            'Numeric_DNIS': numeric_dnis,
            'Matches_TFN': matches_tfn,
            'Matched_PID': tfn_map.get(numeric_dnis, 'No match')
        })
        if matches_tfn:
            matching_counts += 1
    
    st.write(f"Checked {dnis_to_check} DNIS values, {matching_counts} match a TFN in the mapping")
    
    # Show sample of the DNIS check results
    st.write("\nSample DNIS Matching Check (first 20):")
    st.dataframe(pd.DataFrame(checked_dnis[:20]))
    
    # Special check for 42038
    st.write("\n### Special check for PID 42038 (DNIS 8009734275)")
    pid_42038_records = raw_phone_records[raw_phone_records['Lead_DNIS'].str.contains('8009734275', na=False)]
    st.write(f"Found {len(pid_42038_records)} raw records with DNIS 8009734275")
    if not pid_42038_records.empty:
        st.write("Sample records:")
        st.write(pid_42038_records[['Lead_DNIS', 'INSTALL_METHOD', 'Sale_Date', 'Install_Date']].head())
        
        # Check if numeric DNIS match
        for _, row in pid_42038_records.iterrows():
            dnis = row['Lead_DNIS']
            numeric_dnis = ''.join(c for c in str(dnis) if c.isdigit())
            st.write(f"Original DNIS: {dnis}, Numeric DNIS: {numeric_dnis}, In TFN Map: {numeric_dnis in tfn_map}")
            if numeric_dnis in tfn_map:
                st.write(f"Mapped to PID: {tfn_map[numeric_dnis]}")
            else:
                st.write("Not found in TFN map")
    
    return pre_match_stats, raw_phone_records

def analyze_post_matchback_metrics_by_pid(phone_df):
    """
    Analyze phone metrics by PID after matchback.
    Creates visualizations to help understand how phone metrics are distributed across PIDs.
    
    Args:
        phone_df: DataFrame of phone records after PID matching
    """
    st.write("\n### Post-Matchback Phone Metrics by PID")
    st.write(f"Total phone records with matched PIDs: {len(phone_df)}")
    
    # Get counts of records by PID
    pid_counts = phone_df['PID'].value_counts().reset_index()
    pid_counts.columns = ['PID', 'Record Count']
    
    st.write("\n#### Records by PID")
    st.dataframe(pid_counts)
    
    # Export button for the phone metrics by PID
    if st.button("Export Phone Metrics by PID"):
        try:
            # Create a BytesIO object to hold the CSV file in memory
            output = io.BytesIO()
            
            # Write the DataFrame to CSV
            pid_counts.to_csv(output, index=False)
            
            # Seek to the beginning of the BytesIO object
            output.seek(0)
            
            # Create download button
            st.download_button(
                label="Download Phone Metrics by PID",
                data=output,
                file_name="phone_metrics_by_pid.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error exporting phone metrics by PID: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Create a bar chart of record counts by PID
    fig_counts = px.bar(
        pid_counts,
        x='PID',
        y='Record Count',
        title='Phone Records by PID',
    )
    st.plotly_chart(fig_counts)
    
    # Group by PID and INSTALL_METHOD to get sales and installs
    # For each PID and install method, count non-null Sale_Date and Install_Date values
    metrics_by_pid = []
    
    for pid in phone_df['PID'].unique():
        pid_df = phone_df[phone_df['PID'] == pid]
        
        # Get DIFM counts
        difm_df = pid_df[pid_df['INSTALL_METHOD'].str.contains('DIFM', na=False)]
        difm_sales = difm_df['Sale_Date'].notna().sum()
        difm_installs = difm_df['Install_Date'].notna().sum()
        
        # Get DIY counts
        diy_df = pid_df[pid_df['INSTALL_METHOD'].str.contains('DIY', na=False)]
        diy_sales = diy_df['Sale_Date'].notna().sum()
        diy_installs = diy_df['Install_Date'].notna().sum()
        
        metrics_by_pid.append({
            'PID': pid,
            'DIFM Sales': difm_sales,
            'DIFM Installs': difm_installs,
            'DIY Sales': diy_sales,
            'DIY Installs': diy_installs,
            'Total': len(pid_df)
        })
    
    # Create DataFrame and display
    pid_metrics_df = pd.DataFrame(metrics_by_pid)
    
    st.write("\n#### Sales and Installs by PID")
    st.dataframe(pid_metrics_df.sort_values('Total', ascending=False))
    
    # Export button for detailed sales and installs by PID
    if st.button("Export Detailed Sales and Installs by PID"):
        try:
            # Create a BytesIO object to hold the CSV file in memory
            output = io.BytesIO()
            
            # Write the DataFrame to CSV
            pid_metrics_df.to_csv(output, index=False)
            
            # Seek to the beginning of the BytesIO object
            output.seek(0)
            
            # Create download button
            st.download_button(
                label="Download Detailed Sales and Installs by PID",
                data=output,
                file_name="detailed_metrics_by_pid.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error exporting detailed metrics by PID: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Create a stacked bar chart of metrics by PID
    metrics_melted = pid_metrics_df.melt(
        id_vars=['PID'],
        value_vars=['DIFM Sales', 'DIFM Installs', 'DIY Sales', 'DIY Installs'],
        var_name='Metric',
        value_name='Count'
    )
    
    fig_metrics = px.bar(
        metrics_melted,
        x='PID',
        y='Count',
        color='Metric',
        title='Phone Metrics by PID',
        barmode='group'
    )
    
    # Improve readability
    fig_metrics.update_layout(xaxis_tickangle=-45)
    
    st.plotly_chart(fig_metrics)
    
    # Calculate totals
    totals = {
        'DIFM Sales': pid_metrics_df['DIFM Sales'].sum(),
        'DIFM Installs': pid_metrics_df['DIFM Installs'].sum(),
        'DIY Sales': pid_metrics_df['DIY Sales'].sum(),
        'DIY Installs': pid_metrics_df['DIY Installs'].sum()
    }
    
    st.write("\n#### Total Counts After PID Matching")
    st.write(totals)
    
    # Add specific debug for PID 42038
    st.write("\n### Special Check for PID 42038")
    pid_42038_df = phone_df[phone_df['PID'] == '42038']
    st.write(f"Found {len(pid_42038_df)} records for PID 42038")
    
    if not pid_42038_df.empty:
        st.write("Sale dates for PID 42038:")
        st.write(pid_42038_df['Sale_Date'].tolist())
        
        # Count DIFM sales for PID 42038
        difm_sales_42038 = pid_42038_df[
            (pid_42038_df['INSTALL_METHOD'].str.contains('DIFM', na=False)) & 
            (pid_42038_df['Sale_Date'].notna())
        ]
        st.write(f"DIFM Sales for PID 42038: {len(difm_sales_42038)}")
        
        if len(difm_sales_42038) > 0:
            st.write("DIFM Sales records for PID 42038:")
            st.write(difm_sales_42038[['Lead_DNIS', 'INSTALL_METHOD', 'Sale_Date', 'Lead_Creation_Date']])
    else:
        st.write("No records found for PID 42038!")
    
    return pid_metrics_df

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
    # Start date is automatically set to 00:00:00 of the selected day
    start_date = pd.Timestamp(start_date)
    # Ensure the end date includes the entire day (up to 23:59:59)
    end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Display the exact date range being used for analysis
    st.info(f"Analyzing leads created from {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
                # Store in session state for access by other functions
                st.session_state.tfn_df = tfn_df
            except Exception as e:
                st.error(f"Error loading TFN data: {str(e)}")
                return
            
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
            
            # Add an export button for the cleaned Athena data
            st.subheader("Export Cleaned Athena Data")
            if st.button("Export Cleaned Athena Data to CSV"):
                try:
                    # Create a BytesIO object to hold the CSV file in memory
                    output = io.BytesIO()
                    
                    # Write the DataFrame to CSV
                    athena_df.to_csv(output, index=False)
                    
                    # Seek to the beginning of the BytesIO object
                    output.seek(0)
                    
                    # Create download button
                    st.download_button(
                        label="Download Cleaned Athena Data",
                        data=output,
                        file_name=f"cleaned_athena_data_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error exporting Athena data: {str(e)}")
                    st.error("Full error details:")
                    import traceback
                    st.error(traceback.format_exc())
            
            # Step 2: Generate Web + Phone Pivots
            st.write("DEBUG: Generating pivots...")
            try:
                web_pivot, phone_pivot = generate_pivots(athena_df)
                st.write("DEBUG: Successfully generated pivots")
            except Exception as e:
                st.error(f"Error generating pivots: {str(e)}")
                import traceback
                st.error(f"Full traceback:\n{traceback.format_exc()}")
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
                final_df = merge_and_compute(cake_df, web_pivot, phone_pivot, conversion_df)
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
            
            # Verify metrics match between Athena and final report
            verify_metrics_match(athena_df, final_df)
            
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

def allocate_phone_metrics(cake_df, phone_df, athena_df=None):
    """Allocate phone metrics to subIDs based on web activity."""
    st.write("\n### Phone Attribution Debug")
    
    # If athena_df not provided, try to get it from session state
    if athena_df is None and 'athena_df' in st.session_state:
        athena_df = st.session_state['athena_df']
        st.write("Retrieved athena_df from session state")
    
    # Convert phone metrics to integers
    phone_metrics = ['Phone DIFM Sales', 'Phone DIY Sales', 'DIFM Phone Installs']
    for metric in phone_metrics:
        cake_df[metric] = 0  # Initialize all phone metrics to 0
    
    # Debug specific affiliate that has discrepancies
    st.write("\n### Debugging 42299_ Phone DIFM Installs")
    
    # Group by PID to handle each partner's phone metrics
    for pid in phone_df.index.unique():
        if pd.isna(pid) or pid == '': continue
        
        # Special debug for 42299
        if pid == '42299':
            st.write(f"\n--- Detailed Debug for PID {pid} ---")
            st.write("Phone pivot data for this PID:")
            st.write(phone_df.loc[pid])
            
            # Check raw Athena data for this PID
            st.write("\nRaw Athena data counts for this PID:")
            if athena_df is not None:
                from_athena = len(athena_df[
                    (~athena_df['Lead_DNIS'].str.contains("WEB", na=False)) & 
                    (athena_df['PID'] == pid) &
                    (athena_df['INSTALL_METHOD'].str.contains('DIFM', na=False)) &
                    (athena_df['Install_Date'].notna())
                ])
                st.write(f"Phone DIFM installs in raw Athena for PID {pid}: {from_athena}")
            else:
                st.write("Athena data not available for comparison")
            
            # Show 42299_ rows in cake_df before allocation
            st.write("\n42299_ rows in optimization table before allocation:")
            rows_42299 = cake_df[cake_df['PID'] == pid]
            st.write(f"Number of 42299_ rows: {len(rows_42299)}")
            st.write(rows_42299[['Concatenated', 'PID', 'Web DIFM Sales', 'Web DIY Sales', 'DIFM Web Installs']])
        
        # Get phone metrics for this PID
        phone_metrics_for_pid = phone_df.loc[pid]
        
        # Get all rows for this PID in cake_df
        pid_mask = cake_df['PID'] == pid
        pid_rows = cake_df[pid_mask]
        
        if len(pid_rows) == 0: continue
        
        st.write(f"\nProcessing PID: {pid}")
        st.write("Phone metrics to allocate:", {
            'DIFM Sales': int(phone_metrics_for_pid['Phone DIFM Sales']),
            'DIY Sales': int(phone_metrics_for_pid['Phone DIY Sales']),
            'DIFM Installs': int(phone_metrics_for_pid['DIFM Phone Installs'])
        })
        
        # Step 1: Proportional Allocation
        total_web_difm = float(pid_rows['Web DIFM Sales'].sum())
        total_web_diy = float(pid_rows['Web DIY Sales'].sum())
        total_web_installs = float(pid_rows['DIFM Web Installs'].sum())
        
        # Special debug for 42299
        if pid == '42299':
            st.write(f"Total web metrics for 42299_:")
            st.write(f"Total Web DIFM Sales: {total_web_difm}")
            st.write(f"Total Web DIY Sales: {total_web_diy}")
            st.write(f"Total DIFM Web Installs: {total_web_installs}")
            
            # Track allocations in detail
            allocation_details = []
        
        if total_web_difm > 0 or total_web_diy > 0 or total_web_installs > 0:
            # Allocate proportionally
            for idx in pid_rows.index:
                row = pid_rows.loc[idx]
                
                # Allocate DIFM Sales
                if total_web_difm > 0:
                    allocated = int(round(
                        float(phone_metrics_for_pid['Phone DIFM Sales']) * 
                        (float(row['Web DIFM Sales']) / total_web_difm)
                    ))
                    cake_df.loc[idx, 'Phone DIFM Sales'] = allocated
                
                # Allocate DIY Sales
                if total_web_diy > 0:
                    allocated = int(round(
                        float(phone_metrics_for_pid['Phone DIY Sales']) * 
                        (float(row['Web DIY Sales']) / total_web_diy)
                    ))
                    cake_df.loc[idx, 'Phone DIY Sales'] = allocated
                
                # Allocate DIFM Installs
                if total_web_installs > 0:
                    allocated = int(round(
                        float(phone_metrics_for_pid['DIFM Phone Installs']) * 
                        (float(row['DIFM Web Installs']) / total_web_installs)
                    ))
                    cake_df.loc[idx, 'DIFM Phone Installs'] = allocated
                    
                    # Special debug for 42299
                    if pid == '42299':
                        allocation_details.append({
                            'Concatenated': row['Concatenated'],
                            'Web DIFM Installs': float(row['DIFM Web Installs']),
                            'Proportion': float(row['DIFM Web Installs']) / total_web_installs,
                            'Calculation': f"{float(phone_metrics_for_pid['DIFM Phone Installs'])} * {float(row['DIFM Web Installs']) / total_web_installs}",
                            'Raw Value': float(phone_metrics_for_pid['DIFM Phone Installs']) * (float(row['DIFM Web Installs']) / total_web_installs),
                            'Rounded': allocated,
                        })
            
            # Step 2: Fix Under-Allocated Totals
            for metric, phone_metric in [
                ('Phone DIFM Sales', 'Phone DIFM Sales'),
                ('Phone DIY Sales', 'Phone DIY Sales'),
                ('DIFM Phone Installs', 'DIFM Phone Installs')
            ]:
                total_allocated = int(cake_df.loc[pid_mask, metric].sum())
                total_available = int(phone_metrics_for_pid[phone_metric])
                
                # Special debug for 42299
                if pid == '42299' and metric == 'DIFM Phone Installs':
                    st.write(f"\nDIFM Phone Installs allocation for 42299_:")
                    st.write(f"Total DIFM Phone Installs in pivot: {total_available}")
                    st.write(f"Total allocated in first pass: {total_allocated}")
                    st.write("\nAllocation details by row:")
                    st.write(pd.DataFrame(allocation_details))
                
                if total_allocated < total_available:
                    remaining = total_available - total_allocated
                    st.write(f"Under-allocated {metric}: {remaining} units remaining")
                    
                    # Sort by Leads or Web DIFM Sales
                    sorted_idx = pid_rows.sort_values(['Web DIFM Sales', 'Leads'], 
                                                    ascending=[False, False]).index
                    
                    # Special debug for 42299
                    if pid == '42299' and metric == 'DIFM Phone Installs':
                        st.write(f"\nDistributing {remaining} remaining DIFM Phone Installs")
                        st.write("Sorted rows to receive remaining installs:")
                        sorted_rows = pid_rows.loc[sorted_idx[:remaining], ['Concatenated', 'Web DIFM Sales', 'Leads']]
                        st.write(sorted_rows)
                    
                    for idx in sorted_idx[:remaining]:
                        cake_df.loc[idx, metric] += 1
                        
                        # Special debug for 42299
                        if pid == '42299' and metric == 'DIFM Phone Installs':
                            st.write(f"Added 1 to {cake_df.loc[idx, 'Concatenated']}")
        
        else:
            # Step 3: Catch-All Attribution (No Web Activity)
            st.write("No web activity found - using catch-all attribution")
            
            # Find row with highest Leads
            max_leads_idx = pid_rows['Leads'].idxmax()
            
            # Assign all phone metrics to this row
            cake_df.loc[max_leads_idx, 'Phone DIFM Sales'] = int(phone_metrics_for_pid['Phone DIFM Sales'])
            cake_df.loc[max_leads_idx, 'Phone DIY Sales'] = int(phone_metrics_for_pid['Phone DIY Sales'])
            cake_df.loc[max_leads_idx, 'DIFM Phone Installs'] = int(phone_metrics_for_pid['DIFM Phone Installs'])
        
        # Special debug for 42299
        if pid == '42299':
            st.write("\nFinal allocation results for 42299_:")
            st.write(cake_df[pid_mask][['Concatenated', 'Phone DIFM Sales', 'Phone DIY Sales', 'DIFM Phone Installs']])
            st.write(f"Total DIFM Phone Installs allocated: {cake_df.loc[pid_mask, 'DIFM Phone Installs'].sum()}")
    
    # Ensure all phone metrics are integers
    for metric in phone_metrics:
        cake_df[metric] = cake_df[metric].fillna(0).astype(int)
    
    # Final check for 42299
    pid = '42299'
    pid_mask = cake_df['PID'] == pid
    st.write(f"\nFinal check for {pid}:")
    st.write(f"Total DIFM Phone Installs for {pid}: {cake_df.loc[pid_mask, 'DIFM Phone Installs'].sum()}")
    
    return cake_df

if __name__ == "__main__":
    show_bob_analysis() 