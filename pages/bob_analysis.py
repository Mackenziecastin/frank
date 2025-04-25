import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# -------------------------------
# Constants
# -------------------------------

TFN_SHEET_URL = "https://docs.google.com/spreadsheets/d/10BHN_-Wz_ZPmi7rezNtqiDPTguHOoNzmkXzovFOTbaU/edit?gid=1629976834#gid=1629976834"

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
    """
    Load and combine TFN data from both RESI TFN sheet and Display TFN sheet.
    
    Parameters:
    ----------
    sheet_url : str
        The URL of the Google sheet
        
    Returns:
    -------
    pd.DataFrame
        Combined DataFrame with TFN to PID mapping
    """
    # Updated URLs to use the correct GIDs
    base_url = "https://docs.google.com/spreadsheets/d/10BHN_-Wz_ZPmi7rezNtqiDPTguHOoNzmkXzovFOTbaU"
    
    # RESI TFN sheet (GID 1629976834)
    resi_csv_url = f"{base_url}/export?format=csv&gid=1629976834"
    
    # Display TFN sheet (GID 383243987)
    display_csv_url = f"{base_url}/export?format=csv&gid=383243987"
    
    st.write(f"Loading RESI TFN from: {resi_csv_url}")
    
    try:
        # Load RESI TFN sheet - explicitly use row 1 as header (skip_blank_lines=True helps with empty rows)
        resi_df = pd.read_csv(resi_csv_url, header=0, skip_blank_lines=True)
        
        # Debug: print column names to identify the actual PID column
        st.write("RESI TFN sheet columns:", resi_df.columns.tolist())
        
        # Check if PID column exists, if not, look for similarly named columns
        if 'PID' not in resi_df.columns:
            # Look for alternative column names
            pid_candidates = [col for col in resi_df.columns if 'pid' in col.lower() or 'id' in col.lower()]
            st.write(f"PID column not found. Potential PID columns: {pid_candidates}")
            
            if pid_candidates:
                # Rename the first candidate to 'PID'
                resi_df = resi_df.rename(columns={pid_candidates[0]: 'PID'})
                st.write(f"Using '{pid_candidates[0]}' as the PID column")
            else:
                # Try loading again with no header and setting first row as column names
                st.warning("No PID column found. Trying to load with first row as header...")
                
                try:
                    # Re-read the file with no header and manually set the first row as columns
                    resi_df = pd.read_csv(resi_csv_url, header=None, skip_blank_lines=True)
                    resi_df.columns = resi_df.iloc[0]
                    resi_df = resi_df.drop(0)  # Remove the first row which is now the header
                    resi_df = resi_df.reset_index(drop=True)
                    
                    st.write("After setting first row as header, columns:", resi_df.columns.tolist())
                    
                    # Check if PID column exists after the fix
                    if 'PID' not in resi_df.columns:
                        # Look for alternative column names again
                        pid_candidates = [col for col in resi_df.columns if 'pid' in str(col).lower() or 'id' in str(col).lower()]
                        st.write(f"PID column still not found. Potential PID columns: {pid_candidates}")
                        
                        if pid_candidates:
                            # Rename the first candidate to 'PID'
                            resi_df = resi_df.rename(columns={pid_candidates[0]: 'PID'})
                            st.write(f"Using '{pid_candidates[0]}' as the PID column")
                        else:
                            # Create an empty DataFrame with required columns if no PID column found
                            st.error("No suitable PID column found in the RESI TFN sheet. Using a placeholder.")
                            return pd.DataFrame(columns=['PID', 'TFN', 'Clean_TFN'])
                except Exception as e:
                    st.error(f"Error reloading RESI TFN sheet: {str(e)}")
                    return pd.DataFrame(columns=['PID', 'TFN', 'Clean_TFN'])
        
        # Also check for TFN column
        if 'TFN' not in resi_df.columns:
            # Look for alternative column names
            tfn_candidates = [col for col in resi_df.columns if 'tfn' in str(col).lower() or 'phone' in str(col).lower() or 'number' in str(col).lower()]
            st.write(f"TFN column not found. Potential TFN columns: {tfn_candidates}")
            
            if tfn_candidates:
                # Rename the first candidate to 'TFN'
                resi_df = resi_df.rename(columns={tfn_candidates[0]: 'TFN'})
                st.write(f"Using '{tfn_candidates[0]}' as the TFN column")
            else:
                # Create an empty DataFrame with required columns if no TFN column found
                st.error("No suitable TFN column found in the RESI TFN sheet. Using a placeholder.")
                return pd.DataFrame(columns=['PID', 'TFN', 'Clean_TFN'])
        
        # Convert the PID column to integers (first to numeric, then to integer)
        # Convert to numeric first to handle any non-numeric values
        resi_df['PID'] = pd.to_numeric(resi_df['PID'], errors='coerce')
        # Then convert to integer, handling NaN values
        resi_df['PID'] = resi_df['PID'].fillna(0).astype(int)
        
        # Convert TFN to string and clean it (remove any decimal points)
        resi_df['TFN'] = resi_df['TFN'].astype(str)
        resi_df['TFN'] = resi_df['TFN'].apply(lambda x: x.split('.')[0] if '.' in x else x)
        
        # Sample data to verify content
        st.write("Sample rows from RESI TFN sheet:")
        st.write(resi_df.head(3))
        
        # Filter out rows where PID or TFN is empty or zero
        resi_df = resi_df[(resi_df['PID'] != 0) & (resi_df['TFN'] != '') & (resi_df['TFN'] != '0')]
        
        # Check the format of PID after loading
        st.write(f"RESI TFN PIDs (first 5): {resi_df['PID'].head(5).tolist()}")
        
        # Define critical phone numbers to check in the RESI sheet
        critical_numbers = {
            '8446778720': 4790,
            '8005717438': 42299,
            '8009734275': 42038
        }
        
        # Check if critical numbers exist in the RESI sheet
        for phone, expected_pid in critical_numbers.items():
            clean_phone = clean_phone_number(phone)
            matches = resi_df[resi_df['TFN'].apply(lambda x: clean_phone_number(str(x)) == clean_phone)]
            
            if len(matches) == 0:
                st.warning(f"Critical number {phone} not found in RESI sheet")
            else:
                pid_in_sheet = matches['PID'].iloc[0]
                if pid_in_sheet != expected_pid:
                    st.warning(f"Critical number {phone} maps to PID {pid_in_sheet} in RESI sheet, but expected {expected_pid}")
        
        st.write(f"Loaded {len(resi_df)} rows from RESI TFN sheet")
        
        # Load Display TFN sheet
        st.write(f"Loading Display TFN from: {display_csv_url}")
        try:
            # Load Display TFN sheet - explicitly use row 1 as header
            display_df = pd.read_csv(display_csv_url, header=0, skip_blank_lines=True)
            
            # Debug: print column names to identify the actual PID column
            st.write("Display TFN sheet columns:", display_df.columns.tolist())
            
            # Check if we need to try loading with first row as header
            if 'pid' not in [col.lower() for col in display_df.columns]:
                st.warning("No PID column found in Display TFN sheet. Trying to load with first row as header...")
                
                try:
                    # Re-read the file with no header and manually set the first row as columns
                    display_df = pd.read_csv(display_csv_url, header=None, skip_blank_lines=True)
                    display_df.columns = display_df.iloc[0]
                    display_df = display_df.drop(0)  # Remove the first row which is now the header
                    display_df = display_df.reset_index(drop=True)
                    
                    st.write("After setting first row as header, Display TFN columns:", display_df.columns.tolist())
                except Exception as e:
                    st.error(f"Error reloading Display TFN sheet: {str(e)}")
            
            # Identify the PID and TFN columns
            pid_columns = [col for col in display_df.columns if 'pid' in str(col).lower()]
            tfn_columns = [col for col in display_df.columns if 'tfn' in str(col).lower() or 'phone' in str(col).lower() or 'number' in str(col).lower()]
            
            if pid_columns and tfn_columns:
                pid_col = pid_columns[0]
                tfn_col = tfn_columns[0]
                
                # Rename columns to match RESI format
                display_df = display_df.rename(columns={pid_col: 'PID', tfn_col: 'TFN'})
                
                # Convert the PID column to integers (first to numeric, then to integer)
                display_df['PID'] = pd.to_numeric(display_df['PID'], errors='coerce')
                display_df['PID'] = display_df['PID'].fillna(0).astype(int)
                
                # Convert TFN to string and clean it (remove any decimal points)
                display_df['TFN'] = display_df['TFN'].astype(str)
                display_df['TFN'] = display_df['TFN'].apply(lambda x: x.split('.')[0] if '.' in x else x)
                
                # Filter out rows where PID or TFN is empty or zero
                display_df = display_df[(display_df['PID'] != 0) & (display_df['TFN'] != '') & (display_df['TFN'] != '0')]
                
                st.write(f"Loaded {len(display_df)} rows from Display TFN sheet")
                
                # Combine both dataframes
                combined_df = pd.concat([resi_df[['PID', 'TFN']], display_df[['PID', 'TFN']]], ignore_index=True)
            else:
                st.warning("Could not identify PID and TFN columns in Display TFN sheet. Using only RESI data.")
                combined_df = resi_df[['PID', 'TFN']]
        except Exception as e:
            st.warning(f"Error loading Display TFN sheet: {str(e)}. Using only RESI data.")
            combined_df = resi_df[['PID', 'TFN']]
        
        # Add clean TFN column - make sure all TFNs are cleaned properly
        combined_df['Clean_TFN'] = combined_df['TFN'].apply(lambda x: clean_phone_number(str(x)))
        
        # Remove duplicates based on Clean_TFN
        combined_df = combined_df.drop_duplicates(subset=['Clean_TFN'])
        
        # Ensure critical numbers are in the final mapping
        for phone, expected_pid in critical_numbers.items():
            clean_phone = clean_phone_number(phone)
            if clean_phone not in combined_df['Clean_TFN'].values:
                st.warning(f"Adding missing critical number {phone} to TFN mapping with PID {expected_pid}")
                new_row = pd.DataFrame({
                    'PID': [expected_pid],
                    'TFN': [phone],
                    'Clean_TFN': [clean_phone]
                })
                combined_df = pd.concat([combined_df, new_row], ignore_index=True)
            else:
                row_idx = combined_df[combined_df['Clean_TFN'] == clean_phone].index[0]
                pid_in_mapping = combined_df.loc[row_idx, 'PID']
                
                if pid_in_mapping != expected_pid:
                    st.warning(f"Correcting PID for {phone} from {pid_in_mapping} to {expected_pid}")
                    combined_df.loc[row_idx, 'PID'] = expected_pid
        
        # Make sure all PIDs are integers
        combined_df['PID'] = combined_df['PID'].astype(int)
        
        # Create a mapping from clean TFN to PID
        tfn_to_pid = dict(zip(combined_df['Clean_TFN'], combined_df['PID']))
        
        st.write(f"Final combined TFN mapping contains {len(tfn_to_pid)} entries")
        st.write("Sample of TFN to PID mapping:")
        sample_mapping = {k: v for i, (k, v) in enumerate(tfn_to_pid.items()) if i < 5}
        st.write(sample_mapping)
        
        # Print a summary of the final mapping
        print(f"TFN mapping summary: {len(tfn_to_pid)} total entries")
        
        return combined_df
    except Exception as e:
        st.error(f"Error processing TFN data: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a placeholder DataFrame with the required columns
        st.warning("Using a placeholder TFN mapping with critical numbers only")
        critical_numbers = {
            '8446778720': 4790,
            '8005717438': 42299,
            '8009734275': 42038
        }
        
        # Create a DataFrame with just the critical numbers
        fallback_data = []
        for phone, pid in critical_numbers.items():
            fallback_data.append({
                'PID': int(pid),  # Ensure PID is an integer
                'TFN': phone,
                'Clean_TFN': clean_phone_number(phone)
            })
        
        return pd.DataFrame(fallback_data)

def clean_phone_number(phone_str):
    """
    Clean a phone number string to a standard format (digits only).
    
    Parameters:
    ----------
    phone_str : str
        The phone number string to clean
        
    Returns:
    -------
    str
        Cleaned phone number with only digits
    """
    if pd.isna(phone_str) or phone_str == '':
        return ''
        
    # Extract only digits
    digits_only = ''.join(c for c in str(phone_str) if c.isdigit())
    
    # If it starts with country code 1 and is 11 digits, remove the 1
    if len(digits_only) == 11 and digits_only.startswith('1'):
        return digits_only[1:]
    
    return digits_only

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

def clean_athena(athena_df, tfn_df, leads_df, start_date, end_date):
    """
    Clean and process Athena data for analysis.
    
    Parameters:
    ----------
    athena_df : DataFrame
        Raw Athena data
    tfn_df : DataFrame
        TFN mapping data
    leads_df : DataFrame
        Leads data
    start_date : str or datetime
        Start date for filtering
    end_date : str or datetime
        End date for filtering
    
    Returns:
    -------
    DataFrame
        Cleaned Athena data with matched PIDs
    """
    # Display column names for debugging
    st.write("Available columns in Athena data:", athena_df.columns.tolist())
    
    # Filter by date range
    if start_date and end_date:
        st.write(f"Filtering data between {start_date} and {end_date}")
        
        # Convert to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Specifically use Lead_Creation_Date as requested
        if 'Lead_Creation_Date' in athena_df.columns:
            date_column = 'Lead_Creation_Date'
            st.write(f"Using '{date_column}' as the date filter column")
            
            try:
                # Convert the date column to datetime
                athena_df[date_column] = pd.to_datetime(athena_df[date_column], errors='coerce')
                # Filter by date range
                athena_df = athena_df[(athena_df[date_column] >= start_date) & (athena_df[date_column] <= end_date)]
                st.write(f"Records after date filtering: {len(athena_df)}")
            except Exception as e:
                st.warning(f"Error converting date column '{date_column}': {str(e)}. Using all data.")
        else:
            # Fall back to other date columns if Lead_Creation_Date is not available
            st.warning("Lead_Creation_Date column not found. Trying alternative date columns.")
            
            date_column = None
            date_column_candidates = ['Date', 'date', 'created_date', 'Created_Date', 'lead_date']
            
            for col in date_column_candidates:
                if col in athena_df.columns:
                    date_column = col
                    st.write(f"Using '{date_column}' as the date filter column")
                    break
            
            if date_column is None:
                # Look for any column with 'date' in the name
                date_cols = [col for col in athena_df.columns if 'date' in col.lower()]
                if date_cols:
                    date_column = date_cols[0]
                    st.write(f"Using '{date_column}' as the date filter column")
                else:
                    st.warning("No date column found for filtering. Using all data.")
            
            # Filter the dataframe if a date column was found
            if date_column:
                try:
                    # Convert the date column to datetime
                    athena_df[date_column] = pd.to_datetime(athena_df[date_column], errors='coerce')
                    # Filter by date range
                    athena_df = athena_df[(athena_df[date_column] >= start_date) & (athena_df[date_column] <= end_date)]
                    st.write(f"Records after date filtering: {len(athena_df)}")
                except Exception as e:
                    st.warning(f"Error converting date column '{date_column}': {str(e)}. Using all data.")
    
    # Clean affiliate code and filter records
    athena_df['Clean_Affiliate_Code'] = athena_df['Affiliate_Code'].apply(clean_affiliate_code)
    athena_df = athena_df[
        (athena_df['Clean_Affiliate_Code'].notna()) & 
        (athena_df['Clean_Affiliate_Code'] != "") & 
        (athena_df['Clean_Affiliate_Code'] != "CAKE")
    ]
    st.write(f"Records after affiliate code filtering: {len(athena_df)}")
    
    # Count records with "WEB" in Lead_DNIS
    web_count = athena_df[athena_df['Lead_DNIS'].str.contains('WEB', na=False, case=False)].shape[0]
    st.write(f"Records with 'WEB' in Lead_DNIS: {web_count}")
    
    # For non-WEB records, try to match PIDs
    athena_df['PID'] = None
    non_web_mask = ~athena_df['Lead_DNIS'].str.contains('WEB', na=False, case=False)
    non_web_count = non_web_mask.sum()
    st.write(f"Non-WEB records to match: {non_web_count}")
    
    # Create TFN mapping from clean TFN to PID
    tfn_map = dict(zip(tfn_df['Clean_TFN'], tfn_df['PID']))
    st.write(f"TFN mapping contains {len(tfn_map)} entries")
    
    match_count = 0
    for idx, row in athena_df[non_web_mask].iterrows():
        pid = match_pid(row, tfn_map)
        if pid is not None:
            athena_df.at[idx, 'PID'] = pid
            match_count += 1
    
    st.write(f"Successfully matched {match_count} out of {non_web_count} non-WEB records")
    
    # Special debug for DNIS that maps to PID 42038
    if '8009734275' in athena_df['Lead_DNIS'].values:
        dnis_42038 = athena_df[athena_df['Lead_DNIS'].str.contains('8009734275', na=False)]
        st.write(f"Found {len(dnis_42038)} records with DNIS 8009734275")
        st.write(dnis_42038[['Lead_DNIS', 'PID', 'INSTALL_METHOD']])
    
    # Analyze records by PID
    analyze_records_by_pid(athena_df)
    
    return athena_df

def analyze_pre_matchback_phone_metrics(athena_df, tfn_map=None):
    """
    Perform pre-matchback analysis on phone records to identify potential matching issues.
    
    Parameters:
    -----------
    athena_df : pd.DataFrame
        DataFrame containing the Athena records
    tfn_map : dict, optional
        Mapping from cleaned phone numbers to PIDs for verification
        
    Returns:
    --------
    pre_match_stats : dict
        Dictionary containing statistics about pre-matchback phone records
    """
    st.subheader("Pre-Matchback Phone Analysis")
    
    # Count raw phone records (non-WEB)
    non_web_mask = ~athena_df['Lead_DNIS'].str.contains('WEB', na=False)
    raw_phone_count = non_web_mask.sum()
    
    st.write(f"Raw Phone Records (non-WEB): {raw_phone_count}")
    
    # Define problematic numbers to track specifically
    problematic_numbers = ['8446778720', '8005717438', '8009734275']
    problematic_pids = {'8446778720': '4790', '8005717438': '42299', '8009734275': '42038'}
    
    # Check for specific problematic DNIS values
    for dnis in problematic_numbers:
        cleaned_dnis = clean_phone_number(dnis)
        mask = athena_df['Lead_DNIS'].apply(lambda x: clean_phone_number(str(x)) == cleaned_dnis if pd.notna(x) else False)
        count = mask.sum()
        if count > 0:
            in_map = "Yes" if tfn_map and cleaned_dnis in tfn_map else "No"
            mapped_pid = tfn_map.get(cleaned_dnis, "Not found") if tfn_map else "N/A"
            expected_pid = problematic_pids.get(dnis, "Unknown")
            
            st.write(f"**DNIS {dnis}**: Found {count} records")
            st.write(f"  - Cleaned to: {cleaned_dnis}")
            st.write(f"  - In TFN map: {in_map}")
            st.write(f"  - Mapped PID: {mapped_pid}")
            st.write(f"  - Expected PID: {expected_pid}")
            
            # Sample a few records for this DNIS
            sample_records = athena_df[mask].head(3)
            st.write("  - Sample records:")
            st.dataframe(sample_records[['PID', 'Lead_DNIS', 'Affiliate_Code', 'INSTALL_METHOD']])
    
    # Sample a subset of DNIS values to check match status
    sample_size = min(50, raw_phone_count)
    if raw_phone_count > 0:
        sampled_dnis = athena_df[non_web_mask]['Lead_DNIS'].sample(sample_size, random_state=42).to_list()
        checked_dnis = []
        
        for dnis in sampled_dnis:
            cleaned = clean_phone_number(str(dnis))
            matched = "Yes" if tfn_map and cleaned in tfn_map else "No"
            mapped_to = tfn_map.get(cleaned, "Not found") if tfn_map else "N/A"
            
            checked_dnis.append({
                "DNIS": dnis,
                "Cleaned_DNIS": cleaned,
                "Matched": matched,
                "Mapped_To": mapped_to
            })
        
        st.write(f"Sample of {sample_size} DNIS values checked against TFN map:")
        check_df = pd.DataFrame(checked_dnis)
        st.dataframe(check_df)
        
        # Calculate match rate
        if tfn_map:
            match_count = sum(1 for item in checked_dnis if item["Matched"] == "Yes")
            match_rate = (match_count / sample_size) * 100
            st.write(f"Match rate in sample: {match_rate:.1f}% ({match_count}/{sample_size})")
            
            if match_rate < 90:
                st.warning(f"Low match rate detected! Only {match_rate:.1f}% of sampled DNIS values matched to a PID.")
    
    # Return statistics for further analysis
    pre_match_stats = {
        "raw_phone_count": raw_phone_count,
        "problematic_numbers_found": {dnis: athena_df['Lead_DNIS'].apply(
            lambda x: clean_phone_number(str(x)) == clean_phone_number(dnis) if pd.notna(x) else False
        ).sum() for dnis in problematic_numbers}
    }
    
    return pre_match_stats

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

def match_pid(row, tfn_map):
    """
    Match a phone record to a PID using the TFN mapping.
    
    Parameters:
    ----------
    row : pd.Series
        Row from the phone records DataFrame
    tfn_map : dict
        Mapping from cleaned phone numbers to PIDs
        
    Returns:
    -------
    pid : str or np.nan
        Matched PID or np.nan if no match found
    """
    # Only process non-WEB records
    if 'WEB' in str(row['Lead_DNIS']):
        return np.nan
    
    # Clean the phone number from Lead_DNIS
    phone_num = clean_phone_number(str(row['Lead_DNIS']))
    
    # Debug specific problematic numbers
    problematic_numbers = ['8446778720', '8005717438', '8009734275']
    clean_problematic = [clean_phone_number(num) for num in problematic_numbers]
    
    if phone_num in clean_problematic:
        print(f"DEBUG: Processing problematic number {phone_num}")
        
    # Try to match the cleaned phone number to a PID
    if phone_num in tfn_map:
        pid = tfn_map[phone_num]
        
        if phone_num in clean_problematic:
            print(f"DEBUG: Matched {phone_num} to PID {pid}")
        
        return pid
    else:
        if phone_num in clean_problematic:
            print(f"DEBUG: Failed to match problematic number {phone_num}")
            print(f"DEBUG: Available keys in tfn_map: {list(tfn_map.keys())[:10]}... (total: {len(tfn_map)})")
            
            # Check if any similar numbers exist in the mapping
            similar_keys = [k for k in tfn_map.keys() if phone_num in k or k in phone_num]
            if similar_keys:
                print(f"DEBUG: Found similar keys: {similar_keys}")
        
        return np.nan

def analyze_records_by_pid(athena_df):
    """
    Analyze records by PID, count occurrences, and generate visualizations.
    
    Parameters:
    ----------
    athena_df : DataFrame
        Athena data with PIDs matched
    
    Returns:
    -------
    None
    """
    st.subheader("Analysis of Records by PID")
    
    # Count records by PID
    pid_counts = athena_df[athena_df['PID'].notna()].groupby('PID').size().reset_index(name='count')
    pid_counts = pid_counts.sort_values('count', ascending=False)
    
    st.write(f"Total PIDs with records: {len(pid_counts)}")
    
    # Display top 20 PIDs by record count
    st.write("Top 20 PIDs by record count:")
    st.dataframe(pid_counts.head(20))
    
    # Create bar chart of top 20 PIDs
    top_pids = pid_counts.head(20)
    fig = px.bar(top_pids, x='PID', y='count', title='Top 20 PIDs by Record Count')
    fig.update_layout(xaxis_title='PID', yaxis_title='Number of Records')
    st.plotly_chart(fig)
    
    # Key PIDs of interest
    key_pids = ['42299', '4790', '42038']
    st.subheader("Details for PIDs of Interest")
    
    key_pid_data = []
    for pid in key_pids:
        # Get observed count
        observed = 0
        if pid in pid_counts['PID'].values:
            observed = pid_counts.loc[pid_counts['PID'] == pid, 'count'].iloc[0]
        
        # Get DNIS for this PID
        dnis_values = athena_df[athena_df['PID'] == pid]['Lead_DNIS'].unique()
        dnis_str = ', '.join(str(d) for d in dnis_values) if len(dnis_values) > 0 else 'None found'
        
        # Append to data for visualization
        key_pid_data.append({
            'PID': pid,
            'Observed Records': observed,
            'DNIS Values': dnis_str
        })
    
    # Show table of key PIDs
    st.write("Details for Key PIDs:")
    st.dataframe(pd.DataFrame(key_pid_data))

if __name__ == "__main__":
    show_bob_analysis()