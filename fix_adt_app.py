import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import logging
import sys
import os
import uuid
import requests
import re

def show_adt_pixel():
    """Simple ADT Pixel Firing interface with no warnings or pattern messages"""
    st.title("ADT Pixel Firing")
    
    st.write("""
    This tool processes ADT Athena reports and fires pixels for qualifying sales.
    Upload your ADT Athena report (CSV format) to begin.
    """)
    
    uploaded_file = st.file_uploader("Upload ADT Athena Report (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Process and Fire Pixels"):
            with st.spinner("Processing file..."):
                try:
                    # Save uploaded file to temp
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                    temp_file.write(uploaded_file.getvalue())
                    temp_file.close()
                    
                    # Import the module and process
                    from adt_pixel_firing import process_adt_report
                    process_adt_report(temp_file.name)
                    
                    # Clean up
                    os.unlink(temp_file.name)
                    
                    # Show the log file
                    log_filename = f'adt_pixel_firing_{datetime.now().strftime("%Y%m%d")}.log'
                    if os.path.exists(log_filename):
                        with open(log_filename, 'r') as f:
                            log_content = f.read()
                            
                            # Extract DIFM and DIY counts
                            difm_match = re.search(r'DIFM Sales: (\d+)', log_content)
                            diy_match = re.search(r'DIY Sales: (\d+)', log_content)
                            
                            difm_count = int(difm_match.group(1)) if difm_match else 0
                            diy_count = int(diy_match.group(1)) if diy_match else 0
                            
                            # Display success message
                            if difm_count > 0 or diy_count > 0:
                                st.success("✅ Process completed successfully!")
                                
                                # Display counts in columns
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("DIFM Pixels", difm_count)
                                with col2:
                                    st.metric("DIY Pixels", diy_count)
                                with col3:
                                    st.metric("Total Pixels", difm_count + diy_count)
                            else:
                                st.warning("Process completed, but no qualifying sales were found.")
                            
                            # Show log
                            with st.expander("View Processing Log"):
                                st.code(log_content)
                    else:
                        st.error("Log file not found. Process may have failed.")
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Run this as a standalone app for testing
if __name__ == "__main__":
    show_adt_pixel() 