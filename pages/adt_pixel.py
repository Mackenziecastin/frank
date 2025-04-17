import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import os

def show_adt_pixel():
    st.title("ADT Pixel Firing")
    
    st.write("""
    This tool processes ADT Athena reports and fires pixels for qualifying leads.
    Please upload your ADT Athena report below.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload ADT Athena Report (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            # Save the uploaded file temporarily
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run the pixel firing script
            if st.button("Process and Fire Pixels"):
                st.write("Running pixel firing process...")
                
                try:
                    # Run the adt_pixel_firing.py script
                    result = subprocess.run(
                        ["python3", "adt_pixel_firing.py", temp_file_path],
                        capture_output=True,
                        text=True
                    )
                    
                    # Display the output
                    st.code(result.stdout)
                    
                    if result.stderr:
                        st.error("Errors encountered:")
                        st.code(result.stderr)
                    
                    # Check for log file
                    log_file = f'adt_pixel_firing_{datetime.now().strftime("%Y%m%d")}.log'
                    if os.path.exists(log_file):
                        st.download_button(
                            label="Download Log File",
                            data=open(log_file, 'rb').read(),
                            file_name=log_file,
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"Error running pixel firing script: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    show_adt_pixel() 