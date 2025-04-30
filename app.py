import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from datetime import datetime, timedelta
import logging
import sys
import os
import uuid
import requests
import io

# Set page config at the start
st.set_page_config(page_title="Partner Optimization Report Generator", layout="wide")

def show_adt_pixel():
    """Display the ADT Pixel Firing interface"""
    st.title("ADT Pixel Firing")
    
    st.write("""
    This tool processes ADT Athena reports and fires pixels for qualifying sales.
    Upload your ADT Athena report (CSV format) to begin.
    """)
    
    uploaded_file = st.file_uploader("Upload ADT Athena Report (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Process and Fire Pixels"):
            process_adt_report(uploaded_file)

def main():
    """Main application entry point"""
    # Create the navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Frank (LaserAway)", "Bob (ADT)", "ADT Pixel Firing", "Brinks"])
    
    if page == "Frank (LaserAway)":
        show_main_page()
    elif page == "Bob (ADT)":
        # Import and show Bob's analysis page
        from pages.bob_analysis import show_bob_analysis
        show_bob_analysis()
    elif page == "ADT Pixel Firing":
        # Use the directly defined function instead of importing
        show_adt_pixel()
    elif page == "Brinks":
        # Import and show Brinks page
        from pages.brinks_optimization import show_brinks_optimization
        show_brinks_optimization()

if __name__ == "__main__":
    main() 