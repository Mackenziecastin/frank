# Partner Optimization Report Generator

This Streamlit application processes marketing data files to generate partner optimization reports. It extracts PID and SUBID information from URLs and performs various analyses on the data.

## Features

- Upload and process two CSV files (Affiliate Leads QA and Advanced Action Sheet)
- Automatic extraction of PID and SUBID from URLs
- Generation of comprehensive analysis including:
  - Leads, Spend, Sales, and Revenue metrics
  - Leads to Sale ratio
  - ROAS (Return on Ad Spend)
  - Current Rate
  - ECPL at $1.50
- Download results as Excel file with multiple sheets

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally

To run the application locally:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

To deploy this application so it's accessible online:

1. Create a GitHub repository and push this code to it
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Deploy the application by connecting it to your repository
5. Share the provided URL with others

## Usage

1. Open the application URL
2. Upload your Affiliate Leads QA file
3. Upload your Advanced Action Sheet
4. The application will automatically process the files
5. Preview the results in the browser
6. Download the complete report using the "Download Full Report" button

## File Requirements

Your input files should be CSV files containing:
- A URL column with tracking parameters
- Required metrics columns (Leads, Spend, Net Sales Amount, Order ID) 