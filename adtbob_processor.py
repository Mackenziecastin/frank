import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

class ADTBobProcessor:
    def __init__(self):
        self.affiliate_df = None
        self.advanced_df = None
        self.partner_list_df = None
        self.processed_affiliate_df = None
        self.processed_advanced_df = None
        
    def load_data(self, affiliate_file, advanced_file, partner_list_file=None):
        """Load data from CSV files."""
        self.affiliate_df = pd.read_csv(affiliate_file)
        self.advanced_df = pd.read_csv(advanced_file)
        
        if partner_list_file:
            self.partner_list_df = pd.read_csv(partner_list_file)
            
        # Convert date columns to datetime
        date_columns = {
            'affiliate': ['Created Date', 'Sale_Date'],
            'advanced': ['Action Date']
        }
        
        for col in date_columns['affiliate']:
            if col in self.affiliate_df.columns:
                self.affiliate_df[col] = pd.to_datetime(self.affiliate_df[col])
                
        for col in date_columns['advanced']:
            if col in self.advanced_df.columns:
                self.advanced_df[col] = pd.to_datetime(self.advanced_df[col])
    
    def clean_data(self):
        """Clean and process the data."""
        if self.affiliate_df is None or self.advanced_df is None:
            raise ValueError("Data not loaded. Please load data first.")
            
        # Process affiliate data
        self.processed_affiliate_df = self.affiliate_df.copy()
        
        # Process advanced data
        self.processed_advanced_df = self.advanced_df.copy()
        
        # Extract partner IDs from URLs if needed
        if 'Click URL' in self.processed_affiliate_df.columns:
            self.processed_affiliate_df['Partner_ID'] = self.processed_affiliate_df['Click URL'].apply(
                lambda x: x.split('partner=')[-1].split('&')[0] if 'partner=' in str(x) else 'Unknown'
            )
            
        if 'Landing Page URL' in self.processed_advanced_df.columns:
            self.processed_advanced_df['Partner_ID'] = self.processed_advanced_df['Landing Page URL'].apply(
                lambda x: x.split('partner=')[-1].split('&')[0] if 'partner=' in str(x) else 'Unknown'
            )
    
    def generate_optimization_report(self):
        """Generate the optimization report."""
        if self.processed_affiliate_df is None or self.processed_advanced_df is None:
            raise ValueError("Data not processed. Please clean data first.")
            
        # Aggregate affiliate data by partner
        affiliate_metrics = self.processed_affiliate_df.groupby('Partner_ID').agg({
            'Transaction Count': 'sum',
            'Net Sales Amount': 'sum'
        }).reset_index()
        
        # Aggregate advanced data by partner
        advanced_metrics = self.processed_advanced_df.groupby('Partner_ID').agg({
            'Event Type': lambda x: (x == 'Lead Submission').sum(),
            'Action Earnings': 'sum'
        }).reset_index()
        
        # Merge the metrics
        report = pd.merge(
            advanced_metrics,
            affiliate_metrics,
            on='Partner_ID',
            how='outer'
        ).fillna(0)
        
        # Rename columns
        report = report.rename(columns={
            'Event Type': 'Leads',
            'Action Earnings': 'Spend',
            'Transaction Count': 'Sales',
            'Net Sales Amount': 'Revenue'
        })
        
        # Calculate additional metrics
        report['Conversion_Rate'] = (report['Sales'] / report['Leads'] * 100).round(2)
        report['ROAS'] = (report['Revenue'] / report['Spend']).round(2)
        
        # Add partner names if partner list is available
        if self.partner_list_df is not None:
            report = pd.merge(
                report,
                self.partner_list_df[['Partner_ID', 'Partner_Name']],
                on='Partner_ID',
                how='left'
            )
        else:
            report['Partner_Name'] = report['Partner_ID']
            
        return report
    
    def export_report(self, filename):
        """Export the report to Excel."""
        report = self.generate_optimization_report()
        
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Write the report
            report.to_excel(writer, sheet_name='Optimization Report', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Optimization Report']
            
            # Add formats
            money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
            percent_fmt = workbook.add_format({'num_format': '0.00%'})
            
            # Apply formats to columns
            for idx, col in enumerate(report.columns):
                if col in ['Spend', 'Revenue']:
                    worksheet.set_column(idx, idx, 15, money_fmt)
                elif col in ['Conversion_Rate']:
                    worksheet.set_column(idx, idx, 15, percent_fmt)
                else:
                    worksheet.set_column(idx, idx, 15) 