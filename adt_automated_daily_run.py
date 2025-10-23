#!/usr/bin/env python3
"""
ADT Daily Report Automation Script

This script automatically:
1. Connects to ADT's SFTP server
2. Downloads today's report from /Resi_New directory
3. Processes the report and fires pixels
4. Sends email notification if any errors occur

Schedule this to run daily at 12:30 PM EST
"""

import os
import sys
import logging
import smtplib
import traceback
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pysftp
from dotenv import load_dotenv

# Import the existing pixel firing logic
from adt_pixel_firing import process_adt_report

# Load environment variables from .env file
load_dotenv()

# Configure logging to both file and console
log_filename = f"adt_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Email configuration (load from environment variables)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

# Parse recipient emails (handle comma-separated list)
RECIPIENT_EMAILS = [email.strip() for email in RECIPIENT_EMAIL.split(',')] if RECIPIENT_EMAIL else []

# SFTP configuration (load from environment variables for security)
SFTP_HOST = os.getenv('SFTP_HOST', 'mfts.adt.com')
SFTP_USERNAME = os.getenv('SFTP_USERNAME', 'directagent')
SFTP_PASSWORD = os.getenv('SFTP_PASSWORD')
SFTP_PORT = int(os.getenv('SFTP_PORT', '22'))
SFTP_DIRECTORY = os.getenv('SFTP_DIRECTORY', '/Resi_New')

# Local directory to store downloaded reports
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), 'adt_reports')


def send_error_email(error_message, log_content=None):
    """
    Send an email notification when an error occurs
    """
    try:
        if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
            logging.error("Email configuration is incomplete. Cannot send error notification.")
            logging.error("Please set SENDER_EMAIL, SENDER_PASSWORD, and RECIPIENT_EMAIL in .env file")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ', '.join(RECIPIENT_EMAILS)  # Join multiple recipients
        msg['Subject'] = f'ADT Automation Error - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        
        # Email body
        body = f"""
ADT Daily Report Automation Error

An error occurred during the automated ADT report processing.

Error Details:
{error_message}

Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S EST")}

{"="*60}
Log Output:
{"="*60}
{log_content if log_content else "No log content available"}

Please review the full log file: {log_filename}

This is an automated message. Please do not reply.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        logging.info(f"Attempting to send error notification to {len(RECIPIENT_EMAILS)} recipient(s): {', '.join(RECIPIENT_EMAILS)}")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, text)  # Send to list of recipients
        server.quit()
        
        logging.info("Error notification email sent successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to send error notification email: {str(e)}")
        return False


def send_success_email(report_filename, summary_info):
    """
    Send a success notification email with processing summary
    """
    try:
        if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
            logging.warning("Email configuration incomplete. Skipping success notification.")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ', '.join(RECIPIENT_EMAILS)  # Join multiple recipients
        msg['Subject'] = f'ADT Automation Success - {datetime.now().strftime("%Y-%m-%d")}'
        
        # Email body
        body = f"""
ADT Daily Report Automation - Success

The ADT report has been successfully processed.

Report Details:
- File: {report_filename}
- Processing Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S EST")}

Summary:
{summary_info}

Log file: {log_filename}

This is an automated message. Please do not reply.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        logging.info(f"Attempting to send success notification to {len(RECIPIENT_EMAILS)} recipient(s): {', '.join(RECIPIENT_EMAILS)}")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, text)  # Send to list of recipients
        server.quit()
        
        logging.info("Success notification email sent")
        return True
        
    except Exception as e:
        logging.error(f"Failed to send success notification email: {str(e)}")
        return False


def download_todays_report():
    """
    Connect to SFTP and download today's report from /Resi_New directory
    Returns the local file path if successful, None otherwise
    """
    try:
        # Create download directory if it doesn't exist
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        
        # Generate today's expected filename
        today = datetime.now().strftime('%Y%m%d')
        expected_filename = f'ADT_Athena_DLY_Lead_CallData_Direct_Agnts_{today}.csv'
        local_filepath = os.path.join(DOWNLOAD_DIR, expected_filename)
        
        logging.info("="*60)
        logging.info("Starting SFTP connection to ADT server")
        logging.info(f"Host: {SFTP_HOST}")
        logging.info(f"Port: {SFTP_PORT}")
        logging.info(f"Username: {SFTP_USERNAME}")
        logging.info(f"Directory: {SFTP_DIRECTORY}")
        logging.info(f"Looking for file: {expected_filename}")
        logging.info("="*60)
        
        # Configure connection options - disable host key checking
        # In production, you should verify the host key for security
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None  # Disable host key checking (not recommended for production)
        
        # Connect to SFTP server
        with pysftp.Connection(
            host=SFTP_HOST,
            username=SFTP_USERNAME,
            password=SFTP_PASSWORD,
            port=SFTP_PORT,
            cnopts=cnopts
        ) as sftp:
            logging.info("Successfully connected to SFTP server")
            
            # Navigate to the directory
            sftp.cwd(SFTP_DIRECTORY)
            logging.info(f"Changed directory to: {SFTP_DIRECTORY}")
            
            # List files in directory for debugging
            files = sftp.listdir()
            logging.info(f"Files in directory: {len(files)} total")
            
            # Look for today's file
            csv_files = [f for f in files if f.endswith('.csv')]
            logging.info(f"CSV files found: {csv_files}")
            
            if expected_filename in files:
                logging.info(f"Found today's report: {expected_filename}")
                
                # Download the file
                sftp.get(expected_filename, local_filepath)
                logging.info(f"Successfully downloaded to: {local_filepath}")
                
                # Verify file was downloaded
                if os.path.exists(local_filepath):
                    file_size = os.path.getsize(local_filepath)
                    logging.info(f"File size: {file_size} bytes")
                    return local_filepath
                else:
                    raise Exception("File download verification failed - file not found locally")
            else:
                # Try alternative date formats or most recent file
                logging.warning(f"Today's file {expected_filename} not found")
                
                # Try yesterday's file as fallback
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                yesterday_filename = f'ADT_Athena_DLY_Lead_CallData_Direct_Agnts_{yesterday}.csv'
                
                if yesterday_filename in files:
                    logging.info(f"Found yesterday's report instead: {yesterday_filename}")
                    local_filepath = os.path.join(DOWNLOAD_DIR, yesterday_filename)
                    sftp.get(yesterday_filename, local_filepath)
                    logging.info(f"Successfully downloaded to: {local_filepath}")
                    return local_filepath
                else:
                    raise Exception(f"Neither today's ({expected_filename}) nor yesterday's ({yesterday_filename}) report found on SFTP server. Available CSV files: {csv_files}")
    
    except Exception as e:
        error_msg = f"Failed to download report from SFTP: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        raise Exception(error_msg)


def main():
    """
    Main automation function
    """
    try:
        logging.info("="*60)
        logging.info("ADT DAILY AUTOMATION STARTED")
        logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
        logging.info("="*60)
        
        # Check if SFTP password is configured
        if not SFTP_PASSWORD:
            raise Exception("SFTP_PASSWORD not set in environment variables. Please configure .env file.")
        
        # Step 1: Download today's report from SFTP
        logging.info("\n[Step 1/2] Downloading today's report from SFTP...")
        report_filepath = download_todays_report()
        
        if not report_filepath:
            raise Exception("Failed to download report - no file path returned")
        
        logging.info(f"✓ Report downloaded successfully: {report_filepath}\n")
        
        # Step 2: Process the report and fire pixels
        logging.info("[Step 2/2] Processing report and firing pixels...")
        process_adt_report(report_filepath)
        logging.info("✓ Report processed successfully\n")
        
        # Success!
        logging.info("="*60)
        logging.info("ADT DAILY AUTOMATION COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        
        # Send success notification
        summary = f"Report: {os.path.basename(report_filepath)}\nCheck log file for detailed pixel firing results."
        send_success_email(os.path.basename(report_filepath), summary)
        
        return True
        
    except Exception as e:
        # Log the error
        error_msg = f"AUTOMATION FAILED: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        logging.error("="*60)
        logging.error(error_msg)
        logging.error("="*60)
        
        # Read log file content for email
        try:
            with open(log_filename, 'r') as f:
                log_content = f.read()
        except:
            log_content = "Could not read log file"
        
        # Send error notification email
        send_error_email(error_msg, log_content)
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

