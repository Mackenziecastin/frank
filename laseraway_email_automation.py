#!/usr/bin/env python3
"""
LaserAway Email Automation Script

This script automatically:
1. Connects to email inbox via IMAP
2. Finds today's LaserAway report email
3. Downloads the attachment
4. Processes it and fires pixels
5. Sends email notification with results

Schedule this to run daily after the report email arrives
"""

import os
import sys
import email
import imaplib
import logging
import smtplib
import traceback
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import re

# Import the existing LaserAway processing logic
from laseraway_revshare_pixel_firing import process_laseraway_report

# Load environment variables
load_dotenv()

# Configure logging
log_filename = f"laseraway_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Email configuration for IMAP (receiving)
IMAP_SERVER = os.getenv('LASERAWAY_IMAP_SERVER', 'imap.gmail.com')
IMAP_PORT = int(os.getenv('LASERAWAY_IMAP_PORT', '993'))
IMAP_EMAIL = os.getenv('LASERAWAY_IMAP_EMAIL')
IMAP_PASSWORD = os.getenv('LASERAWAY_IMAP_PASSWORD')

# Email configuration for SMTP (sending notifications)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

# Parse recipient emails (handle comma-separated list)
RECIPIENT_EMAILS = [email.strip() for email in RECIPIENT_EMAIL.split(',')] if RECIPIENT_EMAIL else []

# LaserAway report email settings
REPORT_SENDER_EMAIL = os.getenv('LASERAWAY_REPORT_SENDER', '')  # Email address that sends the report
REPORT_SUBJECT_PATTERN = os.getenv('LASERAWAY_REPORT_SUBJECT', '')  # Subject line pattern to match
REPORT_ATTACHMENT_PATTERN = os.getenv('LASERAWAY_REPORT_ATTACHMENT', '*.xlsx')  # Filename pattern

# Date range for processing (default: yesterday)
DEFAULT_START_DAYS_AGO = int(os.getenv('LASERAWAY_START_DAYS_AGO', '1'))
DEFAULT_END_DAYS_AGO = int(os.getenv('LASERAWAY_END_DAYS_AGO', '1'))

# Local directory to store downloaded reports
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), 'laseraway_reports')


def send_notification_email(subject, body):
    """
    Send notification email about processing results
    """
    try:
        if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
            logging.warning("Email configuration incomplete. Skipping notification.")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ', '.join(RECIPIENT_EMAILS)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        logging.info(f"Sending notification to {len(RECIPIENT_EMAILS)} recipient(s)")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        server.quit()
        
        logging.info("Notification email sent successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to send notification email: {str(e)}")
        return False


def connect_to_email():
    """
    Connect to email inbox via IMAP
    """
    try:
        if not all([IMAP_EMAIL, IMAP_PASSWORD]):
            raise Exception("Email credentials not configured. Please set LASERAWAY_IMAP_EMAIL and LASERAWAY_IMAP_PASSWORD in .env")
        
        logging.info(f"Connecting to {IMAP_SERVER}...")
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        
        logging.info(f"Logging in as {IMAP_EMAIL}...")
        mail.login(IMAP_EMAIL, IMAP_PASSWORD)
        
        logging.info("Successfully connected to email")
        return mail
        
    except Exception as e:
        logging.error(f"Failed to connect to email: {str(e)}")
        raise


def find_todays_report_email(mail):
    """
    Search for today's LaserAway report email
    """
    try:
        mail.select('inbox')
        
        # Build search criteria
        today = datetime.now().strftime('%d-%b-%Y')
        
        # Search for emails from today
        search_criteria = f'(ON {today})'
        
        # If sender is specified, add it to criteria
        if REPORT_SENDER_EMAIL:
            search_criteria = f'(FROM "{REPORT_SENDER_EMAIL}" ON {today})'
        
        logging.info(f"Searching for emails with criteria: {search_criteria}")
        status, messages = mail.search(None, search_criteria)
        
        if status != 'OK':
            raise Exception("Failed to search emails")
        
        email_ids = messages[0].split()
        logging.info(f"Found {len(email_ids)} email(s) from today")
        
        # Search through emails for the LaserAway report
        for email_id in reversed(email_ids):  # Start with most recent
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            
            if status != 'OK':
                continue
            
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)
            
            subject = email_message['subject']
            sender = email_message['from']
            
            logging.info(f"Checking email: From='{sender}', Subject='{subject}'")
            
            # Check if this matches our report criteria
            subject_matches = True
            if REPORT_SUBJECT_PATTERN:
                subject_matches = re.search(REPORT_SUBJECT_PATTERN, subject, re.IGNORECASE) is not None
            
            if subject_matches:
                logging.info(f"✓ Found matching email: {subject}")
                return email_message
        
        logging.warning("No matching LaserAway report email found today")
        return None
        
    except Exception as e:
        logging.error(f"Error searching for report email: {str(e)}")
        raise


def download_attachment(email_message):
    """
    Download the report attachment from the email
    """
    try:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        
        for part in email_message.walk():
            # Check if this part is an attachment
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            
            filename = part.get_filename()
            
            if filename:
                # Check if filename matches our pattern
                pattern_matches = True
                if REPORT_ATTACHMENT_PATTERN and REPORT_ATTACHMENT_PATTERN != '*.*':
                    pattern = REPORT_ATTACHMENT_PATTERN.replace('*', '.*')
                    pattern_matches = re.search(pattern, filename, re.IGNORECASE) is not None
                
                if pattern_matches:
                    # Save the attachment
                    filepath = os.path.join(DOWNLOAD_DIR, filename)
                    
                    logging.info(f"Downloading attachment: {filename}")
                    with open(filepath, 'wb') as f:
                        f.write(part.get_payload(decode=True))
                    
                    file_size = os.path.getsize(filepath)
                    logging.info(f"✓ Downloaded {filename} ({file_size} bytes)")
                    
                    return filepath
        
        raise Exception("No matching attachment found in email")
        
    except Exception as e:
        logging.error(f"Error downloading attachment: {str(e)}")
        raise


def main():
    """
    Main automation function
    """
    try:
        logging.info("="*60)
        logging.info("LASERAWAY EMAIL AUTOMATION STARTED")
        logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*60)
        
        # Step 1: Connect to email
        logging.info("\n[Step 1/3] Connecting to email inbox...")
        mail = connect_to_email()
        
        # Step 2: Find and download report
        logging.info("\n[Step 2/3] Finding today's LaserAway report...")
        email_message = find_todays_report_email(mail)
        
        if not email_message:
            error_msg = "No LaserAway report email found for today"
            logging.warning(error_msg)
            send_notification_email(
                "LaserAway Automation - No Report Found",
                f"{error_msg}\n\nTimestamp: {datetime.now()}\n\nPlease check if the report was sent."
            )
            mail.logout()
            return False
        
        report_filepath = download_attachment(email_message)
        mail.logout()
        
        logging.info(f"✓ Report downloaded: {report_filepath}\n")
        
        # Step 3: Process the report
        logging.info("[Step 3/3] Processing report and firing pixels...")
        
        # Calculate date range (default: yesterday)
        end_date = datetime.now() - timedelta(days=DEFAULT_END_DAYS_AGO)
        start_date = datetime.now() - timedelta(days=DEFAULT_START_DAYS_AGO)
        
        logging.info(f"Processing date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Process the report and get summary data
        summary_data = process_laseraway_report(report_filepath, start_date, end_date)
        
        logging.info("✓ Report processed successfully\n")
        
        # Success!
        logging.info("="*60)
        logging.info("LASERAWAY AUTOMATION COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        
        # Build detailed summary email
        summary_text = f"""LaserAway report processed successfully!

Report: {os.path.basename(report_filepath)}
Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{"="*50}
PIXELS FIRED
{"="*50}

"""
        
        if summary_data and summary_data['total_pixels'] > 0:
            summary_text += f"""✅ Total Pixels Fired: {summary_data['successful_pixels']} out of {summary_data['total_pixels']}

{"="*50}
REVENUE DETAILS
{"="*50}

💰 Total Net Sales Processed: ${summary_data['total_revenue']:,.2f}
📊 Revenue Share Amount (Net Sales / 1.75): ${summary_data['revenue_share']:,.2f}
💵 Average Revenue per Pixel: ${summary_data['revenue_share'] / summary_data['successful_pixels']:,.2f}

"""
        else:
            summary_text += """⚠️  No pixels fired - no qualifying sales found for the specified date range.

"""
        
        summary_text += f"""{"="*50}

Check the log file for detailed pixel firing results: {log_filename}
"""
        
        send_notification_email("LaserAway Automation - Success", summary_text)
        
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
        
        # Send error notification
        send_notification_email(
            f"LaserAway Automation Error - {datetime.now().strftime('%Y-%m-%d')}",
            f"{error_msg}\n\n{'='*60}\nFull Log:\n{'='*60}\n\n{log_content}"
        )
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

