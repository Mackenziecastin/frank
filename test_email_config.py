#!/usr/bin/env python3
"""
Test Email Configuration Script

Use this to verify your email settings are working before running the full automation.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

# Parse recipient emails (handle comma-separated list)
RECIPIENT_EMAILS = [email.strip() for email in RECIPIENT_EMAIL.split(',')] if RECIPIENT_EMAIL else []

def test_email():
    """
    Send a test email to verify configuration
    """
    print("="*60)
    print("Testing Email Configuration")
    print("="*60)
    
    # Check if all required variables are set
    print(f"\nSMTP Server: {SMTP_SERVER}")
    print(f"SMTP Port: {SMTP_PORT}")
    print(f"Sender Email: {SENDER_EMAIL if SENDER_EMAIL else '❌ NOT SET'}")
    print(f"Recipient Email(s): {', '.join(RECIPIENT_EMAILS) if RECIPIENT_EMAILS else '❌ NOT SET'}")
    print(f"Sender Password: {'✓ SET' if SENDER_PASSWORD else '❌ NOT SET'}")
    
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        print("\n❌ ERROR: Email configuration is incomplete!")
        print("Please edit your .env file and set all email variables.")
        return False
    
    try:
        # Create test message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ', '.join(RECIPIENT_EMAILS)  # Join multiple recipients
        msg['Subject'] = f'ADT Automation - Email Test - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        
        body = f"""
This is a test email from the ADT automation system.

If you're reading this, your email configuration is working correctly! ✓

Configuration Details:
- SMTP Server: {SMTP_SERVER}
- SMTP Port: {SMTP_PORT}
- Sender: {SENDER_EMAIL}
- Recipients: {', '.join(RECIPIENT_EMAILS)}
- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

You can now run the full automation script with confidence.

Next steps:
1. Run: python adt_automated_daily_run.py (to test the full automation)
2. Schedule the task to run daily at 12:30 PM EST

This is a test message.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect and send
        print("\n📧 Connecting to SMTP server...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        
        print("🔐 Starting TLS encryption...")
        server.starttls()
        
        print("🔑 Logging in...")
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        print(f"📨 Sending test email to {len(RECIPIENT_EMAILS)} recipient(s)...")
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, text)  # Send to list of recipients
        
        print("👋 Closing connection...")
        server.quit()
        
        print("\n" + "="*60)
        print("✓ SUCCESS! Test email sent successfully!")
        print("="*60)
        print(f"\nCheck your inbox at:")
        for email in RECIPIENT_EMAILS:
            print(f"  - {email}")
        print("(Don't forget to check spam folder if you don't see it)")
        
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("\n" + "="*60)
        print("❌ AUTHENTICATION FAILED")
        print("="*60)
        print("\nYour email or password is incorrect.")
        print("\nIf using Gmail:")
        print("1. Make sure you're using an App Password, not your regular password")
        print("2. Create one at: https://myaccount.google.com/apppasswords")
        print("3. You need 2-Factor Authentication enabled first")
        return False
        
    except smtplib.SMTPException as e:
        print("\n" + "="*60)
        print("❌ SMTP ERROR")
        print("="*60)
        print(f"\nError: {str(e)}")
        print("\nCheck your SMTP server and port settings in .env file")
        return False
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ UNEXPECTED ERROR")
        print("="*60)
        print(f"\nError: {str(e)}")
        return False


if __name__ == "__main__":
    test_email()

