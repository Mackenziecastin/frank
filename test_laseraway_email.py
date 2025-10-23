#!/usr/bin/env python3
"""
Test script for LaserAway email configuration

This script tests:
1. Loading environment variables
2. Connecting to IMAP email
3. Searching for LaserAway report emails
4. Listing recent emails to help debug
"""

import os
import sys
import imaplib
import email
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
IMAP_SERVER = os.getenv('LASERAWAY_IMAP_SERVER', 'imap.gmail.com')
IMAP_PORT = int(os.getenv('LASERAWAY_IMAP_PORT', '993'))
IMAP_EMAIL = os.getenv('LASERAWAY_IMAP_EMAIL')
IMAP_PASSWORD = os.getenv('LASERAWAY_IMAP_PASSWORD')

REPORT_SENDER = os.getenv('LASERAWAY_REPORT_SENDER', '')
REPORT_SUBJECT = os.getenv('LASERAWAY_REPORT_SUBJECT', '')
REPORT_ATTACHMENT = os.getenv('LASERAWAY_REPORT_ATTACHMENT', '*.xlsx')

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def print_status(label, value):
    status = "[SET]" if value else "[NOT SET]"
    print(f"{label}: {status}")
    if value and label != "IMAP Password":
        print(f"  Value: {value}")

def main():
    print_header("LaserAway Email Configuration Test")
    
    print("\n1. Environment Variables:")
    print_status("IMAP Server", IMAP_SERVER)
    print_status("IMAP Port", str(IMAP_PORT))
    print_status("IMAP Email", IMAP_EMAIL)
    print_status("IMAP Password", IMAP_PASSWORD)
    print_status("Report Sender Pattern", REPORT_SENDER)
    print_status("Report Subject Pattern", REPORT_SUBJECT)
    print_status("Report Attachment Pattern", REPORT_ATTACHMENT)
    
    # Check if credentials are set
    if not IMAP_EMAIL or not IMAP_PASSWORD:
        print("\n[ERROR] Email credentials not configured!")
        print("Please update your .env file with:")
        print("  LASERAWAY_IMAP_EMAIL=debbie.tanudirjo@directagents.com")
        print("  LASERAWAY_IMAP_PASSWORD=your-app-password-here")
        return False
    
    # Test IMAP connection
    print_header("2. Testing IMAP Connection")
    
    try:
        print(f"Connecting to {IMAP_SERVER}:{IMAP_PORT}...")
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        
        print(f"Logging in as {IMAP_EMAIL}...")
        mail.login(IMAP_EMAIL, IMAP_PASSWORD)
        
        print("[SUCCESS] Connected to email successfully!")
        
        # Select inbox
        mail.select('inbox')
        
        # Get recent emails
        print_header("3. Recent Emails (Last 10)")
        
        # Search for emails from today and yesterday
        search_criteria = '(OR (SINCE "' + datetime.now().strftime('%d-%b-%Y') + '") (SINCE "' + (datetime.now()).strftime('%d-%b-%Y') + '"))'
        status, messages = mail.search(None, 'ALL')
        
        email_ids = messages[0].split()
        recent_emails = email_ids[-10:] if len(email_ids) > 10 else email_ids
        
        print(f"\nFound {len(email_ids)} total emails in inbox")
        print(f"Showing last {len(recent_emails)} emails:\n")
        
        for i, email_id in enumerate(reversed(recent_emails), 1):
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)
            
            subject = email_message['subject']
            sender = email_message['from']
            date = email_message['date']
            
            print(f"Email #{i}:")
            print(f"  From: {sender}")
            print(f"  Subject: {subject}")
            print(f"  Date: {date}")
            
            # Check for attachments
            attachments = []
            for part in email_message.walk():
                if part.get_content_maintype() == 'multipart':
                    continue
                if part.get('Content-Disposition') is None:
                    continue
                filename = part.get_filename()
                if filename:
                    attachments.append(filename)
            
            if attachments:
                print(f"  Attachments: {', '.join(attachments)}")
            else:
                print(f"  Attachments: None")
            
            # Check if this matches our criteria
            matches = []
            if REPORT_SENDER and REPORT_SENDER.lower() in sender.lower():
                matches.append("sender")
            if REPORT_SUBJECT and REPORT_SUBJECT.lower() in subject.lower():
                matches.append("subject")
            if attachments and REPORT_ATTACHMENT:
                attachment_pattern = REPORT_ATTACHMENT.replace('*', '').replace('.xlsx', '')
                for att in attachments:
                    if attachment_pattern and attachment_pattern.lower() in att.lower() and att.endswith('.xlsx'):
                        matches.append("attachment")
                        break
            
            if matches:
                print(f"  [MATCH] This email matches: {', '.join(matches)}")
            
            print()
        
        # Search specifically for LaserAway reports
        print_header("4. Searching for LaserAway Reports")
        
        if REPORT_SENDER:
            print(f"Searching for emails from: *{REPORT_SENDER}")
            status, messages = mail.search(None, f'(FROM "{REPORT_SENDER}")')
            sender_emails = messages[0].split()
            print(f"  Found {len(sender_emails)} email(s) matching sender pattern")
        
        if REPORT_SUBJECT:
            print(f"Searching for emails with subject: {REPORT_SUBJECT}")
            status, messages = mail.search(None, f'(SUBJECT "{REPORT_SUBJECT}")')
            subject_emails = messages[0].split()
            print(f"  Found {len(subject_emails)} email(s) matching subject pattern")
        
        # Search for today's emails
        today = datetime.now().strftime('%d-%b-%Y')
        print(f"\nSearching for emails from today ({today})...")
        status, messages = mail.search(None, f'(ON {today})')
        today_emails = messages[0].split()
        print(f"  Found {len(today_emails)} email(s) from today")
        
        mail.logout()
        
        print_header("Test Complete!")
        print("\n[SUCCESS] All tests passed!")
        print("\nNext steps:")
        print("1. Review the emails above to confirm the patterns match")
        print("2. If you see a matching email, the automation should work")
        print("3. Run the automation: python laseraway_email_automation.py")
        
        return True
        
    except imaplib.IMAP4.error as e:
        print(f"\n[ERROR] IMAP Authentication failed: {str(e)}")
        print("\nPossible solutions:")
        print("1. Make sure you're using a Gmail App Password (not your regular password)")
        print("2. Enable IMAP in Gmail settings")
        print("3. Check that the email address is correct")
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

