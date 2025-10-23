# LaserAway Email Automation Setup Guide

## 📧 Overview

This automation automatically:
1. **Connects to your email inbox** to find the daily LaserAway report
2. **Downloads the Excel attachment** from the email
3. **Processes the report** and fires pixels based on yesterday's sales
4. **Sends email notifications** about success/failure

---

## 🔧 Prerequisites

1. **Python 3.8+** installed
2. **Required packages** installed (see below)
3. **Email account** that receives LaserAway reports
4. **Email credentials** (App Password if using Gmail)

---

## 📋 Step 1: Install Required Packages

```bash
pip install python-dotenv imaplib-utf7
```

All other required packages should already be installed from the ADT automation setup.

---

## 🔑 Step 2: Configure Email Settings

### If the LaserAway reports come to a **Gmail account**:

1. **Enable IMAP access:**
   - Go to Gmail Settings → Forwarding and POP/IMAP
   - Enable IMAP access

2. **Create an App Password:**
   - Go to https://myaccount.google.com/apppasswords
   - Generate an app-specific password for "Mail"
   - Save this password (you'll need it in Step 3)

### If the reports come to an **Outlook/Office365 account**:

1. **IMAP Server:** `outlook.office365.com`
2. **IMAP Port:** `993`
3. **Password:** Use your regular email password (or app password if 2FA is enabled)

---

## ⚙️ Step 3: Update .env File

Open your `.env` file and add these settings:

```env
# LaserAway Email Automation Configuration
# Email account that RECEIVES the LaserAway reports
LASERAWAY_IMAP_SERVER=imap.gmail.com
LASERAWAY_IMAP_PORT=993
LASERAWAY_IMAP_EMAIL=your-email@directagents.com
LASERAWAY_IMAP_PASSWORD=your-app-password-here

# LaserAway Report Email Identification
LASERAWAY_REPORT_SENDER=sender@laseraway.com
LASERAWAY_REPORT_SUBJECT=LaserAway Report
LASERAWAY_REPORT_ATTACHMENT=*.xlsx

# Date range for processing (days ago from today)
LASERAWAY_START_DAYS_AGO=1
LASERAWAY_END_DAYS_AGO=1
```

### ✏️ Fill in these values:

1. **LASERAWAY_IMAP_EMAIL**: The email address that receives LaserAway reports
2. **LASERAWAY_IMAP_PASSWORD**: Your email password or App Password
3. **LASERAWAY_REPORT_SENDER**: Email address that sends the LaserAway report (optional - helps filter)
4. **LASERAWAY_REPORT_SUBJECT**: Part of the subject line to identify the report (e.g., "LaserAway", "Daily Report")
5. **LASERAWAY_REPORT_ATTACHMENT**: Filename pattern (e.g., `LaserAway*.xlsx`, `*.xlsx`, `Report*.xlsx`)

> **📌 Note:** `LASERAWAY_START_DAYS_AGO=1` and `LASERAWAY_END_DAYS_AGO=1` means it will process **yesterday's sales** by default.

---

## 🧪 Step 4: Test the Automation

### Manual Test:

```bash
python laseraway_email_automation.py
```

**What should happen:**
1. ✅ Connects to your email
2. ✅ Finds today's LaserAway report email
3. ✅ Downloads the attachment to `laseraway_reports/` folder
4. ✅ Processes the report and fires pixels
5. ✅ Sends success/error notification email

### Check the output:
- **Console:** Shows real-time progress
- **Log file:** `laseraway_automation_YYYYMMDD_HHMMSS.log` contains detailed logs
- **Email:** You should receive a notification email

---

## ⏰ Step 5: Schedule the Automation

### Windows (Task Scheduler):

1. **Open Task Scheduler**
   - Search for "Task Scheduler" in Start menu

2. **Create Basic Task**
   - Click "Create Basic Task"
   - Name: `LaserAway Email Automation`
   - Description: `Daily automation to download and process LaserAway reports`

3. **Set Trigger**
   - Trigger: Daily
   - Start time: **[TIME WHEN REPORT ARRIVES + 30 minutes]**
     - Example: If reports arrive at 9:00 AM, set this to 9:30 AM
   - Recur: Every 1 day

4. **Set Action**
   - Action: Start a program
   - Program/script: `C:\Users\dtanudirjo\AppData\Local\Programs\Python\Python312\python.exe`
   - Add arguments: `laseraway_email_automation.py`
   - Start in: `C:\Users\dtanudirjo\Documents\GitHub\frank`

5. **Configure Additional Settings**
   - Right-click the task → Properties
   - General tab:
     - ✅ "Run whether user is logged on or not"
     - ✅ "Run with highest privileges"
   - Settings tab:
     - ✅ "Run task as soon as possible after a scheduled start is missed"

6. **Save and Test**
   - Right-click the task → Run
   - Check if it executes successfully

---

## 📧 Email Notification Details

### Success Email:
```
Subject: LaserAway Automation - Success

LaserAway report processed successfully!

Report: LaserAway_Report_20251023.xlsx
Date Range: 2025-10-22 to 2025-10-22
Timestamp: 2025-10-23 09:30:15

Check the log file for detailed pixel firing results: laseraway_automation_20251023_093015.log
```

### Error Email:
```
Subject: LaserAway Automation Error - 2025-10-23

AUTOMATION FAILED: No LaserAway report email found for today

Full traceback:
[Detailed error information]

[Full log content for debugging]
```

---

## 🔍 Troubleshooting

### ❌ "Email credentials not configured"
- **Solution:** Make sure `.env` file has `LASERAWAY_IMAP_EMAIL` and `LASERAWAY_IMAP_PASSWORD` set

### ❌ "No matching LaserAway report email found today"
- **Solution:** 
  - Check if the report email actually arrived today
  - Verify `LASERAWAY_REPORT_SENDER` and `LASERAWAY_REPORT_SUBJECT` match the actual email
  - Try making the subject pattern more generic (e.g., just "LaserAway" instead of "LaserAway Daily Report")

### ❌ "No matching attachment found in email"
- **Solution:**
  - Check `LASERAWAY_REPORT_ATTACHMENT` pattern
  - Try using `*.xlsx` to match any Excel file

### ❌ Authentication errors
- **Gmail:** Make sure you're using an App Password, not your regular password
- **Outlook:** Verify IMAP is enabled and credentials are correct

### ❌ Task doesn't run in Task Scheduler
- **Solution:**
  - Check "History" tab in Task Scheduler for error details
  - Verify Python path is correct: `C:\Users\dtanudirjo\AppData\Local\Programs\Python\Python312\python.exe`
  - Verify "Start in" directory: `C:\Users\dtanudirjo\Documents\GitHub\frank`
  - Ensure `.env` file is in the "Start in" directory

---

## 📁 File Structure

```
frank/
├── laseraway_email_automation.py     # Main automation script
├── laseraway_revshare_pixel_firing.py  # Processing logic
├── .env                               # Your credentials (NEVER commit!)
├── laseraway_reports/                 # Downloaded reports (auto-created)
│   └── LaserAway_Report_20251023.xlsx
└── laseraway_automation_*.log         # Log files
```

---

## 🔒 Security Notes

1. **Never commit `.env` file** - It contains passwords!
2. **Downloaded reports contain PII** - They're automatically ignored by Git
3. **Log files may contain sensitive data** - Review before sharing
4. **Use App Passwords** for Gmail instead of your main password

---

## ℹ️ Need Help?

If you encounter issues:
1. Check the log file: `laseraway_automation_YYYYMMDD_HHMMSS.log`
2. Run manually first to test: `python laseraway_email_automation.py`
3. Verify your `.env` settings match the actual email format
4. Check email notifications for error details

---

## 📝 Customization

### Change date range:
In `.env`, adjust:
```env
LASERAWAY_START_DAYS_AGO=7  # Process last 7 days
LASERAWAY_END_DAYS_AGO=1    # Up to yesterday
```

### Process specific dates:
Modify `laseraway_email_automation.py` to use hardcoded dates instead of relative dates.

---

**🎉 Once set up, this runs completely automatically every day!**

