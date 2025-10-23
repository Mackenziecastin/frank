# ADT Daily Report Automation Setup Guide

This guide will help you set up the automated daily ADT report processing system.

## Overview

The automation system will:
1. Connect to ADT's SFTP server at 12:30 PM EST daily
2. Download today's report from `/Resi_New` directory
3. Process the report and fire pixels automatically
4. Email you if any errors occur

---

## Step 1: Install Required Dependencies

First, install the necessary Python packages:

```bash
pip install pysftp python-dotenv
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Step 2: Configure Email Notifications

### For Gmail Users:

1. **Enable 2-Factor Authentication** on your Google account (if not already enabled)
   - Go to: https://myaccount.google.com/security

2. **Create an App Password:**
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Windows Computer" (or Other)
   - Click "Generate"
   - Copy the 16-character password (this is your `SENDER_PASSWORD`)

### For Other Email Providers:

Update the SMTP settings in your `.env` file according to your provider:
- **Outlook/Office365:** `smtp.office365.com`, port `587`
- **Yahoo:** `smtp.mail.yahoo.com`, port `587`
- **Custom SMTP:** Contact your email provider for SMTP settings

---

## Step 3: Create Configuration File

1. **Copy the example configuration file:**
   ```bash
   copy env.example .env
   ```
   (On Mac/Linux use `cp env.example .env`)

2. **Edit the `.env` file** with your actual credentials:
   ```
   # SFTP Configuration (already filled in)
   SFTP_HOST=mfts.adt.com
   SFTP_USERNAME=directagent
   SFTP_PASSWORD=qM%1773M
   SFTP_PORT=22
   SFTP_DIRECTORY=/Resi_New

   # Email Configuration - UPDATE THESE
   SENDER_EMAIL=your-email@gmail.com
   SENDER_PASSWORD=your-gmail-app-password
   RECIPIENT_EMAIL=your-email@gmail.com
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   ```

3. **Important:** The `.env` file contains sensitive credentials. Make sure it's in `.gitignore` so it doesn't get committed to Git!

---

## Step 4: Test the Automation

Before scheduling, test the script manually:

```bash
python adt_automated_daily_run.py
```

This will:
- Connect to the SFTP server
- Download today's report (or yesterday's as fallback)
- Process it and fire pixels
- Send you a success or error email

**Check the output:**
- Look for "ADT DAILY AUTOMATION COMPLETED SUCCESSFULLY"
- Check your email for the notification
- Review the log file: `adt_automation_YYYYMMDD_HHMMSS.log`

---

## Step 5: Schedule the Automation

### Option A: Windows Task Scheduler (Recommended for Windows)

1. **Open Task Scheduler:**
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create a new task:**
   - Click "Create Basic Task" in the right panel
   - Name: `ADT Daily Report Automation`
   - Description: `Automatically downloads and processes ADT reports at 12:30 PM EST`

3. **Set the trigger:**
   - Trigger: Daily
   - Start time: 12:30 PM
   - Recur every: 1 day

4. **Set the action:**
   - Action: Start a program
   - Program/script: `python` (or full path: `C:\Python\python.exe`)
   - Add arguments: `adt_automated_daily_run.py`
   - Start in: `C:\Users\dtanudirjo\Documents\GitHub\frank` (your project directory)

5. **Adjust timezone:**
   - In the "Edit Trigger" dialog, ensure timezone is set to EST

6. **Configure additional settings:**
   - Check "Run whether user is logged on or not" (requires admin)
   - Check "Run with highest privileges"
   - Under "Conditions" tab, uncheck "Start the task only if the computer is on AC power"

7. **Test the scheduled task:**
   - Right-click the task and select "Run" to test it immediately

### Option B: Python Scheduler (Alternative)

If you prefer to keep everything in Python, I can create a scheduler script that runs continuously and triggers the automation daily.

### Option C: Cloud Hosting (For More Reliability)

For production use, consider hosting on:
- **AWS Lambda** with EventBridge (serverless)
- **Heroku Scheduler** (easy setup)
- **Azure Functions** with Timer trigger
- **Google Cloud Functions** with Cloud Scheduler

---

## Step 6: Monitor and Maintain

### Log Files
- Each run creates a log file: `adt_automation_YYYYMMDD_HHMMSS.log`
- Review logs regularly to ensure everything is working
- Consider setting up log rotation to prevent disk space issues

### Email Notifications
- **Success emails:** Summary of pixels fired (optional, currently configured)
- **Error emails:** Sent automatically when something goes wrong
- Check your email daily around 12:35 PM EST to confirm automation ran

### Troubleshooting
If you don't receive the report or pixels aren't firing:

1. **Check the log file** for detailed error messages
2. **Verify SFTP credentials** are still valid
3. **Check if report is available** on the FTP at the expected time
4. **Test manually** by running the script: `python adt_automated_daily_run.py`
5. **Check email spam folder** for error notifications

---

## Security Best Practices

1. **Never commit `.env` file** to Git (it's in `.gitignore`)
2. **Restrict file permissions** on `.env` to your user only
3. **Use App Passwords** instead of your main email password
4. **Rotate passwords** periodically
5. **Keep logs secure** as they may contain sensitive information

---

## Customization Options

### Change Run Time
Edit the scheduled task time if reports arrive earlier/later than expected.

### Add Success Notifications
Currently, success emails are sent. To disable them, comment out line 249 in `adt_automated_daily_run.py`:
```python
# send_success_email(os.path.basename(report_filepath), summary)
```

### Process Multiple Days
If you need to process multiple reports, you can run the script with a date argument:
```bash
python adt_pixel_firing.py ADT_Athena_DLY_Lead_CallData_Direct_Agnts_20251021.csv
```

---

## Support

If you encounter any issues:
1. Check the log files first
2. Verify your `.env` configuration
3. Test SFTP connection manually using FileZilla
4. Check your email configuration with a test email script

---

## Files Created

- `adt_automated_daily_run.py` - Main automation script
- `env.example` - Configuration template
- `AUTOMATION_SETUP.md` - This setup guide (you're reading it!)
- `.env` - Your actual configuration (create from env.example)
- `adt_reports/` - Directory where downloaded reports are saved (auto-created)
- `adt_automation_*.log` - Log files for each run

---

**You're all set!** 🚀

The automation should now run daily at 12:30 PM EST and email you if there are any issues.

