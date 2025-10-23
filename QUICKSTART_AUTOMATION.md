# Quick Start Guide - ADT Automation

## 🚀 Get Started in 5 Minutes

### 1️⃣ Install Dependencies
```bash
pip install pysftp python-dotenv
```

### 2️⃣ Create Configuration File
```bash
copy env.example .env
```
Then edit `.env` and add your email settings:
```
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-gmail-app-password
RECIPIENT_EMAIL=your-email@gmail.com
```

**Gmail Users:** Get your App Password here: https://myaccount.google.com/apppasswords

### 3️⃣ Test Email Configuration
```bash
python test_email_config.py
```
✓ You should receive a test email

### 4️⃣ Test Full Automation
```bash
python adt_automated_daily_run.py
```
✓ This will download today's report and process it

### 5️⃣ Schedule Daily Runs

**Windows Task Scheduler:**
1. Open Task Scheduler (`Win + R` → `taskschd.msc`)
2. Create Basic Task → Name it "ADT Daily Automation"
3. Trigger: Daily at 12:30 PM
4. Action: Start a program
   - Program: `python`
   - Arguments: `adt_automated_daily_run.py`
   - Start in: `C:\Users\dtanudirjo\Documents\GitHub\frank`
5. Save and test by right-clicking → Run

---

## 📧 What You'll Receive

**Success Email (daily at ~12:35 PM EST):**
- Confirms report was downloaded and processed
- Shows summary of pixels fired

**Error Email (only if something breaks):**
- Details of what went wrong
- Full log file content
- You can then investigate and fix the issue

---

## 🔧 Troubleshooting

**"Email authentication failed"**
→ Use Gmail App Password, not your regular password

**"SFTP connection failed"**
→ Check that SFTP credentials are still valid in `.env`

**"File not found"**
→ Report may not be available yet, check FTP manually

**Still stuck?**
→ Check the detailed guide: `AUTOMATION_SETUP.md`

---

## 📁 Files Overview

- `adt_automated_daily_run.py` - Main automation script
- `adt_pixel_firing.py` - Pixel firing logic (already exists)
- `.env` - Your credentials (create from `env.example`)
- `env.example` - Template with SFTP details
- `test_email_config.py` - Test your email settings
- `AUTOMATION_SETUP.md` - Detailed setup instructions
- `QUICKSTART_AUTOMATION.md` - This file!

---

**That's it! You're automated! 🎉**

The system will now run daily at 12:30 PM EST and email you if there are any issues.

