# 🚀 LaserAway Email Automation - Quick Start

## ⚡ What You Need to Provide

Before the automation can run, I need the following information from you:

### 📧 Email Details:

1. **What email address receives the LaserAway reports?**
   - Example: `reports@directagents.com` or `debbie.tanudirjo@directagents.com`

2. **What email provider?**
   - Gmail
   - Outlook/Office365
   - Other (specify)

3. **What is the subject line of the report emails?**
   - Example: "LaserAway Daily Report" or "Daily Performance Report"
   - Can be partial match, e.g., just "LaserAway"

4. **Who sends the report emails?**
   - Example: `noreply@laseraway.com` or `reports@majorrocket.com`

5. **What's the attachment filename?**
   - Example: `LaserAway_Report_10-23-2025.xlsx`
   - Pattern: `LaserAway_Report_*.xlsx` or just `*.xlsx`

### ⏰ Schedule Details:

6. **What time do the report emails typically arrive?**
   - Example: "Around 9:00 AM EST"

7. **What sales date should we process?**
   - Default: Yesterday's sales (most common)
   - Or: Specific date range

---

## ✅ 3-Minute Setup (After You Provide Details Above)

### Step 1: Update `.env` file
I'll provide you with the exact values to add based on your answers above.

### Step 2: Test it manually
```bash
python laseraway_email_automation.py
```

### Step 3: Schedule it (Windows Task Scheduler)
- Set to run daily at [TIME + 30 minutes after report arrives]
- Done! It runs automatically every day.

---

## 📨 What Happens Automatically

**Every day at the scheduled time:**

1. 🔍 Script connects to your email inbox
2. 📥 Finds and downloads today's LaserAway report
3. 🔥 Processes yesterday's sales and fires pixels
4. ✅ Sends you a success email with the summary
5. 🪵 Creates a detailed log file

**If something goes wrong:**
- 🚨 You get an error email immediately
- 📋 Log file shows exactly what happened
- 🔧 You can fix and re-run manually

---

## 🎯 Next Steps

**👉 Please provide the information listed at the top of this document so I can configure the automation for you!**

Once you provide the details:
1. I'll update your `.env` file with the correct settings
2. You can test it immediately
3. Then schedule it and you're done!

---

## 💡 Questions to Answer

**Copy and paste this, fill in your answers:**

```
1. Email address that receives LaserAway reports: _________________

2. Email provider (Gmail/Outlook/Other): _________________

3. Subject line of report emails: _________________

4. Sender email address: _________________

5. Attachment filename or pattern: _________________

6. What time do reports arrive (EST): _________________

7. Process which sales? (Yesterday/Last 7 days/Other): _________________
```

**Once you provide these, we can have the automation running in minutes! 🚀**

