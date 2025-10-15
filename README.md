# LaserAway Revshare Pixel Firing

This script fires tracking pixels for Schemathics on LaserAway based on Net Sales revenue share calculations.

## Features

- Filters data for `affiliate_directagent_subid1 = 42865`
- Processes sales within a specified date range
- Calculates revenue share using formula: `Net Sales / 1.75`
- Fires pixels with proper date formatting and revenue amounts
- Comprehensive logging and error handling

## Requirements

- Python 3.7+
- pandas==2.2.0
- requests>=2.25.0
- chardet>=4.0.0 (optional, for better encoding detection)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python laseraway_revshare_pixel_firing.py <report_file.csv> <start_date> <end_date>
```

### Example

```bash
python laseraway_revshare_pixel_firing.py laseraway_report.csv 2024-06-01 2024-06-30
```

**Note:** The script supports both CSV and XLSX file formats.

## Required File Columns

Your CSV or XLSX file must contain these columns:
- `affiliate_directagent_subid1` - Filtered for value "42865"
- `Purchased` - Date column for purchase dates
- `Net Sales` - Revenue amount for calculations

## Pixel Details

- **URL**: https://trkstar.com/m.ashx
- **Parameters**:
  - `o=32067` (Organization ID)
  - `e=865` (Event ID)
  - `f=pb` (Format)
  - `t=TRANSACTION_ID` (Unique transaction ID)
  - `pubid=42865` (Publisher ID)
  - `campid=96548` (Campaign ID)
  - `dt=YYYY-MM-DDTHH:MM:SS+00:00` (Purchase date in ISO format)
  - `p=REVENUE_AMOUNT` (Calculated revenue share amount)

## Revenue Calculation

The script calculates the revenue share amount using:
```
Revenue Amount = Net Sales / 1.75
```

The result is formatted to 2 decimal places (e.g., 1024.19).

## Logging

The script creates detailed logs in `laseraway_revshare_pixel_firing_YYYYMMDD.log` including:
- Processing statistics
- Pixel firing results
- Error messages
- Summary of total revenue processed

## Error Handling

- Automatic encoding detection for CSV files
- Excel file support with openpyxl
- Graceful handling of missing or invalid data
- Comprehensive error logging
- Validation of required columns