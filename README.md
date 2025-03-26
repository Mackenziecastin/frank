# ADT Pixel Firing

This script processes ADT Athena daily lead call data and fires tracking pixels for qualifying sales.

## Requirements

- Python 3.7+
- pandas
- requests

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with the path to your ADT Athena report:

```bash
python adt_pixel_firing.py path/to/ADT_Athena_DLY_Lead_CallData_Direct_Agnts_[date].csv
```

## Data Processing Steps

1. Data Cleaning:
   - Filters out "Health" rows from Ln_of_Busn column
   - Filters out "US: Health" rows from DNIS_BUSN_SEG_CD column
   - Filters for yesterday's date in Sale_Date column
   - Filters for "WEB0021011" in Lead_DNIS column
   - Filters for "NEW" and "RESALE" values in Ordr_Type column

2. Pixel Firing:
   - Fires a pixel for each qualifying sale
   - Uses unique transaction IDs
   - Logs all activities and results

## Logging

The script creates a daily log file named `adt_pixel_firing_YYYYMMDD.log` that contains:
- Data processing information
- Number of qualifying sales
- Pixel firing results
- Any errors or issues encountered

## Error Handling

The script includes comprehensive error handling and logging for:
- File reading issues
- Data processing errors
- Pixel firing failures

All errors are logged to the daily log file. 