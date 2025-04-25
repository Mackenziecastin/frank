#!/usr/bin/env python
# This script truncates the bob_analysis.py file to match the version at commit d0a08e6f94da7944e1b2f29f01cefa0998245e38

import os

# Path to the bob_analysis.py file
file_path = "pages/bob_analysis.py"

# Read the current file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the end of the analyze_records_by_pid function
marker = "                st.success(f\"PID {pid} metrics match expected values\")"
if marker in content:
    # Get the content up to the end of the analyze_records_by_pid function
    truncated_content = content.split(marker)[0] + marker + "\n\nif __name__ == \"__main__\":\n    show_bob_analysis()"
    
    # Write the truncated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(truncated_content)
    
    print(f"Successfully truncated {file_path} to match commit d0a08e6f94da7944e1b2f29f01cefa0998245e38")
else:
    print(f"Could not find the marker in {file_path}. No changes made.") 