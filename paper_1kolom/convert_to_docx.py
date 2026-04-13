#!/usr/bin/env python3
"""
Script to convert LaTeX to DOCX with proper citations and images.
Converts PDF images to PNG for better Word compatibility.
"""

import re
import os
import subprocess
import shutil

work_dir = r"C:\Users\ADMIN\Documents\project\disertasis3\SIMULASI_EXPERIMENT\seminarproposal\experiments\github_upload\paper_1kolom"
os.chdir(work_dir)

# Read the LaTeX file
input_file = "1kolomByzantine_Robust_Federated_Learning_with_Adaptive_Aggregation_and_Blockchain.tex"
output_file = "1kolomByzantine_Robust_Federated_Learning_with_Adaptive_Aggregation_and_Blockchain_FIXED.tex"
docx_file = "1kolomByzantine_Robust_Federated_Learning_with_Adaptive_Aggregation_and_Blockchain.docx"

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract bibitem keys and their numbers
bibitem_pattern = r'\\bibitem\{([^}]+)\}'
bibitems = re.findall(bibitem_pattern, content)
cite_map = {key: str(i+1) for i, key in enumerate(bibitems)}

print(f"Found {len(cite_map)} bibliography entries")

# Function to replace \cite{key1,key2,...} with [num1, num2, ...]
def replace_cite(match):
    keys = match.group(1).split(',')
    nums = []
    for key in keys:
        key = key.strip()
        if key in cite_map:
            nums.append(cite_map[key])
        else:
            nums.append('?')
            print(f"Warning: Citation key '{key}' not found")
    return '[' + ', '.join(nums) + ']'

# Replace all \cite{...} with [num, ...]
content = re.sub(r'\\cite\{([^}]+)\}', replace_cite, content)

# Replace PDF with PNG for images (Word handles PNG better)
def fix_image_path(match):
    options = match.group(1) if match.group(1) else ""
    filename = match.group(2)
    
    # If it's a PDF, try to use PNG version if exists
    if filename.endswith('.pdf'):
        png_filename = filename.replace('.pdf', '.png')
        # Check both in current dir and visualizations
        if os.path.exists(png_filename):
            filename = png_filename
            print(f"  Using PNG: {filename}")
        elif os.path.exists(png_filename.replace('visualizations/', '')):
            filename = png_filename.replace('visualizations/', '')
            print(f"  Using PNG (no path): {filename}")
        else:
            # Keep PDF but warn
            print(f"  Warning: No PNG for {filename}")
    
    return f'\\includegraphics{options}{{{filename}}}'

print("\nProcessing images:")
content = re.sub(r'\\includegraphics(\[[^\]]*\])?\{([^}]+)\}', fix_image_path, content)

# Write the processed file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nProcessed LaTeX saved to: {output_file}")

# List available images
print("\nAvailable images:")
for img in os.listdir('.'):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        print(f"  {img}")
print("In visualizations/:")
if os.path.exists('visualizations'):
    for img in os.listdir('visualizations'):
        if img.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            print(f"  visualizations/{img}")

# Now convert to DOCX using pandoc with better options
print("\nConverting to DOCX with pandoc...")

# Use semicolon for Windows path separator in resource-path
cmd = [
    'pandoc',
    output_file,
    '-f', 'latex',
    '-t', 'docx',
    '-o', docx_file,
    '--standalone',
    '--resource-path=.;visualizations',
]

print(f"Command: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    file_size = os.path.getsize(docx_file) / 1024
    print(f"\nSuccessfully created: {docx_file}")
    print(f"File size: {file_size:.1f} KB")
else:
    print(f"Error: {result.stderr}")
    print(f"Output: {result.stdout}")

print("\nDone!")
