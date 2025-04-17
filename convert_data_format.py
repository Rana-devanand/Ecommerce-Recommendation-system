#!/usr/bin/env python
"""
Utility script to convert between CSV and JSON formats for the recommendation system.
This can be helpful during the migration from CSV to JSON format.
"""

import os
import argparse
import pandas as pd
from modules.utils.data_loader import load_product_data, save_product_data

def main():
    parser = argparse.ArgumentParser(description='Convert data between CSV and JSON formats')
    parser.add_argument('input_file', help='Path to input file (.csv or .json)')
    parser.add_argument('output_file', help='Path to output file (.csv or .json)')
    
    args = parser.parse_args()
    
    # Check file extensions
    input_ext = os.path.splitext(args.input_file)[1].lower()
    output_ext = os.path.splitext(args.output_file)[1].lower()
    
    if input_ext not in ['.csv', '.json']:
        print(f"Error: Input file must be either .csv or .json, got {input_ext}")
        return
        
    if output_ext not in ['.csv', '.json']:
        print(f"Error: Output file must be either .csv or .json, got {output_ext}")
        return
    
    if input_ext == output_ext:
        print(f"Warning: Input and output formats are the same ({input_ext}). Will copy the data.")
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_product_data(args.input_file)
    
    if data.empty:
        print("Error: Failed to load data or data is empty")
        return
    
    print(f"Loaded {len(data)} records")
    
    # Save data in the output format
    print(f"Saving data to {args.output_file}...")
    format = 'csv' if output_ext == '.csv' else 'json'
    success = save_product_data(data, args.output_file, format)
    
    if success:
        print(f"Successfully converted {args.input_file} to {args.output_file}")
    else:
        print(f"Failed to save data to {args.output_file}")

if __name__ == "__main__":
    main()