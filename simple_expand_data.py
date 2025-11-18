#!/usr/bin/env python3
"""
Simple Data Expansion - No Dependencies
"""

import csv
import random
from datetime import datetime, timedelta

def expand_csv_data(input_file, output_file, multiplier=5):
    """Expand CSV data with variations"""
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    print(f"Original: {len(rows)} rows")
    
    expanded_rows = []
    
    # Add original data
    expanded_rows.extend(rows)
    
    # Create variations
    for variant in range(multiplier - 1):
        for row in rows:
            new_row = row.copy()
            
            # Add small variations to OHLCV (±1-3%)
            for i in range(1, 6):  # Skip timestamp, modify OHLCV
                try:
                    value = float(new_row[i])
                    variation = random.uniform(0.97, 1.03)  # ±3%
                    new_row[i] = str(value * variation)
                except:
                    pass
            
            expanded_rows.append(new_row)
    
    # Write expanded data
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(expanded_rows)
    
    print(f"Expanded: {len(expanded_rows)} rows -> {output_file}")

def main():
    """Expand all data files"""
    
    print("🚀 Simple Data Expansion")
    print("="*30)
    
    files = [
        ('btc_data.csv', 'btc_expanded.csv'),
        ('eth_data.csv', 'eth_expanded.csv'),
        ('sol_data.csv', 'sol_expanded.csv')
    ]
    
    for input_file, output_file in files:
        try:
            expand_csv_data(input_file, output_file, multiplier=5)
        except Exception as e:
            print(f"❌ {input_file}: {e}")
    
    print("\n✅ Data expansion complete!")
    print("Now have ~36K rows per symbol for training")

if __name__ == "__main__":
    main()
