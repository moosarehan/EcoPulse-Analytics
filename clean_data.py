import pandas as pd
import glob
import os

DATA_DIR = "air_quality_data"
OUTPUT_FILE = "urban_air_quality_2025_master.parquet"

def consolidate_and_clean():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    if not all_files:
        print("No data files found to consolidate.")
        return
    
    print(f"Consolidating {len(all_files)} files...")
    
    dfs = []
    for f in all_files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    master_df = pd.concat(dfs, ignore_index=True)
    
    print("Performing initial cleaning...")
    # Convert timestamp to datetime
    master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
    
    # Pivot parameters to columns for easier analysis (NodeID, Timestamp, PM25, PM10, etc.)
    # We use pivot_table to handle any potential duplicate entries
    pivot_df = master_df.pivot_table(
        index=['location_id', 'location_name', 'timestamp'],
        columns='parameter',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Handle missing values (e.g., forward fill within each location)
    pivot_df = pivot_df.groupby('location_id', group_keys=False).apply(lambda x: x.sort_values('timestamp').ffill())
    
    # Save the cleaned master dataset
    pivot_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Master dataset saved to {OUTPUT_FILE}")
    print(f"Final shape: {pivot_df.shape}")
    
    # Summary info
    print("\nMissing values per parameter:")
    print(pivot_df.isnull().sum())

if __name__ == "__main__":
    consolidate_and_clean()
