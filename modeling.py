import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import json

def run_modeling(input_file="urban_air_quality_2025_cleaned.parquet",
                 pca_output_file="urban_air_quality_2025_pca.parquet",
                 pca_meta_file="pca_metadata.json",
                 region_stats_file="region_stats.csv"):
                 
    print(f"Loading cleaned data from {input_file}...")
    df = pd.read_parquet(input_file)
    
    # --- TASK 1: Dimensionality Reduction (PCA) ---
    print("Running PCA on standardized variables...")
    std_params = ['pm25_std', 'pm10_std', 'no2_std', 'o3_std', 'temperature_std', 'relativehumidity_std']
    
    pca = PCA(n_components=2)
    # Fit PCA on the standardized variables
    pca_result = pca.fit_transform(df[std_params])
    
    df['PC1'] = pca_result[:, 0]
    df['PC2'] = pca_result[:, 1]
    
    explained_variance = pca.explained_variance_ratio_.tolist()
    
    # Component loadings (how much each original variable contributes to each PC)
    # Shape of pca.components_ is (n_components, n_features)
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=['PM2.5', 'PM10', 'NO2', 'O3', 'Temperature', 'Relative Humidity']
    )
    
    # Save PCA enriched dataset
    df.to_parquet(pca_output_file, index=False)
    print(f"Saved PCA enriched data to {pca_output_file}")
    
    # Save PCA metadata
    pca_meta = {
        'explained_variance': explained_variance,
        'loadings': loadings.to_dict()
    }
    with open(pca_meta_file, 'w') as f:
        json.dump(pca_meta, f, indent=4)
    print(f"Saved PCA metadata to {pca_meta_file}")

    # --- TASK 4 preparations ---
    # Need Population Density vs Mean PM2.5 per region for Small Multiples (Scatter plots)
    print("Computing regional statistics for Visual Integrity Audit...")
    region_stats = df.groupby('Region').agg(
        Mean_PM25=('pm25', 'mean'),
        Population_Density=('Population_Density', 'first')
    ).reset_index()
    
    region_stats.to_csv(region_stats_file, index=False)
    print(f"Saved regional statistics to {region_stats_file}")

if __name__ == "__main__":
    run_modeling()
