import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def create_payment_profile(df, segment_col=None, segment_value=None):
    """Create a payment timing profile from historical data"""
    if segment_col and segment_value is not None:
        data = df[df[segment_col] == segment_value]['payment_days'].dropna()
    else:
        data = df['payment_days'].dropna()
    
    profile = {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75),
        'percentiles': {p: data.quantile(p/100) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
        'n_observations': len(data),
        'raw_data': data.values
    }
    
    return profile

def save_payment_profiles(df, output_dir='payment_profiles'):
    """Create and save all payment profiles"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    profiles = {}
    
    # Overall profile
    print("Creating overall profile...")
    profiles['overall'] = create_payment_profile(df)
    
    # Segment columns to analyze
    segment_configs = [
        'is_delinquent',
        'is_seriously_delinquent',
        'is_revolver',
        'delinquency_severity',
        'utilization_category',
        'revolver_intensity',
        'is_frequent_payer',
        'is_full_payer',
        'is_underpayer'
    ]
    
    for segment_col in segment_configs:
        print(f"Creating profiles for: {segment_col}")
        for segment_value in df[segment_col].unique():
            if pd.notna(segment_value):
                profile = create_payment_profile(df, segment_col, segment_value)
                key = f'{segment_col}_{segment_value}'
                profiles[key] = profile
                print(f"  - {key}: {profile['n_observations']:,} observations")
    
    # Save to pickle
    output_file = Path(output_dir) / 'payment_profiles.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(profiles, f)
    
    print(f"\n{'='*70}")
    print(f"Saved {len(profiles)} payment profiles to: {output_file}")
    print(f"{'='*70}")
    
    return profiles

if __name__ == "__main__":
    # Load data
    file_path_master = r"t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\Clean Code\master_dataset_complete.parquet"
    df = pd.read_parquet(file_path_master)
    
    # Create and save profiles
    profiles = save_payment_profiles(df)