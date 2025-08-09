import pandas as pd
import numpy as np
from pathlib import Path
import json

def fix_dataset_metadata():
    """Fix the dataset metadata and target identification"""
    
    # Load the combined dataset
    df = pd.read_csv('data/processed/pirvision_combined.csv')
    
    print("="*70)
    print("FIXING DATASET METADATA")
    print("="*70)
    
    # Correct target identification
    target_col = 'Label'  # The actual target column
    
    # Get the correct class distribution
    print(f"\n✅ Correct Target Column: '{target_col}'")
    print(f"Class Distribution:")
    for val, count in df[target_col].value_counts().sort_index().items():
        print(f"  Class {val}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Identify PIR sensors correctly (PIR_1 to PIR_55)
    pir_sensors = [col for col in df.columns if 'PIR' in col]
    
    # Update metadata
    metadata = {
        'total_samples': len(df),
        'total_features': len(df.columns) - 1,  # Exclude target
        'target_column': target_col,
        'target_classes': sorted(df[target_col].unique().tolist()),
        'class_distribution': df[target_col].value_counts().to_dict(),
        'feature_groups': {
            'pir_sensors': len(pir_sensors),
            'temporal': 2,  # Date, Time
            'environmental': 1,  # Temperature_F
        },
        'pir_sensors': pir_sensors,
        'num_pir_sensors': len(pir_sensors),
        'prepared_date': '2025-08-08'
    }
    
    # Save corrected metadata
    with open('data/processed/dataset_metadata_fixed.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  • Total samples: {metadata['total_samples']:,}")
    print(f"  • Total features: {metadata['total_features']}")
    print(f"  • PIR sensors: {metadata['num_pir_sensors']}")
    print(f"  • Target classes: {metadata['target_classes']}")
    
    return df, metadata

if __name__ == "__main__":
    df, metadata = fix_dataset_metadata()