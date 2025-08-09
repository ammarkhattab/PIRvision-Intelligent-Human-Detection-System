"""
Dataset Exploration Script for PIRvision_FoG
Author: Ammar Tarek Khattab
Course: CSCI417 Machine Intelligence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

class DatasetExplorer:
    """Explore and understand the PIRvision dataset structure"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_datasets(self):
        """Load both CSV files and explore their structure"""
        
        print("="*70)
        print("PIRVISION FOG DATASET EXPLORATION")
        print("="*70)
        
        # Load dataset 1
        dataset1_path = self.data_dir / 'pirvision_office_dataset1.csv'
        if dataset1_path.exists():
            self.datasets['dataset1'] = pd.read_csv(dataset1_path)
            print(f"\n✅ Dataset 1 loaded from: {dataset1_path}")
            self.explore_single_dataset(self.datasets['dataset1'], "Dataset 1")
        else:
            print(f"❌ Dataset 1 not found at: {dataset1_path}")
            
        # Load dataset 2
        dataset2_path = self.data_dir / 'pirvision_office_dataset2.csv'
        if dataset2_path.exists():
            self.datasets['dataset2'] = pd.read_csv(dataset2_path)
            print(f"\n✅ Dataset 2 loaded from: {dataset2_path}")
            self.explore_single_dataset(self.datasets['dataset2'], "Dataset 2")
        else:
            print(f"❌ Dataset 2 not found at: {dataset2_path}")
            
        return self.datasets
    
    def explore_single_dataset(self, df, name):
        """Explore a single dataset"""
        print(f"\n{'='*50}")
        print(f"{name} Analysis")
        print('='*50)
        
        # Basic info
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.shape[1]}")
        print(f"Rows: {df.shape[0]:,}")
        
        # Column names
        print(f"\nFirst 10 columns:")
        for i, col in enumerate(df.columns[:10], 1):
            print(f"  {i:2}. {col}")
        
        # Check for PIR columns
        pir_cols = [col for col in df.columns if 'PIR' in col.upper()]
        print(f"\nPIR Sensor Columns: {len(pir_cols)}")
        if pir_cols:
            print(f"  PIR columns: {pir_cols[:5]}...")  # Show first 5
        
        # Check for target/label columns
        potential_targets = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['target', 'label', 'class', 'presence', 'occupancy']
        )]
        print(f"\nPotential Target Columns: {potential_targets}")
        
        # Check last column (often the target)
        last_col = df.columns[-1]
        print(f"\nLast column (potential target): '{last_col}'")
        if df[last_col].dtype in ['int64', 'float64']:
            print(f"  Unique values: {sorted(df[last_col].unique())}")
            print(f"  Value counts:")
            for val, count in df[last_col].value_counts().items():
                print(f"    Class {val}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Data types summary
        print(f"\nData Types Summary:")
        print(df.dtypes.value_counts())
        
        # Missing values
        missing = df.isnull().sum().sum()
        print(f"\nMissing Values: {missing}")
        
        # Sample data
        print(f"\nFirst 3 rows preview:")
        print(df.head(3))
        
        return df
    
    def compare_datasets(self):
        """Compare the two datasets"""
        if len(self.datasets) == 2:
            print("\n" + "="*70)
            print("DATASET COMPARISON")
            print("="*70)
            
            df1 = self.datasets['dataset1']
            df2 = self.datasets['dataset2']
            
            # Check if columns are the same
            same_columns = list(df1.columns) == list(df2.columns)
            print(f"Same columns: {same_columns}")
            
            if same_columns:
                print("✅ Both datasets have identical column structure")
                print("→ Can be combined vertically (concatenated)")
            else:
                print("⚠️ Datasets have different columns")
                cols1_only = set(df1.columns) - set(df2.columns)
                cols2_only = set(df2.columns) - set(df1.columns)
                if cols1_only:
                    print(f"  Columns only in Dataset 1: {cols1_only}")
                if cols2_only:
                    print(f"  Columns only in Dataset 2: {cols2_only}")
            
            # Size comparison
            print(f"\nSize Comparison:")
            print(f"  Dataset 1: {len(df1):,} rows")
            print(f"  Dataset 2: {len(df2):,} rows")
            print(f"  Total: {len(df1) + len(df2):,} rows")
            
    def identify_features(self):
        """Identify different types of features"""
        if not self.datasets:
            return
            
        df = list(self.datasets.values())[0]  # Use first dataset
        
        print("\n" + "="*70)
        print("FEATURE IDENTIFICATION")
        print("="*70)
        
        features = {
            'pir_sensors': [],
            'temporal': [],
            'temperature': [],
            'target': None,
            'other': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # PIR sensors (usually PIR_1, PIR_2, etc.)
            if 'pir' in col_lower:
                features['pir_sensors'].append(col)
            # Temporal features
            elif any(t in col_lower for t in ['date', 'time', 'hour', 'minute', 'second']):
                features['temporal'].append(col)
            # Temperature
            elif any(t in col_lower for t in ['temp', 'temperature']):
                features['temperature'].append(col)
            # Target (last column or specific keywords)
            elif col == df.columns[-1] or any(t in col_lower for t in ['target', 'label', 'class']):
                features['target'] = col
            else:
                features['other'].append(col)
        
        # Print findings
        print(f"📡 PIR Sensors ({len(features['pir_sensors'])} features):")
        if features['pir_sensors']:
            for i, feat in enumerate(features['pir_sensors'][:6], 1):
                print(f"   {i}. {feat}")
            if len(features['pir_sensors']) > 6:
                print(f"   ... and {len(features['pir_sensors'])-6} more")
        
        print(f"\n⏰ Temporal Features ({len(features['temporal'])} features):")
        for feat in features['temporal']:
            print(f"   • {feat}")
        
        print(f"\n🌡️ Temperature Features ({len(features['temperature'])} features):")
        for feat in features['temperature']:
            print(f"   • {feat}")
        
        print(f"\n🎯 Target Variable: {features['target']}")
        
        print(f"\n📊 Other Features ({len(features['other'])} features)")
        
        return features

if __name__ == "__main__":
    # Initialize explorer
    explorer = DatasetExplorer()
    
    # Load and explore datasets
    datasets = explorer.load_datasets()
    
    # Compare datasets if both loaded
    if len(datasets) == 2:
        explorer.compare_datasets()
    
    # Identify feature types
    features = explorer.identify_features()
    
    print("\n" + "="*70)
    print("✅ Exploration Complete!")
    print("="*70)