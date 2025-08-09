"""
Dataset Preparation Script - Combines and prepares the PIRvision datasets
Author: Ammar Tarek Khattab
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """Prepare and combine PIRvision datasets for analysis"""
    
    def __init__(self, data_dir='data/raw', output_dir='data/processed'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_combine(self):
        """Load and combine both dataset files"""
        
        # Load both datasets
        df1 = pd.read_csv(self.data_dir / 'pirvision_office_dataset1.csv')
        df2 = pd.read_csv(self.data_dir / 'pirvision_office_dataset2.csv')
        
        logger.info(f"Dataset 1 shape: {df1.shape}")
        logger.info(f"Dataset 2 shape: {df2.shape}")
        
        # Check if they have the same columns
        if list(df1.columns) == list(df2.columns):
            # Combine datasets
            df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)
            logger.info(f"✅ Datasets combined successfully!")
            logger.info(f"Combined shape: {df_combined.shape}")
        else:
            logger.warning("⚠️ Datasets have different columns. Using Dataset 1 only.")
            df_combined = df1
        
        return df_combined
    
    def identify_target_column(self, df):
        """Identify the target column"""
        
        # The last column is typically the target
        # Based on PIRvision dataset, it should be presence detection
        last_col = df.columns[-1]
        
        # Check if it's categorical (for classification)
        unique_values = df[last_col].nunique()
        
        logger.info(f"Identified target column: '{last_col}'")
        logger.info(f"Unique values in target: {unique_values}")
        logger.info(f"Target value distribution:\n{df[last_col].value_counts()}")
        
        return last_col
    
    def create_feature_groups(self, df):
        """Organize features into groups"""
        
        feature_groups = {
            'pir_sensors': [],
            'temporal': [],
            'environmental': [],
            'other': []
        }
        
        target_col = df.columns[-1]
        
        for col in df.columns[:-1]:  # Exclude target column
            col_lower = col.lower()
            
            if 'pir' in col_lower:
                feature_groups['pir_sensors'].append(col)
            elif any(t in col_lower for t in ['date', 'time', 'hour', 'day']):
                feature_groups['temporal'].append(col)
            elif any(t in col_lower for t in ['temp', 'humidity', 'light']):
                feature_groups['environmental'].append(col)
            else:
                feature_groups['other'].append(col)
        
        # Log feature groups
        for group, features in feature_groups.items():
            logger.info(f"{group.upper()}: {len(features)} features")
        
        return feature_groups, target_col
    
    def basic_preprocessing(self, df):
        """Basic preprocessing steps"""
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"Missing values found: {missing.sum()} total")
            # Simple forward fill for time series data
            df = df.fillna(method='ffill').fillna(method='bfill')
            logger.info("Missing values filled using forward/backward fill")
        
        # Remove duplicates if any
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        if df.shape[0] < initial_shape:
            logger.info(f"Removed {initial_shape - df.shape[0]} duplicate rows")
        
        return df
    
    def save_prepared_data(self, df, feature_groups, target_col):
        """Save the prepared dataset and metadata"""
        
        # Save combined dataset
        output_path = self.output_dir / 'pirvision_combined.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved combined dataset to: {output_path}")
        
        # Save metadata
        metadata = {
            'total_samples': len(df),
            'total_features': len(df.columns) - 1,
            'target_column': target_col,
            'target_classes': df[target_col].unique().tolist(),
            'feature_groups': {k: len(v) for k, v in feature_groups.items()},
            'pir_sensors': feature_groups['pir_sensors'][:10],  # First 10
            'prepared_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save metadata
        import json
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"📋 Saved metadata to: {self.output_dir / 'dataset_metadata.json'}")
        
        return output_path, metadata
    
    def create_train_test_split_indices(self, df, test_size=0.2):
        """Create train/test split indices for later use"""
        
        # For time series, we should use temporal split
        split_idx = int(len(df) * (1 - test_size))
        
        train_indices = list(range(split_idx))
        test_indices = list(range(split_idx, len(df)))
        
        # Save indices
        split_info = {
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'train_indices': train_indices[:100],  # Save first 100 for reference
            'test_indices': test_indices[:100],
            'split_method': 'temporal',
            'test_ratio': test_size
        }
        
        import json
        with open(self.output_dir / 'train_test_split.json', 'w') as f:
            json.dump(split_info, f, indent=4)
        
        logger.info(f"📊 Train/Test split: {len(train_indices)}/{len(test_indices)}")
        
        return train_indices, test_indices

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("PIRVISION FOG DATASET PREPARATION")
    print("="*70)
    
    # Initialize preparer
    preparer = DatasetPreparer()
    
    # Step 1: Load and combine datasets
    print("\n📁 Step 1: Loading and combining datasets...")
    df = preparer.load_and_combine()
    
    # Step 2: Identify target and features
    print("\n🎯 Step 2: Identifying features...")
    feature_groups, target_col = preparer.create_feature_groups(df)
    
    # Step 3: Basic preprocessing
    print("\n🔧 Step 3: Basic preprocessing...")
    df = preparer.basic_preprocessing(df)
    
    # Step 4: Save prepared data
    print("\n💾 Step 4: Saving prepared data...")
    output_path, metadata = preparer.save_prepared_data(df, feature_groups, target_col)
    
    # Step 5: Create train/test split indices
    print("\n✂️ Step 5: Creating train/test split...")
    train_idx, test_idx = preparer.create_train_test_split_indices(df)
    
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Dataset Summary:")
    print(f"  • Total samples: {metadata['total_samples']:,}")
    print(f"  • Total features: {metadata['total_features']}")
    print(f"  • Target classes: {metadata['target_classes']}")
    print(f"  • PIR sensors: {metadata['feature_groups']['pir_sensors']}")
    print(f"  • Output location: {output_path}")
    
    return df, metadata

if __name__ == "__main__":
    df, metadata = main()