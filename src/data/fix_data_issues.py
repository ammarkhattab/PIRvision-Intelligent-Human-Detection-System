"""
Fix critical data issues found during EDA
Author: Ammar Tarek Khattab
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_pir1_outliers(df, threshold=1000000):
    """Fix extreme outliers in PIR_1 sensor"""
    logger.info(f"Fixing PIR_1 outliers (values > {threshold})")
    
    # Count outliers
    outliers = df['PIR_1'] > threshold
    outlier_count = outliers.sum()
    logger.info(f"Found {outlier_count} outliers in PIR_1")
    
    if outlier_count > 0:
        # Replace with median of non-outlier values
        median_val = df.loc[~outliers, 'PIR_1'].median()
        df.loc[outliers, 'PIR_1'] = median_val
        logger.info(f"Replaced outliers with median: {median_val}")
    
    return df

def fix_temperature_class3(df):
    """Fix temperature values for Class 3 (all zeros issue)"""
    logger.info("Fixing temperature for Class 3")
    
    # Check current state
    class3_temps = df[df['Label'] == 3]['Temperature_F']
    logger.info(f"Class 3 temperature mean: {class3_temps.mean()}")
    
    if class3_temps.mean() == 0:
        # Use overall median temperature as replacement
        median_temp = df[df['Label'] != 3]['Temperature_F'].median()
        df.loc[df['Label'] == 3, 'Temperature_F'] = median_temp
        logger.info(f"Replaced Class 3 temperatures with median: {median_temp}")
    
    return df

def remove_highly_correlated_features(df, threshold=0.95):
    """Remove highly correlated PIR sensors to reduce redundancy"""
    logger.info(f"Removing features with correlation > {threshold}")
    
    pir_cols = [col for col in df.columns if 'PIR' in col]
    corr_matrix = df[pir_cols].corr().abs()
    
    # Find features to drop
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > threshold)]
    
    logger.info(f"Dropping {len(to_drop)} highly correlated features")
    df_cleaned = df.drop(columns=to_drop)
    
    return df_cleaned, to_drop

def main():
    # Load data
    df = pd.read_csv('data/processed/pirvision_combined.csv')
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Apply fixes
    df = fix_pir1_outliers(df)
    df = fix_temperature_class3(df)
    
    # Optional: Remove highly correlated features
    # df, dropped_cols = remove_highly_correlated_features(df, threshold=0.95)
    
    # Save cleaned data
    output_path = 'data/processed/pirvision_cleaned.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    
    # Verify fixes
    logger.info("\nVerification:")
    logger.info(f"PIR_1 max value: {df['PIR_1'].max()}")
    logger.info(f"Class 3 temperature mean: {df[df['Label']==3]['Temperature_F'].mean()}")
    
    return df

if __name__ == "__main__":
    df_cleaned = main()