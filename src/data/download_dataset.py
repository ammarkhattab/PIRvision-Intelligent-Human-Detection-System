"""
Dataset Download Script for PIRvision_FoG
Author: Ammar Tarek Khattab
Course: CSCI417 Machine Intelligence
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_download.log'),
        logging.StreamHandler()
    ]
)

class DatasetDownloader:
    """Download and organize PIRvision_FoG dataset from UCI Repository"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def download_from_uci(self):
        """Download dataset using ucimlrepo"""
        try:
            self.logger.info("Starting dataset download from UCI...")
            
            # Fetch dataset (ID: 1101 for PIRvision_FoG)
            pirvision_fog = fetch_ucirepo(id=1101)
            
            # Extract features and targets
            X = pirvision_fog.data.features
            y = pirvision_fog.data.targets
            
            # Get metadata
            metadata = pirvision_fog.metadata
            variables = pirvision_fog.variables
            
            self.logger.info(f"Dataset shape: {X.shape}")
            self.logger.info(f"Number of features: {X.shape[1]}")
            self.logger.info(f"Number of samples: {X.shape[0]}")
            
            return X, y, metadata, variables
            
        except Exception as e:
            self.logger.error(f"Error downloading from UCI: {e}")
            self.logger.info("Trying alternative download method...")
            return self.download_manual()
    
    def download_manual(self):
        """Alternative manual download method"""
        # Provide manual download instructions
        print("""
        Manual Download Instructions:
        1. Visit: https://archive.ics.uci.edu/dataset/1101/pirvision+fog+presence+detection
        2. Download the following files:
           - pirvision_office_dataset1.csv
           - pirvision_office_dataset2.csv
        3. Place them in the 'data/raw/' directory
        """)
        return None, None, None, None
    
    def save_dataset(self, X, y, metadata, variables):
        """Save dataset to CSV files"""
        if X is not None and y is not None:
            # Combine features and target
            full_data = pd.concat([X, y], axis=1)
            
            # Save to CSV
            output_path = self.data_dir / 'pirvision_fog_complete.csv'
            full_data.to_csv(output_path, index=False)
            self.logger.info(f"Dataset saved to {output_path}")
            
            # Save metadata
            with open(self.data_dir / 'metadata.txt', 'w') as f:
                f.write(str(metadata))
            
            # Save variable information
            if variables is not None:
                variables.to_csv(self.data_dir / 'variables_info.csv', index=False)
            
            return output_path
        return None
    
    def verify_dataset(self, data_path):
        """Verify downloaded dataset"""
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            print("\n" + "="*50)
            print("DATASET VERIFICATION")
            print("="*50)
            print(f"✓ Total samples: {len(df)}")
            print(f"✓ Total features: {len(df.columns) - 1}")
            print(f"✓ Target column: {df.columns[-1]}")
            print(f"✓ Missing values: {df.isnull().sum().sum()}")
            print(f"✓ Data types: {df.dtypes.value_counts().to_dict()}")
            print("="*50)
            
            return True
        return False

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize downloader
    downloader = DatasetDownloader()
    
    # Download dataset
    X, y, metadata, variables = downloader.download_from_uci()
    
    # Save dataset
    data_path = downloader.save_dataset(X, y, metadata, variables)
    
    # Verify dataset