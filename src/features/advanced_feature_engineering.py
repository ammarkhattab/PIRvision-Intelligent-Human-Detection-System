import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for PIR sensor data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def create_statistical_features(self, df, pir_cols):
        """Enhanced statistical features"""
        print("Creating advanced statistical features...")
        
        # Basic statistics
        df['pir_mean'] = df[pir_cols].mean(axis=1)
        df['pir_std'] = df[pir_cols].std(axis=1)
        df['pir_max'] = df[pir_cols].max(axis=1)
        df['pir_min'] = df[pir_cols].min(axis=1)
        df['pir_range'] = df['pir_max'] - df['pir_min']
        df['pir_skew'] = df[pir_cols].skew(axis=1)
        df['pir_kurtosis'] = df[pir_cols].kurtosis(axis=1)
        
        # Quartiles and IQR
        df['pir_q25'] = df[pir_cols].quantile(0.25, axis=1)
        df['pir_q75'] = df[pir_cols].quantile(0.75, axis=1)
        df['pir_iqr'] = df['pir_q75'] - df['pir_q25']
        
        # Coefficient of variation
        df['pir_cv'] = df['pir_std'] / (df['pir_mean'] + 1e-10)
        
        # Peak to average ratio
        df['pir_par'] = df['pir_max'] / (df['pir_mean'] + 1e-10)
        
        return df
    
    def create_activation_patterns(self, df, pir_cols):
        """Detect activation patterns across sensors"""
        print("Creating activation pattern features...")
        
        # Dynamic threshold based on percentile
        thresholds = df[pir_cols].quantile(0.75)
        
        # Count activated sensors
        activated = df[pir_cols] > thresholds
        df['sensors_activated'] = activated.sum(axis=1)
        df['activation_ratio'] = df['sensors_activated'] / len(pir_cols)
        
        # Consecutive activations (spatial coherence)
        for i in range(len(pir_cols)-1):
            if i < 5:  # First 5 pairs
                col_name = f'consecutive_activation_{i}'
                df[col_name] = (activated.iloc[:, i] & activated.iloc[:, i+1]).astype(int)
        
        # Activation clusters
        df['activation_cluster_size'] = activated.apply(
            lambda x: self._find_largest_cluster(x.values), axis=1
        )
        
        return df
    
    def _find_largest_cluster(self, arr):
        """Find the largest consecutive cluster of True values"""
        clusters = []
        current = 0
        for val in arr:
            if val:
                current += 1
            else:
                if current > 0:
                    clusters.append(current)
                current = 0
        if current > 0:
            clusters.append(current)
        return max(clusters) if clusters else 0
    
    def create_frequency_features(self, df, pir_cols):
        """Advanced frequency domain features"""
        print("Creating frequency domain features...")
        
        # FFT features
        fft_features = []
        for idx, row in df[pir_cols].iterrows():
            fft_vals = np.fft.fft(row.values)
            fft_abs = np.abs(fft_vals)
            
            features = {
                'fft_mean_magnitude': np.mean(fft_abs),
                'fft_max_magnitude': np.max(fft_abs),
                'fft_std_magnitude': np.std(fft_abs),
                'fft_dominant_freq': np.argmax(fft_abs[1:len(fft_abs)//2]) + 1,
                'fft_energy': np.sum(fft_abs**2),
                'fft_entropy': stats.entropy(fft_abs + 1e-10)
            }
            fft_features.append(features)
        
        fft_df = pd.DataFrame(fft_features)
        df = pd.concat([df, fft_df], axis=1)
        
        return df
    
    def create_pca_features(self, df, pir_cols, n_components=10):
        """Create PCA features from PIR sensors"""
        print(f"Creating PCA features (n_components={n_components})...")
        
        # Standardize PIR data
        pir_scaled = self.scaler.fit_transform(df[pir_cols])
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(pir_scaled)
        
        # Add PCA features
        for i in range(n_components):
            df[f'pca_{i+1}'] = pca_features[:, i]
        
        # Print explained variance
        explained_var = self.pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_var.sum():.3f}")
        print(f"First 5 components: {explained_var[:5]}")
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between key variables"""
        print("Creating interaction features...")
        
        # Temperature interactions
        df['temp_x_pir_mean'] = df['Temperature_F'] * df['pir_mean']
        df['temp_x_activation'] = df['Temperature_F'] * df['activation_ratio']
        
        # Temporal interactions
        if 'Hour' in df.columns:
            df['hour_x_pir_mean'] = df['Hour'] * df['pir_mean']
            df['is_night'] = ((df['Hour'] >= 20) | (df['Hour'] <= 6)).astype(int)
            df['night_x_activation'] = df['is_night'] * df['activation_ratio']
        
        return df
    
    def create_all_features(self, df):
        """Apply all feature engineering methods"""
        
        # Identify PIR columns
        pir_cols = [col for col in df.columns if 'PIR' in col]
        print(f"\nEngineering features for {len(pir_cols)} PIR sensors...")
        
        # Original feature count
        original_features = len(df.columns)
        
        # Apply all engineering methods
        df = self.create_statistical_features(df, pir_cols)
        df = self.create_activation_patterns(df, pir_cols)
        df = self.create_frequency_features(df, pir_cols)
        df = self.create_pca_features(df, pir_cols, n_components=15)
        df = self.create_interaction_features(df)
        
        # Add datetime features if not present
        if 'DateTime' not in df.columns and 'Date' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df['Hour'] = df['DateTime'].dt.hour
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        print(f"\n✅ Feature engineering complete!")
        print(f"Original features: {original_features}")
        print(f"Engineered features: {len(df.columns)}")
        print(f"New features created: {len(df.columns) - original_features}")
        
        return df

def main():
    # Load cleaned data
    df = pd.read_csv('data/processed/pirvision_cleaned.csv')
    print(f"Loaded dataset: {df.shape}")
    
    # Initialize engineer
    engineer = AdvancedFeatureEngineer()
    
    # Create features
    df_engineered = engineer.create_all_features(df)
    
    # Save engineered dataset
    output_path = 'data/processed/pirvision_engineered.csv'
    df_engineered.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved engineered dataset to: {output_path}")
    
    # Display feature groups
    feature_groups = {
        'Original PIR': len([c for c in df_engineered.columns if 'PIR_' in c]),
        'Statistical': len([c for c in df_engineered.columns if any(x in c for x in ['mean', 'std', 'max', 'min', 'skew'])]),
        'Activation': len([c for c in df_engineered.columns if 'activation' in c.lower()]),
        'Frequency': len([c for c in df_engineered.columns if 'fft' in c]),
        'PCA': len([c for c in df_engineered.columns if 'pca' in c]),
        'Interaction': len([c for c in df_engineered.columns if '_x_' in c])
    }
    
    print("\nFeature Groups:")
    for group, count in feature_groups.items():
        print(f"  {group}: {count}")
    
    return df_engineered

if __name__ == "__main__":
    df_final = main()