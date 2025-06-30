import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA
from scipy.stats import f_oneway, skew, kurtosis
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

class BiomedicalPainDataset(Dataset):
    def __init__(
        self,
        signals_df: pd.DataFrame,
        pain_df: pd.DataFrame,
        window_duration: float,
        step_size: float,
        signal_columns: List[str],
        bandpass_freqs: Tuple[float, float] = (0.5, 40),
        freq_bands: List[Tuple[float, float]] = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)],
        resample_freq: int = 100,
        include_time_domain_features: bool = True,
    ):
        # Convert and sort timestamps
        self.signals_df = signals_df.copy()
        self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp']) 
        self.signals_df.sort_values('timestamp', inplace=True)
        
        self.pain_df = pain_df.copy()
        self.pain_df['timestamp'] = pd.to_datetime(self.pain_df['timestamp'])
        self.pain_df = self.pain_df.dropna(subset=['PainLevel'])
        self.pain_df.sort_values('timestamp', inplace=True)
        
        # Store parameters
        self.window_duration = window_duration
        self.step_size = step_size
        self.signal_columns = signal_columns
        self.bandpass_freqs = bandpass_freqs
        self.freq_bands = freq_bands
        self.resample_freq = resample_freq
        self.n_samples_per_window = int(window_duration * resample_freq)
        self.include_time_domain_features = include_time_domain_features
        self.feature_names = self._generate_feature_names()
        
        # Precompute valid window start times
        self.window_starts = self._precompute_window_starts()
        
    def _generate_feature_names(self):
        """Generate descriptive feature names"""
        names = []
        
        # Time-domain features
        if self.include_time_domain_features:
            time_features = ['mean', 'var', 'rms', 'min', 'max', 'range', 
                            'skew', 'kurt', 'zero_cross', 'slope', 'peak_freq']
            for col in self.signal_columns:
                for feat in time_features:
                    names.append(f"{col}_{feat}")
        
        # Frequency-domain features
        for col in self.signal_columns:
            for band in self.freq_bands:
                names.append(f"{col}_{band[0]}-{band[1]}Hz")
        
        return names
        
    def _precompute_window_starts(self):
        min_time = self.signals_df['timestamp'].min()
        max_time = self.signals_df['timestamp'].max() - pd.Timedelta(seconds=self.window_duration)
        
        window_starts = []
        current = min_time
        while current <= max_time:
            window_starts.append(current)
            current += pd.Timedelta(seconds=self.step_size)
        return window_starts

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        window_start = self.window_starts[idx]
        window_end = window_start + pd.Timedelta(seconds=self.window_duration)
        
        # Process signals
        signal_features = self._process_signals(window_start, window_end)
        
        # Compute pain label and convert to category
        pain_label = self._compute_pain_label(window_start, window_end)
        pain_category = self._categorize_pain(pain_label)
        
        return signal_features, pain_category, pain_label
    
    def _categorize_pain(self, pain_level: float) -> int:
        """Categorize pain level into 0: no pain, 1: low pain, 2: high pain"""
        if pain_level == 0:
            return 0  # No pain
        elif 1 <= pain_level <= 5:
            return 1  # Low pain
        elif pain_level > 5:
            return 2  # High pain
        else:
            # Handle values between 0 and 1
            return 1 if pain_level > 0 else 0

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def _compute_time_domain_features(self, signal):
        """Compute various time-domain statistical features"""
        features = []
        
        # Basic statistics
        features.append(np.mean(signal))           # mean
        features.append(np.var(signal))            # variance
        features.append(np.sqrt(np.mean(signal**2)))  # RMS
        features.append(np.min(signal))            # min
        features.append(np.max(signal))            # max
        features.append(np.max(signal) - np.min(signal))  # range
        
        # Higher-order statistics
        features.append(skew(signal))              # skewness
        features.append(kurtosis(signal))          # kurtosis
        
        # Signal characteristics
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        features.append(len(zero_crossings))       # zero crossings
        
        # Linear trend (slope)
        if len(signal) > 1:
            x = np.arange(len(signal))
            slope = np.polyfit(x, signal, 1)[0]
            features.append(slope)
        else:
            features.append(0)
        
        # Peak frequency
        if len(signal) > 1:
            freqs = np.fft.rfftfreq(len(signal), 1/self.resample_freq)
            fft_vals = np.abs(np.fft.rfft(signal))
            if len(freqs) > 0:
                features.append(freqs[np.argmax(fft_vals)])
            else:
                features.append(0)
        else:
            features.append(0)
        
        return np.array(features)

    def _process_signals(self, start, end):
        window_signals = self.signals_df[
            (self.signals_df['timestamp'] >= start) & 
            (self.signals_df['timestamp'] < end)
        ]
        
        t_start = start.timestamp()
        t_end = end.timestamp()
        t_grid = np.linspace(t_start, t_end, self.n_samples_per_window, endpoint=False)
        
        processed_signals = []
        for col in self.signal_columns:
            col_data = window_signals[['timestamp', col]].dropna()
            
            if col_data.empty:
                # If no data, create zeros array
                processed_signals.append(np.zeros(self.n_samples_per_window))
                continue
                
            t_valid = col_data['timestamp'].astype(np.int64).values / 1e9
            y_valid = col_data[col].values
            
            if len(t_valid) < 2:
                processed_signals.append(np.zeros(self.n_samples_per_window))
                continue
                
            # Linear interpolation to common time grid
            interp_fn = interp1d(t_valid, y_valid, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_resampled = interp_fn(t_grid)
            
            # Apply bandpass filter
            if self.bandpass_freqs:
                y_resampled = self._butter_bandpass_filter(
                    y_resampled, 
                    self.bandpass_freqs[0], 
                    self.bandpass_freqs[1],
                    self.resample_freq
                )
                
            processed_signals.append(y_resampled)
        
        signal_matrix = np.vstack(processed_signals)
        
        # Extract features for each signal
        all_features = []
        
        for i, signal in enumerate(signal_matrix):
            signal_features = []
            
            # Skip if signal is all zeros
            if np.all(signal == 0):
                # Create zeros for all features
                if self.include_time_domain_features:
                    signal_features.extend(np.zeros(11))  # 11 time-domain features
                signal_features.extend(np.zeros(len(self.freq_bands)))  # frequency bands
                all_features.append(signal_features)
                continue
                
            # Time-domain features
            if self.include_time_domain_features:
                time_features = self._compute_time_domain_features(signal)
                signal_features.extend(time_features)
            
            # Frequency-domain features
            freqs, psd = welch(
                signal, 
                fs=self.resample_freq, 
                nperseg=min(256, len(signal)))
            
            band_powers = []
            for low, high in self.freq_bands:
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_power = np.trapezoid(psd[mask], freqs[mask])
                else:
                    band_power = 0.0
                band_powers.append(band_power)
            
            signal_features.extend(band_powers)
            all_features.append(signal_features)
        
        # Flatten all features
        features = np.concatenate(all_features)
        return torch.tensor(features, dtype=torch.float32)

    def _compute_pain_label(self, start, end):
        prev_reports = self.pain_df[self.pain_df['timestamp'] <= end]
        if prev_reports.empty:
            return 0.0
        
        next_report = self.pain_df[self.pain_df['timestamp'] > end].head(1)
        all_reports = pd.concat([prev_reports, next_report]).sort_values('timestamp')
        
        segments = []
        for i in range(len(all_reports) - 1):
            seg_start = all_reports.iloc[i]['timestamp']
            seg_end = all_reports.iloc[i+1]['timestamp']
            pain_level = all_reports.iloc[i]['PainLevel']
            
            seg_start = max(seg_start, start)
            seg_end = min(seg_end, end)
            
            if seg_start < seg_end:
                duration = (seg_end - seg_start).total_seconds()
                segments.append((duration, pain_level))
        
        total_duration = 0
        weighted_sum = 0
        for duration, pain in segments:
            total_duration += duration
            weighted_sum += duration * pain
            
        return weighted_sum / total_duration if total_duration > 0 else 0.0

    def perform_anova(self, max_samples_per_class=1000, plot_results=True):
        """Perform ANOVA on signal features across pain categories"""
        # Collect data for each pain category
        categories = {0: [], 1: [], 2: []}
        category_counts = {0: 0, 1: 0, 2: 0}
        
        # We'll also collect pain categories for visualization
        all_features = []
        all_categories = []
        
        for i in range(len(self)):
            features, pain_cat, _ = self[i]
            features = features.numpy()
            
            if category_counts[pain_cat] < max_samples_per_class:
                categories[pain_cat].append(features)
                all_features.append(features)
                all_categories.append(pain_cat)
                category_counts[pain_cat] += 1
                
            # Stop if we have enough samples for all categories
            if all(count >= max_samples_per_class for count in category_counts.values()):
                break
        
        # Prepare data for ANOVA
        data_by_category = [np.array(categories[i]) for i in range(3)]
        n_features = data_by_category[0].shape[1]
        all_features = np.array(all_features)
        all_categories = np.array(all_categories)
        
        # Perform ANOVA for each feature
        f_values = np.zeros(n_features)
        p_values = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            samples = [data[feature_idx] for data in data_by_category]
            f_val, p_val = f_oneway(*samples)
            f_values[feature_idx] = f_val
            p_values[feature_idx] = p_val
        
        # Apply Bonferroni correction
        bonferroni_threshold = 0.05 / n_features
        significant_features = p_values < bonferroni_threshold
        
        # Print results
        print("\n" + "="*80)
        print("ANOVA Results for Pain Categories")
        print("="*80)
        print(f"{'Feature':<40} {'F-value':>10} {'p-value':>15} {'Significant':>12}")
        print("-"*80)
        
        for i, (name, f_val, p_val) in enumerate(zip(self.feature_names, f_values, p_values)):
            sig_star = "***" if significant_features[i] else ""
            print(f"{name:<40} {f_val:>10.3f} {p_val:>15.5e} {sig_star:>12}")
        
        num_sig = np.sum(significant_features)
        print("\nSummary:")
        print(f"Total features: {n_features}")
        print(f"Significant features (p < {bonferroni_threshold:.5e}): {num_sig} ({num_sig/n_features:.1%})")
        print("="*80 + "\n")
        
        # Plot results
        if plot_results:
            self._plot_anova_results(f_values, p_values, significant_features, all_features, all_categories)
        
        return f_values, p_values, significant_features

    def _plot_anova_results(self, f_values, p_values, significant_features, all_features, all_categories):
        """Visualize ANOVA results with distribution plots"""
        plt.figure(figsize=(18, 12))
        
        # Volcano plot
        plt.subplot(2, 2, 1)
        plt.scatter(f_values, -np.log10(p_values), 
                    c=significant_features, cmap='viridis', alpha=0.7)
        plt.axhline(-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        plt.axhline(-np.log10(0.05/len(p_values)), color='g', linestyle='--', label='Bonferroni')
        plt.xlabel('F-value')
        plt.ylabel('-log10(p-value)')
        plt.title('Volcano Plot of ANOVA Results')
        plt.colorbar(label='Significant')
        plt.legend()
        
        # P-value distribution
        plt.subplot(2, 2, 2)
        plt.hist(p_values, bins=50, alpha=0.7)
        plt.axvline(0.05, color='r', linestyle='--', label='p=0.05')
        plt.axvline(0.05/len(p_values), color='g', linestyle='--', label='Bonferroni')
        plt.xlabel('p-value')
        plt.ylabel('Frequency')
        plt.title('Distribution of p-values')
        plt.yscale('log')
        plt.legend()
        
        # Top significant features
        plt.subplot(2, 2, 3)
        top_features = 20
        sorted_idx = np.argsort(f_values)[::-1]
        top_f = f_values[sorted_idx][:top_features]
        top_names = [self.feature_names[i] for i in sorted_idx[:top_features]]
        
        plt.barh(top_names, top_f, color='skyblue')
        plt.xlabel('F-value')
        plt.title(f'Top {top_features} Features by F-value')
        
        # Feature distributions for top feature
        if top_features > 0:
            plt.subplot(2, 2, 4)
            top_idx = sorted_idx[0]
            top_feature = all_features[:, top_idx]
            
            # Create a DataFrame for seaborn
            df = pd.DataFrame({
                'Feature Value': top_feature,
                'Pain Category': all_categories
            })
            
            # Map category codes to names
            category_names = {0: 'No Pain', 1: 'Low Pain', 2: 'High Pain'}
            df['Pain Category'] = df['Pain Category'].map(category_names)
            
            # Plot distributions
            sns.violinplot(x='Pain Category', y='Feature Value', data=df, 
                          palette='viridis', inner='quartile', cut=0)
            plt.title(f'Distribution of Top Feature: {self.feature_names[top_idx]}')
            plt.ylabel('Feature Value')
            plt.xlabel('')
        
        plt.tight_layout()
        plt.savefig('anova_results.png', dpi=300)
        plt.show()
        



#self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'], unit=unit_signals_ts) 
#self.signals_df['timestamp'] = self.signals_df['timestamp'] - np.timedelta64(time_difference, 'h')


class BaseSlidingWindowDataset(BaseTabularDataset):
    """
    Specialized for time-series: creates windows of past data to predict future.
    Returns (X_window, Y) where X_window is 2D tensor [T x features].
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        window_size: int,
        timestamp_col: str='timestamp',
        horizon: int = 1,
        step_size: int = 1,
        window_type: str = 'fixed',
        stride: int = 1,
        source_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        transform: Optional[Dict[str, Callable]] = None
    ):
        # Initialize base without scaling state
        super().__init__(
            data=data,
            source_features=source_features,
            target_features=target_features,
            indices=list(range(len(data))) if isinstance(data, pd.DataFrame) else None,
            transform=transform
        )
        # Validate parameters
        if timestamp_col not in self.df.columns:
            raise ValueError(f"{timestamp_col} not in data columns.")
        if window_size <= 0 or step_size <= 0 or horizon < 0 or stride <= 0:
            raise ValueError("window_size, step_size, stride must be >0 and horizon >=0.")
        if window_type not in ('fixed','expanding'):
            raise ValueError("window_type must be 'fixed' or 'expanding'.")
        # Store config
        self.timestamp_col = timestamp_col
        self.window_size = window_size
        self.horizon = horizon
        self.step_size = step_size
        self.window_type = window_type
        self.stride = stride
        # Sort by timestamp
        self.df = self.df.sort_values(timestamp_col).reset_index(drop=True)
        # Build window index pairs
        self.window_indices = self._generate_window_indices()

    def _generate_window_indices(self) -> List[Tuple[List[int], int]]:
        """Return list of (window_idxs, target_idx) pairs."""
        seq_len = len(self.df)
        windows = []
        for start in range(0, seq_len - self.window_size - self.horizon + 1, self.step_size):
            end = start + self.window_size
            # Compute window indices
            if self.window_type == 'fixed':
                idxs = list(range(start, end, self.stride))
            else:  # expanding
                idxs = list(range(0, end, self.stride))
            if len(idxs) < self.window_size // self.stride:
                continue
            target_idx = end + self.horizon - 1
            windows.append((idxs, target_idx))
        return windows

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (X_window [T x features], Y) for window idx."""
        idxs, tgt = self.window_indices[idx]
        # Extract window data
        win_df = self.df.iloc[idxs]
        # Build matrix
        mat = win_df[self.source_features].fillna(0).values
        X = torch.tensor(mat, dtype=torch.float32)
        # Y from target_features at target index
        y_row = self.df.iloc[tgt]
        Y_arr = y_row[self.target_features].fillna(0).values if self.target_features else mat[-1]
        Y = torch.tensor(Y_arr, dtype=torch.float32)
        return X, Y