
class BiomedicalPainDataset(BaseSlidingWindowDataset):
    """
    Efficient dataset for biomedical pain analysis using precomputed sliding windows.
    Extracts time-domain, frequency-domain, and EDA tonic/phasic features.
    """

    def __init__(
        self,
        signals_df: pd.DataFrame,
        pain_df: pd.DataFrame,
        window_size: float,
        step_size: float,
        signal_columns: List[str],
        timestamp_col: str = 'timestamp',
        resample_freq: int = 100,
        bandpass_freqs: Tuple[float, float] = (0.5, 40),
        freq_bands: List[Tuple[float, float]] = [(0.5,4), (4,8), (8,12), (12,30), (30,50)],
        include_time_domain: bool = True,
        tonic_cutoff: float = 0.05,
        include_tonic_phasic: bool = True,
        feature_fn=None,
        label_fn=None,
        window_filter: Optional[Callable[[pd.DataFrame], bool]] = None
    ):
        data = signals_df.sort_values('timestamp').reset_index(drop=True)
        
        step = int(step_size * resample_freq)
        """ super().__init__(
            data=data,
            timestamp_col=timestamp_col,
            window_size=window_size,
            horizon=1,
            step_size=step,
            window_type='fixed',
            stride=1,
            source_features=signal_columns,
            target_features=[],
            feature_fn=None,
            label_fn=None,
            window_filter=window_filter,
        ) """
        self.horizon = 1
        self.window_type = 'fixed'
        self.stride = 1
        self.step_size = step
        self.df = data
        self.timestamp_col = timestamp_col
        self.feature_fn = feature_fn or (lambda x: x)
        self.label_fn = label_fn or (lambda x, y: x)
        # Prepare signals and pain reports
        signals = signals_df.copy()
        signals[timestamp_col] = pd.to_datetime(signals[timestamp_col])
        reports = pain_df.copy()
        reports[timestamp_col] = pd.to_datetime(reports[timestamp_col])
        reports = reports.dropna(subset=['PainLevel']).sort_values(timestamp_col).reset_index(drop=True)
        
        # Store parameters
        self.reports = reports
        self.resample_freq = resample_freq
        self.bandpass_freqs = bandpass_freqs if bandpass_freqs is not None else (0.5, 40)
        self.freq_bands = freq_bands
        self.include_time_domain = include_time_domain
        self.tonic_cutoff = tonic_cutoff
        self.include_tonic_phasic = include_tonic_phasic
        self.signal_columns = signal_columns
        self.timestamp_col = timestamp_col
        
        # Calculate window parameters
        self.window_size = int(window_size * resample_freq)
        step = int(step_size * resample_freq)
        
        # Precompute all windows during initialization
        self.X, self.y, self.window_indices = self._precompute_windows(
            signals, 
            self.window_size, 
            step,
            window_filter
        )
        
        # Generate feature names
        self.feature_names = self._generate_feature_names()

    def _precompute_windows(
        self,
        signals: pd.DataFrame,
        window_size: int,
        step: int,
        window_filter: Optional[Callable[[pd.DataFrame], bool]]
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Precompute features and labels for all valid windows.
        Returns:
            X: Feature matrix (n_windows x n_features)
            y: Label vector (n_windows)
            window_indices: List of (window_start_indices, target_index) tuples
        """
        features = []
        labels = []
        window_indices = []
        seq_len = len(signals)
        
        # Generate sliding windows
        for start in range(0, seq_len - window_size, step):
            end = start + window_size
            win_df = signals.iloc[start:end]
            
            # Skip window if filter rejects it
            if window_filter and not window_filter(win_df):
                continue
                
            # Compute features and label
            X_vec = self._compute_features(win_df)
            y_val = self._compute_label(signals.iloc[end]['timestamp'])
            
            features.append(X_vec)
            labels.append(y_val)
            window_indices.append((list(range(start, end)), end))
        
        return np.vstack(features), np.array(labels), window_indices

    def _compute_features(self, win_df: pd.DataFrame) -> np.ndarray:
        """Extract features for a single window"""
        times = win_df[self.timestamp_col].astype('int64').to_numpy(dtype=float) / 1e9
        t_grid = np.linspace(times[0], times[-1], len(times))
        all_feats = []
        
        for col in self.signal_columns:
            # Handle NaNs and interpolate
            series = win_df[col].ffill().fillna(0).to_numpy(dtype=float)
            interp = np.interp(t_grid, times.astype(float), series)
            
            # Bandpass filtering
            bandpass_freqs = self.bandpass_freqs if self.bandpass_freqs is not None else (0.5, 40)
            b_f, a_f = butter(5, np.array(bandpass_freqs)/(0.5*self.resample_freq), btype='band') # type: ignore
            filt = filtfilt(b_f, a_f, interp)
            
            feats = []
            # Tonic/phasic decomposition
            if self.include_tonic_phasic:
                tonic_cutoff = self.tonic_cutoff if self.tonic_cutoff is not None else 0.05
                b_t, a_t = butter(2, tonic_cutoff/(0.5*self.resample_freq), btype='low') # type: ignore
                tonic = filtfilt(b_t, a_t, interp)
                phasic = interp - tonic
                feats.extend([tonic.mean(), tonic.var(), phasic.mean(), phasic.var()])
            
            # Time-domain features
            if self.include_time_domain:
                feats.extend(self._compute_time_features(filt, self.resample_freq))
            
            # Frequency-domain features
            nperseg = min(256, len(filt))
            freqs, psd = welch(filt, fs=self.resample_freq, nperseg=nperseg)
            for low, high in self.freq_bands:
                mask = (freqs >= low) & (freqs < high)
                feats.append(np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0)
            
            all_feats.extend(feats)
        
        return np.array(all_feats)

    def _compute_label(self, window_end_time: pd.Timestamp) -> int:
        """Determine pain label for window end time"""
        # Find most recent pain report before window end
        past = self.reports[self.reports[self.timestamp_col] <= window_end_time]
        if past.empty or (window_end_time - past.iloc[-1][self.timestamp_col]).total_seconds() > 60:
            return 0  # No recent report
        
        # Categorize pain level
        level = past.iloc[-1]['PainLevel']
        if level == 0: 
            return 0
        if level <= 5: 
            return 1
        return 2

    @staticmethod
    def _compute_time_features(signal: np.ndarray, fs: int) -> List[float]:
        """Calculate time-domain statistics"""
        feats = [
            signal.mean(), signal.var(), np.sqrt(np.mean(signal**2)),
            signal.min(), signal.max(), signal.max()-signal.min(),
            float(skew(signal)), float(kurtosis(signal)),
            np.sum(np.diff(np.sign(signal)) != 0)  # Zero crossings
        ]
        
        # Additional features for longer signals
        if len(signal) > 1:
            x = np.arange(len(signal))
            feats.append(np.polyfit(x, signal, 1)[0])  # Slope
            f, P = welch(signal, fs=fs, nperseg=min(256, len(signal)))
            feats.append(f[np.argmax(P)] if len(P) > 0 else 0.0)  # Peak frequency
        else:
            feats.extend([0.0, 0.0])
            
        return feats

    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature names"""
        names = []
        # Tonic/phasic features
        if self.include_tonic_phasic:
            for col in self.signal_columns:
                names += [f"{col}_tonic_mean", f"{col}_tonic_var", 
                         f"{col}_phasic_mean", f"{col}_phasic_var"]
        
        # Time-domain features
        if self.include_time_domain:
            time_feats = ['mean', 'var', 'rms', 'min', 'max', 'range',
                          'skew', 'kurt', 'zero_cross', 'slope', 'peak_freq']
            for col in self.signal_columns:
                names += [f"{col}_{feat}" for feat in time_feats]
        
        # Frequency-band features
        for col in self.signal_columns:
            names += [f"{col}_{low}-{high}Hz" for low, high in self.freq_bands]
        
        return names

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return precomputed features and label"""
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )