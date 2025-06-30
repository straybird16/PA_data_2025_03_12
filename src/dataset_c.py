import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Literal


class BaseTabularDataset(Dataset):
    """ 
    A flexible PyTorch Dataset for tabular data supporting:
      - Zero-copy slicing via index lists
      - Stratified/random train/val/test splitting
      - Stateful scaling shared across splits
      - Per-feature transforms
      - Clean NA handling

    Returns (X, Y) tensors: X from source_features, Y from target_features (or X if targets unset)
    """

    """ def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        source_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        transform: Optional[Dict[str, Callable]] = None,
        scaler: Optional[Any] = None,
        scale_cols: Optional[List[str]] = None
    ):
        # Load raw data
        self.df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if self.df.empty:
            raise ValueError("Empty input data provided.")

        # Feature groups
        self.source_features = source_features or list(self.df.columns)
        self.target_features = target_features or []
        # all_features covers both for dict mapping
        self.all_features = list({*self.source_features, *self.target_features})

        # Active indices for slicing
        self.indices = indices if indices is not None else list(range(len(self.df)))

        # Transforms and scaling state
        self.transform = transform or {}
        self.scaler = scaler
        self.scale_cols = scale_cols
        self._scaled_array = None """
    """
    A PyTorch Dataset for tabular data supporting:
      - Zero-copy slicing via index lists
      - Stratified, random, or group-based train/val/test splitting
      - Stateful scaling shared across splits
      - Per-feature transforms
      - Clean NA handling

    Returns (X, Y) tensors: X from source_features, Y from target_features (or X if targets unset)

    Splitting options:
      - stratify_by: column for stratified split
      - split_by: list of columns for group-wise split (all rows of each group stay together)
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        source_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        transform: Optional[Dict[str, Callable]] = None,
        scaler: Optional[Any] = None,
        scale_cols: Optional[List[str]] = None
    ):
        # Load raw data
        self.df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if self.df.empty:
            raise ValueError("Empty input data provided.")

        # Feature groups
        self.source_features = source_features or list(self.df.columns)
        self.target_features = target_features or []
        self.all_features = list({*self.source_features, *self.target_features})

        # Indices for zero-copy slicing
        self.indices = indices if indices is not None else list(range(len(self.df)))

        # Optional transforms and scaler state
        self.transform = transform or {}
        self.scaler = scaler
        self.scale_cols = scale_cols
        self._scaled_array = None

    def split(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify_by: Optional[str] = None,
        split_by: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Split into train/test or train/val/test with options:
          - Stratified split by a column (stratify_by)
          - Group-wise split by columns (split_by)
          - Random split otherwise

        :param test_size: Fraction for test split.
        :param val_size: Fraction for validation split (of original total).
        :param stratify_by: Column name for stratification.
        :param split_by: List of columns for group-wise splitting.
        :param random_state: Seed for reproducibility.
        :return: train_ds, test_ds [, val_ds]
        """
        # Work on subset
        subset = self.df.iloc[self.indices]

        # Group-based splitting
        if split_by is not None:
            # unique group identifiers
            groups = subset[split_by].drop_duplicates()
            group_vals = groups.apply(tuple, axis=1).tolist()
            # split group values
            train_groups, test_groups = train_test_split(
                group_vals, test_size=test_size, random_state=random_state
            )
            # map back to row indices
            train_mask = subset[split_by].apply(tuple, axis=1).isin(train_groups)
            test_mask  = subset[split_by].apply(tuple, axis=1).isin(test_groups)
            train_idx = [self.indices[i] for i in np.where(train_mask)[0]]
            test_idx  = [self.indices[i] for i in np.where(test_mask)[0]]
        else:
            # Stratified or random split
            if stratify_by is not None:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=test_size, random_state=random_state
                )
                train_loc, test_loc = next(
                    splitter.split(subset, subset[stratify_by])
                )
            else:
                train_loc, test_loc = train_test_split(
                    range(len(subset)), test_size=test_size, random_state=random_state
                )
            # map to absolute indices
            train_idx = [self.indices[i] for i in train_loc]
            test_idx  = [self.indices[i] for i in test_loc]

        # Create dataset clones
        train_ds = self._clone(indices=train_idx)
        test_ds  = self._clone(indices=test_idx)

        # Validation split if requested
        if val_size is not None:
            val_frac = val_size / (1 - test_size)
            train_ds, val_ds = train_ds.split( # type: ignore
                test_size=val_frac,
                val_size=None,
                stratify_by=stratify_by,
                split_by=split_by,
                random_state=random_state
            )
            return train_ds, val_ds, test_ds

        return train_ds, test_ds

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (X, Y) of tensors:
          X: features from source_features
          Y: features from target_features or X if unset
        """
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        # Build feature vector
        values = {}
        # Apply scaling first for scale_cols
        if self._scaled_array is not None and self.scale_cols is not None:
            scaled_vals = self._scaled_array[idx]
            for i, col in enumerate(self.scale_cols):
                values[col] = float(scaled_vals[i])

        # Fill remaining features
        for feat in self.source_features + self.target_features:
            if feat in values:
                continue
            val = row[feat]
            val = np.nan if pd.isna(val) else val
            # transform
            if feat in self.transform:
                val = self.transform[feat](val)
            values[feat] = val

        # Assemble X and Y arrays
        X_arr = np.array([values[f] for f in self.source_features], dtype=float)
        Y_arr = (
            np.array([values[f] for f in self.target_features], dtype=float)
            if self.target_features else X_arr.copy()
        )

        # Convert to tensors
        X = torch.tensor(X_arr, dtype=torch.float32)
        Y = torch.tensor(Y_arr, dtype=torch.float32)
        return X, Y

    """ def split(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify_by: Optional[str] = None,
        random_state: int = 42
    ):
        #Split into train/test or train/val/test, preserving transforms/scaler.
        subset = self.df.iloc[self.indices]
        # Choose split method
        if stratify_by:
            splitter = StratifiedShuffleSplit(1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(subset, subset[stratify_by]))
        else:
            train_idx, test_idx = train_test_split(range(len(subset)), test_size=test_size, random_state=random_state)
        # Map back to absolute indices
        train_abs = [self.indices[i] for i in train_idx]
        test_abs = [self.indices[i] for i in test_idx]
        # Create child datasets
        train_ds = self._clone(indices=train_abs)
        test_ds = self._clone(indices=test_abs)
        # Optionally split validation
        if val_size is not None:
            val_frac = val_size / (1 - test_size)
            train_ds, val_ds = train_ds.split(test_size=val_frac, val_size=None, # type: ignore
                                              stratify_by=stratify_by, random_state=random_state)
            return train_ds, val_ds, test_ds
        return train_ds, test_ds """

    def apply_scaling(self, method: str = 'standard', columns: Optional[List[str]] = None, fit: bool = True):
        """Fit or apply a scaler, returns new dataset with scaled values."""
        cols = columns or self.source_features
        scaler_map = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
        if method not in scaler_map:
            raise ValueError(f"Unknown scaler method '{method}'")
        ds = self._clone()  # clone existing state
        # Fit or reuse
        if fit or ds.scaler is None:
            ds.scaler = scaler_map[method]()
            scaled = ds.scaler.fit_transform(self.df.loc[self.indices, cols])
        else:
            scaled = ds.scaler.transform(self.df.loc[self.indices, cols])
        ds.scale_cols = cols
        ds._scaled_array = scaled
        return ds

    def get_data_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """Return DataLoader for this dataset view."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def _clone(self, **kwargs) -> 'BaseTabularDataset':
        """Internal helper to clone dataset, overriding given attrs."""
        params = dict(
            data=self.df,
            source_features=self.source_features,
            target_features=self.target_features,
            indices=self.indices.copy(),
            transform=self.transform,
            scaler=self.scaler,
            scale_cols=self.scale_cols
        )
        params.update(kwargs)
        # Ensure 'data' is never None
        if params.get('data', None) is None:
            params['data'] = self.df
        return BaseTabularDataset(**params) # type: ignore


    
class BaseSlidingWindowDataset(Dataset):
    """
    Specialized for time-series: creates windows of past data to predict future.
    Returns (X_window, Y) where X_window is 2D tensor [T x features].
    Split operates on windows (not raw rows) and supports group-wise splitting
    by passing split_by=['Mode'] to keep all windows ending in each Mode together.
    """
    def __init__(
        self, data: pd.DataFrame, window_size: int,
        timestamp_col: str='timestamp', horizon: int = 1, step_size: int = 1,
        window_type: str = 'fixed', stride: int = 1,
        source_features: Optional[List[str]] = None, target_features: Optional[List[str]] = None,
        transform: Optional[Dict[str, Callable]] = None,
        feature_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        label_fn: Optional[Callable[[pd.DataFrame, int], Any]] = None,
        window_filter: Optional[Callable[[pd.DataFrame], bool]] = None,
    ):
        super().__init__()
        self.df = data.sort_values(timestamp_col).reset_index(drop=True)
        self.timestamp_col = timestamp_col
        self.window_size = window_size
        self.horizon = horizon
        self.step_size = step_size
        self.window_type = window_type
        self.stride = stride
        self.source_features = source_features or list(self.df.columns)
        self.target_features = target_features or []
        self.transform = transform or {}
        
        self.feature_fn = feature_fn or (lambda x: x)
        self.label_fn = label_fn or (lambda x, y: x)
        self.window_filter = window_filter or (lambda x: True)  # default filter that accepts all windows

        self.window_indices = self._generate_window_indices() # precompute indices
        self._precompute_all()

    def _generate_window_indices(self) -> List[Tuple[List[int], int]]:
        seq_len = len(self.df)
        windows = []
        # Generate indices for sliding windows
        if self.window_type not in ['fixed', 'cumulative']:
            raise ValueError(f"Unknown window type '{self.window_type}'")
        for start in range(0, seq_len - self.window_size - self.horizon + 1, self.step_size):
            end = start + self.window_size
            if self.window_type == 'fixed':
                idxs = list(range(start, end, self.stride))
            else:
                idxs = list(range(0, end, self.stride))
            if len(idxs) < self.window_size // self.stride: # ensure enough data
                continue
            target_idx = end + self.horizon - 1 #
            
            win_df = self.df.iloc[idxs]
            if not self.window_filter(win_df):
                continue
            windows.append((idxs, target_idx))
        
        self.window_indices = windows
        return windows

    def _precompute_all(self) -> Tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for idxs, tgt in self.window_indices:
            win_df = self.df.iloc[idxs]
            X_vec = self.feature_fn(win_df)
            y_val = self.label_fn(self.df, tgt)
            features.append(X_vec)
            labels.append(y_val)
        
        self.X, self.y = np.vstack(features), np.array(labels)
        return self.X, self.y

    def __len__(self):
        return len(self.y)

    """ def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs, tgt = self.window_indices[idx]
        win_df = self.df.iloc[idxs]
        mat = win_df[self.source_features].fillna(0).values
        X = torch.tensor(mat, dtype=torch.float32)
        y_val = self.df.iloc[tgt][self.target_features].fillna(0).values \
                if self.target_features else mat[-1]
        Y = torch.tensor(y_val, dtype=torch.float32)
        return X, Y """
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y


    def split( # type: ignore
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify_by: Optional[str] = None,
        split_by: Optional[List[str]] = None,
        random_state: int = 42
    ) -> Union[
        Tuple['BaseSlidingWindowDataset', 'BaseSlidingWindowDataset'],
        Tuple['BaseSlidingWindowDataset', 'BaseSlidingWindowDataset', 'BaseSlidingWindowDataset']
    ]:
        """
        Split windows into train/test (or train/val/test).
        - split_by: list of raw-data columns (e.g. ['Mode']) to group windows by the Mode
                    value at the *end* of each window.
        - stratify_by: a raw-data column for stratified splits (again, at window end).
        """
        n = len(self.window_indices)
        # Determine train/test window indices
        if split_by:
            # get group key at each window's end
            ends = [tgt for (_, tgt) in self.window_indices]
            df_end = self.df.iloc[ends]
            groups = df_end[split_by].apply(tuple, axis=1)
            uniq = groups.unique().tolist()
            train_g, test_g = train_test_split(uniq, test_size=test_size, random_state=random_state, shuffle=False)
            train_locs = np.where(groups.isin(train_g))[0]
            test_locs  = np.where(groups.isin(test_g))[0]
        else:
            if stratify_by:
                ends = [tgt for (_, tgt) in self.window_indices]
                df_end = self.df.iloc[ends]
                splitter = StratifiedShuffleSplit(1, test_size=test_size, random_state=random_state)
                train_locs, test_locs = next(
                    splitter.split(np.zeros(n), df_end[stratify_by])
                )
            else:
                train_locs, test_locs = train_test_split(
                    list(range(n)), test_size=test_size, random_state=random_state
                )

        train_w = [self.window_indices[i] for i in train_locs]
        test_w  = [self.window_indices[i] for i in test_locs]
        train_ds = self._clone_windows(train_w)
        test_ds  = self._clone_windows(test_w)

        if val_size is not None:
            # further split train into train/val
            val_frac = val_size / (1 - test_size)
            train_ds, val_ds = train_ds.split( # type: ignore
                test_size=val_frac, val_size=None,
                stratify_by=stratify_by, split_by=split_by,
                random_state=random_state
            )
            return train_ds, val_ds, test_ds

        return train_ds, test_ds

    def _clone_windows(self, window_indices: List[Tuple[List[int], int]]):
        """Internal: make a new dataset copying config but with a subset of windows."""
        new = self.__class__.__new__(self.__class__)
        # copy attributes
        for attr in (
            'df','timestamp_col','window_size','horizon','step_size',
            'window_type','stride','source_features','target_features','transform'
        ):
            setattr(new, attr, getattr(self, attr))
        new.window_indices = window_indices
        return new
    
    def split(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify_by: Optional[str] = None,
        split_by: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Split dataset into train/test (or train/val/test).

        - split_by: raw-data columns to group windows by value at window end index.
        - stratify_by: raw-data column to stratify by at window end index.
        """
        n = len(self.y)
        # Determine indices 0..n-1 for each split
        if split_by:
            ends = [tgt for (_, tgt) in self.window_indices]
            df_end = self.df.iloc[ends]
            groups = df_end[split_by].apply(tuple, axis=1)
            uniq = groups.unique().tolist()
            train_g, test_g = train_test_split(uniq, test_size=test_size, random_state=random_state)
            train_idxs = np.where(groups.isin(train_g))[0]
            test_idxs  = np.where(groups.isin(test_g))[0]
        else:
            if stratify_by:
                ends = [tgt for (_, tgt) in self.window_indices]
                df_end = self.df.iloc[ends]
                splitter = StratifiedShuffleSplit(1, test_size=test_size, random_state=random_state)
                train_idxs, test_idxs = next(splitter.split(np.zeros(n), df_end[stratify_by]))
            else:
                train_idxs, test_idxs = train_test_split(np.arange(n), test_size=test_size, random_state=random_state)

        # slice X and y

        train_ds = self._clone_subset(train_idxs)
        test_ds  = self._clone_subset(test_idxs)

        if val_size is not None:
            val_frac = val_size / (1 - test_size)
            train_ds, val_ds = train_ds.split( # type: ignore
                test_size=val_frac, val_size=None,
                stratify_by=stratify_by, split_by=split_by,
                random_state=random_state
            )
            return train_ds, val_ds, test_ds
        return train_ds, test_ds
    
    def _clone_subset(self, idxs):
            ds = self.__class__.__new__(self.__class__)
            # copy config
            for attr in ('df','timestamp_col','window_size','horizon','step_size','window_type','stride','feature_fn','label_fn'):
                setattr(ds, attr, getattr(self, attr))
            ds.window_indices = [self.window_indices[i] for i in idxs]
            ds.X = self.X[idxs]
            ds.y = self.y[idxs]
            return ds
        
    def apply_scaling(
        self,
        method: str = 'standard',
        fit: bool = True,
        scaler: Any = None
    ):
        """
        Fit or apply a scaler to self.X in place.

        :param method: 'standard'|'minmax'|'robust'
        :param fit:    True to fit new scaler on X; False to reuse existing
        """
        scaler_map = {
            'standard': StandardScaler,
            'minmax':   MinMaxScaler,
            'robust':   RobustScaler
        }
        if method not in scaler_map:
            raise ValueError(f"Unknown scaler method: {method}")
        # Decide which scaler instance to use
        if fit:
            self.scaler = scaler_map[method]()
            self.X = self.scaler.fit_transform(self.X)
        else:
            if scaler is None:
                raise ValueError("must pass `scaler=` when fit=False")
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
            
    def process_nan(
        self,
        nan_strategy: 'Literal["interpolate", "zero", "none"]' = 'zero',
        nan_interp_method: 'Literal["linear", "time", "index", "values", "nearest", "zero", "slinear"]' = 'linear'
    ):
        if nan_strategy == 'interpolate':
            dfX = pd.DataFrame(self.X)
            self.X = dfX.interpolate(method=nan_interp_method, axis=0)
            # fill any remaining NaNs
            #self.X = self.X.fillna(method='bfill').fillna(method='ffill').values
        elif nan_strategy == 'zero':
            self.X = np.nan_to_num(self.X, nan=0.0)


class BiomedicalPainDataset(BaseSlidingWindowDataset):
    """
    Processes biomedical signals and pain reports into sliding windows.
    Adds time/frequency domain and EDA tonic/phasic features.
    """
    def __init__(
        self,
        signals_df: pd.DataFrame,
        pain_df: pd.DataFrame,
        window_duration: float,
        step_size: float,
        signal_columns: List[str],
        timestamp_col: str='timestamp',
        resample_freq: int = 100,
        bandpass_freqs: Tuple[float,float] = (0.5,40),
        freq_bands: List[Tuple[float,float]] = [(0.5,4),(4,8),(8,12),(12,30),(30,50)],
        include_time_domain: bool = True,
        tonic_cutoff: float = 0.05,
        include_tonic_phasic: bool = True,
        window_filter: Optional[Callable[[pd.DataFrame], bool]] = None
    ):
        # Prepare signals and reports
        signals = signals_df.copy()
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])
        self.signals = signals.sort_values('timestamp').reset_index(drop=True)
        reports = pain_df.copy()
        reports['timestamp'] = pd.to_datetime(reports['timestamp'])
        reports = reports.dropna(subset=['PainLevel']).sort_values('timestamp').reset_index(drop=True)
        self.reports =  reports
        # calculate window params in samples
        window_size = int(window_duration * resample_freq)
        step = int(step_size * resample_freq)
        # Sort signals
        data = signals.sort_values('timestamp').reset_index(drop=True)
        window_size = int(window_duration * resample_freq)
        step = int(step_size * resample_freq)
        
        # define feature and label functions
        def feature_fn(win_df: pd.DataFrame) -> np.ndarray:
            # uniform time grid
            times = win_df['timestamp'].astype('int64').to_numpy(dtype=float) / 1e9
            t_grid = np.linspace(times[0], times[-1], len(times))
            feats_all = []
            for col in signal_columns:
                series = win_df[col].ffill().fillna(0).values
                interp = np.interp(
                    t_grid,
                    np.asarray(times, dtype=float),
                    np.asarray(series, dtype=float)
                )
                # bandpass
                b_f, a_f = butter(5, np.array(bandpass_freqs)/(0.5*resample_freq), btype='bandpass') # type: ignore
                filt = filtfilt(b_f, a_f, interp)
                feats = []
                if include_tonic_phasic:
                    b_t, a_t = butter(2, tonic_cutoff/(0.5*resample_freq), btype='lowpass') # type: ignore
                    tonic = filtfilt(b_t, a_t, interp)
                    phasic = interp - tonic
                    feats.extend([tonic.mean(), tonic.var(), phasic.mean(), phasic.var()])
                if include_time_domain:
                    feats.extend(self._compute_time_features(filt, resample_freq))
                # safe PSD
                nperseg = min(256, len(filt))
                freqs, psd = welch(filt, fs=resample_freq, nperseg=nperseg)
                for low, high in freq_bands:
                    mask = (freqs>=low)&(freqs<high)
                    feats.append(np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0)
                feats_all.append(feats)
            return np.concatenate(feats_all)

        def label_fn(df: pd.DataFrame, tgt_idx: int) -> int:
            t = df.iloc[tgt_idx]['timestamp']
            past = self.reports[self.reports['timestamp']<=t]
            level = past['PainLevel'].iloc[-1] if not past.empty else 0.0
            # categorize
            if level == 0: return 0
            if level <=5: return 1
            return 2
        
        super().__init__(
            data=data,
            timestamp_col=timestamp_col,
            window_size=window_size,
            horizon=1,
            step_size=step,
            window_type='fixed',
            stride=1,
            source_features=signal_columns,
            target_features=[],
            feature_fn=feature_fn,
            label_fn=label_fn,
            window_filter=window_filter,
        )
        self.resample_freq = resample_freq
        self.bandpass_freqs = bandpass_freqs if bandpass_freqs is not None else (0.5, 40)
        self.freq_bands = freq_bands
        self.include_time_domain = include_time_domain
        self.tonic_cutoff = tonic_cutoff if tonic_cutoff is not None else 0.05
        self.include_tonic_phasic = include_tonic_phasic
        self.signal_columns = signal_columns
        self.feature_fn = feature_fn
        self.label_fn = label_fn
        
    @staticmethod
    def _compute_time_features(signal: np.ndarray, fs: int) -> List[float]:
        feats = [signal.mean(), signal.var(), np.sqrt(np.mean(signal**2)),
                 signal.min(), signal.max(), signal.max()-signal.min(),
                 float(skew(signal)), float(kurtosis(signal)),
                 np.sum(np.diff(np.sign(signal))!=0)]
        if len(signal)>1:
            x = np.arange(len(signal))
            feats.append(np.polyfit(x, signal, 1)[0])
            f, P = welch(signal, fs=fs, nperseg=min(256, len(signal)))
            feats.append(f[np.argmax(P)] if len(P)>0 else 0.0)
        else:
            feats.extend([0.0, 0.0])
        return feats
    
    def _generate_feature_names(self) -> List[str]:
        names: List[str] = []
        # Tonic & Phasic
        if self.include_tonic_phasic:
            for col in self.signal_columns:
                names += [f"{col}_tonic_mean", f"{col}_tonic_var",
                          f"{col}_phasic_mean", f"{col}_phasic_var"]
        # Time-domain
        if self.include_time_domain:
            time_feats = ['mean','var','rms','min','max','range',
                          'skew','kurt','zero_cross','slope','peak_freq']
            for col in self.signal_columns:
                for feat in time_feats:
                    names.append(f"{col}_{feat}")
        # Frequency-domain
        for col in self.signal_columns:
            for low, high in self.freq_bands:
                names.append(f"{col}_{low}-{high}Hz")
        self.feature_names = names
        return names

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs, tgt = self.window_indices[idx]
        win_df = self.df.iloc[idxs]
        times = win_df['timestamp'].astype('int64').to_numpy(dtype=float) / 1e9
        t_grid = np.linspace(times[0], times[-1], len(times))
        all_feats = []
        for col in self.source_features:
            series = win_df[col].ffill().fillna(0).to_numpy(dtype=float)
            # Interpolate
            interp = np.interp(t_grid, times.astype(float), series)
            # Bandpass filter
            b_f, a_f = butter(5, np.array(self.bandpass_freqs)/(0.5*self.resample_freq), btype='band') # type: ignore
            filt = filtfilt(b_f, a_f, interp)
            feats = []
            # Tonic/phasic
            if self.include_tonic_phasic:
                b_t, a_t = butter(2, self.tonic_cutoff/(0.5*self.resample_freq), btype='low') # type: ignore
                tonic = filtfilt(b_t, a_t, interp)
                phasic = interp - tonic
                feats.extend([tonic.mean(), tonic.var(), phasic.mean(), phasic.var()])
            # Time-domain
            if self.include_time_domain:
                feats.extend(self._compute_time_domain_features(filt))
            # Frequency-domain with safe nperseg
            nperseg = min(256, len(filt))
            freqs, psd = welch(filt, fs=self.resample_freq, nperseg=nperseg)
            for low, high in self.freq_bands:
                mask = (freqs>=low)&(freqs<high)
                feats.append(np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0)
            all_feats.extend(feats)
        X = torch.tensor(np.array(all_feats), dtype=torch.float32)
        # Pain label
        end_time = win_df['timestamp'].iloc[-1]
        level = self._compute_pain_label(end_time)
        cat = self._categorize_pain(level)
        # return a scalar tensor, not a length-1 vector
        Y = torch.tensor(cat, dtype=torch.long)
        return X, Y

    def _compute_time_domain_features(self, signal: np.ndarray) -> List[float]:
        feats = []
        feats.append(signal.mean())
        feats.append(signal.var())
        feats.append(np.sqrt(np.mean(signal**2)))
        feats.append(signal.min())
        feats.append(signal.max())
        feats.append(signal.max()-signal.min())
        feats.append(float(skew(signal)))
        feats.append(float(kurtosis(signal)))
        zero_cross = np.sum(np.diff(np.sign(signal))!=0)
        feats.append(zero_cross)
        if len(signal)>1:
            x = np.arange(len(signal))
            feats.append(np.polyfit(x, signal,1)[0])
            f, P = welch(signal, fs=self.resample_freq, nperseg=min(256,len(signal)))
            feats.append(f[np.argmax(P)] if len(P)>0 else 0.0)
        else:
            feats.extend([0.0, 0.0])
        return feats

    def _compute_pain_label(self, end_time: pd.Timestamp) -> float:
        past = self.reports[self.reports['timestamp']<=end_time]
        if past.empty:
            return 0.0
        return past.iloc[-1]['PainLevel']

    def _categorize_pain(self, level: float) -> int:
        if level==0:
            return 0
        elif level<=5:
            return 1
        else:
            return 2
        
    def _clone_windows(self, window_indices: List[Tuple[List[int], int]]):
        """Internal: make a new dataset copying config but with a subset of windows."""
        new = self.__class__.__new__(self.__class__)
        # copy attributes
        for attr in (
            'df','timestamp_col','window_size','horizon','step_size',
            'window_type','stride','source_features','target_features','transform',
            'reports',
            'resample_freq', 'bandpass_freqs', 'freq_bands', 'include_time_domain', 'tonic_cutoff',
            'include_tonic_phasic', 'signal_columns',
        ):
            setattr(new, attr, getattr(self, attr))
        new.window_indices = window_indices
        return new
    
    def _clone_subset(self, idxs):
            ds = self.__class__.__new__(self.__class__)
            # copy config
            for attr in (
                'df','timestamp_col','window_size','horizon','step_size',
                'window_type','stride','source_features','target_features','transform',
                'reports',
                'resample_freq', 'bandpass_freqs', 'freq_bands', 'include_time_domain', 'tonic_cutoff',
                'include_tonic_phasic', 'signal_columns', 'feature_fn', 'label_fn',
            ):
                setattr(ds, attr, getattr(self, attr))
            ds.window_indices = [self.window_indices[i] for i in idxs]
            ds.X = self.X[idxs]
            ds.y = self.y[idxs]
            return ds
