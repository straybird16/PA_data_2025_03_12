from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Union, Optional, Callable
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    """
    Efficient PyTorch dataset for 2D tabular data with advanced processing pipeline.
    Maintains original data integrity while providing flexible transformations and splits.
    
    Processing Pipeline:
    1. Pre-filtering (indices/feature selection)
    2. Train-test split (random, stratified, or group-based)
    3. Data scaling
    4. Post-filtering and transformation
    
    Features:
    - Zero-copy operations using indices and masks
    - Sparse data support (NA handling)
    - Lazy transformations applied at access time
    - Memory-efficient scaling
    - Modular split strategies
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        source_feature_names: Optional[List[str]] = None,
        target_feature_names: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        transform: Optional[Dict[str, Callable]] = None,
        pre_filter_func: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    ):
        """
        Initialize dataset with data processing parameters.
        
        :param data: Input data (DataFrame or numpy array)
        :param source_feature_names: Columns for source features (X)
        :param target_feature_names: Columns for target features (Y)
        :param indices: Row indices to select
        :param transform: Feature-specific transformations {feature: callable}
        :param pre_filter_func: Function for initial data filtering (returns boolean mask)
        """
        # Convert to DataFrame if numpy array (zero-copy if possible)
        self.raw_data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if self.raw_data.empty:
            raise ValueError("Input data is empty. Please provide valid data.")
            
        # Initialize feature selection
        self.source_feature_names = source_feature_names or self.raw_data.columns.tolist()
        self.target_feature_names = target_feature_names or []
        
        # Handle indices and filtering
        self.valid_indices = indices if indices is not None else self._generate_valid_indices()
        
        # Apply pre-filter using boolean mask
        if pre_filter_func:
            mask:pd.Series[bool] = pre_filter_func(self.raw_data.loc[self.valid_indices])
            self.valid_indices = np.array(self.valid_indices)[mask].tolist()
            
        # Transformation setup
        self.transform = transform or {}
        self.scaler = None
        self.scale_columns = None

    def _generate_valid_indices(self) -> List[int]:
        """Generate valid indices, handling NA values by default"""
        # Default: drop rows with NA in source/target features
        cols = self.source_feature_names + self.target_feature_names
        mask = self.raw_data[cols].notna().all(axis=1)
        indices = np.where(mask)[0]
        if isinstance(indices, (np.integer, int)):
            return [int(indices)]
        return indices.tolist() # type: ignore

    def split(
        self,
        test_size: float = 0.2,
        stratify_by: Optional[str] = None,
        split_by: Optional[List[str]] = None,
        random_state: int = 42,
        split_method: Union[str, Callable] = 'default',
        split_params: Optional[Dict] = None,
        scale_method: Optional[str] = None,
        scale_columns: Optional[List[str]] = None,
        post_filter_func: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    ) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        Split dataset into train/test sets with various strategies.
        
        :param test_size: Proportion of test set (0-1)
        :param stratify_by: Column name for stratification
        :param split_by: Column(s) for group-based splitting
        :param random_state: Random seed for reproducibility
        :param split_method: 'default', 'random', 'ordered', or custom function
        :param split_params: Additional parameters for split function
        :param scale_method: Scaling method ('standard', 'minmax', 'robust')
        :param scale_columns: Columns to apply scaling
        :param post_filter_func: Filter function applied after splitting
        :return: Tuple of (train_dataset, test_dataset)
        """
        # Prepare split parameters
        split_params = split_params or {}
        if 'test_size' not in split_params:
            split_params['test_size'] = test_size
        if 'random_state' not in split_params:
            split_params['random_state'] = random_state

        # Get current data view
        current_data = self.raw_data.loc[np.array(self.valid_indices)]
        
        # Handle splitting
        if callable(split_method):
            train_idx, test_idx = split_method(current_data, **split_params)
        elif split_method == 'default' and split_by:
            train_idx, test_idx = self._group_split(current_data, split_by, test_size, random_state)
        elif split_method == 'default' and stratify_by:
            sss = StratifiedShuffleSplit(n_splits=1, **split_params)
            train_idx, test_idx = next(sss.split(current_data, current_data[stratify_by]))
        elif split_method == 'ordered':
            n = len(current_data)
            split_point = int(n * (1 - test_size))
            train_idx, test_idx = range(split_point), range(split_point, n)
        else:  # Random split
            train_idx, test_idx = train_test_split(
                range(len(current_data)), **split_params
            )
            
        # Convert to absolute indices
        train_indices = [np.array(self.valid_indices)[i] for i in train_idx]
        test_indices = [np.array(self.valid_indices)[i] for i in test_idx]
        
        # Create base datasets
        train_ds = self._create_subset(train_indices)
        test_ds = self._create_subset(test_indices)
        
        # Handle scaling
        if scale_method:
            scale_columns = scale_columns or self.source_feature_names
            train_ds, test_ds = self._apply_scaling(
                train_ds, test_ds, scale_method, scale_columns
            )
        
        # Apply post-filtering
        if post_filter_func:
            train_ds = train_ds.apply_post_filter(post_filter_func)
            test_ds = test_ds.apply_post_filter(post_filter_func)
            
        return train_ds, test_ds

    def _group_split(
        self,
        data: pd.DataFrame,
        split_by: List[str],
        test_size: float,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficient group-based splitting using hashing for large datasets
        
        :param data: Data subset to split
        :param split_by: Columns defining groups
        :param test_size: Proportion of groups in test set
        :param random_state: Random seed
        :return: (train_indices, test_indices) as positional indices
        """
        # Create group identifiers using hash of tuple
        group_ids = data[split_by].apply(
            lambda row: hash(tuple(row)), axis=1
        ).values
        
        # Get unique groups
        unique_groups = np.unique(group_ids)
        n_test = max(1, int(len(unique_groups) * test_size))
        
        # Select test groups
        rng = np.random.RandomState(random_state)
        test_groups = rng.choice(unique_groups, size=n_test, replace=False)
        test_mask = np.isin(group_ids, test_groups)
        
        return np.where(~test_mask)[0], np.where(test_mask)[0]

    def _apply_scaling(
        self,
        train_ds: 'BaseDataset',
        test_ds: 'BaseDataset',
        method: str,
        columns: List[str]
    ) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        Apply scaling efficiently using in-place operations on numpy arrays
        
        :param train_ds: Training dataset
        :param test_ds: Test dataset
        :param method: Scaling method
        :param columns: Columns to scale
        :return: Scaled datasets
        """
        scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        if method not in scalers:
            raise ValueError(f"Unsupported scaler: {method}. Choose from {list(scalers.keys())}")
        
        # Extract data as numpy arrays for efficiency
        train_data = train_ds.raw_data.loc[train_ds.valid_indices, columns].values
        test_data = test_ds.raw_data.loc[test_ds.valid_indices, columns].values
        
        # Fit and transform
        scaler = scalers[method]()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Create new datasets with scaled data
        train_scaled_df = train_ds.raw_data.loc[train_ds.valid_indices].copy()
        test_scaled_df = test_ds.raw_data.loc[test_ds.valid_indices].copy()
        
        # Efficient assignment without copying full DataFrames
        for i, col in enumerate(columns):
            train_scaled_df[col] = train_scaled[:, i]
            test_scaled_df[col] = test_scaled[:, i]
        
        # Create new datasets preserving state
        scaled_train = self.__class__(
            train_scaled_df,
            source_feature_names=self.source_feature_names,
            target_feature_names=self.target_feature_names,
            transform=self.transform
        )
        scaled_test = self.__class__(
            test_scaled_df,
            source_feature_names=self.source_feature_names,
            target_feature_names=self.target_feature_names,
            transform=self.transform
        )
        
        # Store scaler reference
        scaled_train.scaler = scaler
        scaled_train.scale_columns = columns
        scaled_test.scaler = scaler
        scaled_test.scale_columns = columns
        
        return scaled_train, scaled_test

    def apply_post_filter(self, filter_func: Callable) -> 'BaseDataset':
        """
        Apply post-filter to current dataset and return new subset
        
        :param filter_func: Filter function (returns boolean mask)
        :return: New filtered dataset
        """
        mask = filter_func(self.raw_data.loc[self.valid_indices])
        new_indices = np.array(self.valid_indices)[mask].tolist()
        return self._create_subset(new_indices)

    def _create_subset(self, indices: List[int]) -> 'BaseDataset':
        """Create new dataset instance with subset of indices"""
        return self.__class__(
            self.raw_data,
            source_feature_names=self.source_feature_names,
            target_feature_names=self.target_feature_names,
            indices=indices,
            transform=self.transform
        )

    def __len__(self) -> int:
        """Return number of samples in dataset"""
        return len(list(self.valid_indices)) # type: ignore

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve sample by index with applied transformations.
        Handles sparse data by converting NA to NaN.
        
        :param idx: Index in valid_indices
        :return: Dictionary of {feature: tensor} pairs
        """
        raw_idx = self.valid_indices[idx]
        sample = self.raw_data.iloc[raw_idx]
        
        # Apply transformations
        transformed = {}
        for feature in self.source_feature_names + self.target_feature_names:
            value = sample[feature]
            
            # Handle NA values
            if pd.isna(value):
                value = np.nan
            
            # Apply feature-specific transform if exists
            if feature in self.transform:
                transformed[feature] = self.transform[feature](value)
            else:
                # Convert to tensor with type inference
                transformed[feature] = torch.tensor(value)
                
        return transformed

    @property
    def shape(self) -> Tuple[int, int]:
        """Return dataset dimensions (samples, features)"""
        return len(self), len(self.source_feature_names + self.target_feature_names)

    def get_data_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
        """
        Create DataLoader for this dataset.
        
        :param batch_size: Number of samples per batch
        :param shuffle: Whether to shuffle data
        :return: Configured DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    def __repr__(self) -> str:
        """Return string representation of dataset"""
        return (f"<BaseDataset: samples={len(self)}, "
                f"features={len(self.source_feature_names + self.target_feature_names)}, "
                f"valid_indices={len(self.valid_indices)}>")
    


class BaseDataset(Dataset):
    """
    Efficient PyTorch dataset for 2D tabular data with robust processing pipeline.
    Maintains raw data immutability while providing flexible transformations and splits.
    
    Processing Pipeline:
    1. Initialization: Store raw data and set valid indices
    2. Pre-filtering: Apply user-defined filtering (optional)
    3. Splitting: Create train/test subsets using indices only
    4. Transformation: Apply scaling and feature transforms (stateful for train, applied to test)
    5. Post-filtering: Final filtering after transformations (optional)

    Key Features:
    - Zero-copy operations via index manipulation
    - Lazy transformations to minimize memory usage
    - Strict separation of train/test processing
    - Comprehensive NA handling
    - Modular design for easy extension
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        source_feature_names: Optional[List[str]] = None,
        target_feature_names: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        transform: Optional[Dict[str, Callable]] = None,
        pre_filter_func: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    ):
        """
        Initialize dataset while preserving raw data.
        
        :param data: Input data (DataFrame or numpy array)
        :param source_feature_names: Columns for source features (X)
        :param target_feature_names: Columns for target features (Y)
        :param indices: Initial valid indices
        :param transform: Feature-specific transformations
        :param pre_filter_func: Initial data filter (applies to raw data)
        """
        # Preserve raw data without modification
        self.raw_data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if self.raw_data.empty:
            raise ValueError("Input data is empty")
            
        # Initialize feature selection
        self.source_feature_names = source_feature_names or self.raw_data.columns.tolist()
        self.target_feature_names = target_feature_names or []
        self.all_features = list(set(self.source_feature_names + self.target_feature_names))
        
        # Initialize stateful components
        self.scaler = None
        self.scale_columns = None
        self.scaled_values = None 
        self.transform = transform or {}
        
        # Handle indices and filtering
        self.valid_indices = indices if indices is not None else list(range(len(self.raw_data)))
        
        # Apply pre-filter to raw data
        if pre_filter_func:
            self._apply_pre_filter(pre_filter_func)

    def _apply_pre_filter(self, filter_func: Callable):
        """Apply initial filter to raw data"""
        mask = filter_func(self.raw_data.iloc[self.valid_indices])
        self.valid_indices = [self.valid_indices[i] for i in np.where(mask)[0]]

    def _generate_valid_indices(self) -> List[int]:
        """Generate valid indices (override for custom logic)"""
        return list(range(len(self.raw_data)))

    def split(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify_by: Optional[str] = None,
        split_by: Optional[List[str]] = None,
        random_state: int = 42,
        split_method: Union[str, Callable] = 'default',
        split_params: Optional[Dict] = None
    ) -> Union[
        Tuple['BaseDataset', 'BaseDataset'],
        Tuple['BaseDataset', 'BaseDataset', 'BaseDataset']
    ]:
        """
        Split dataset into train/test or train/val/test subsets.
        
        :param test_size: Proportion of test set (0-1)
        :param val_size: Proportion of validation set (0-1, optional)
        :param stratify_by: Column for stratification
        :param split_by: Columns for group-based splitting
        :param random_state: Random seed
        :param split_method: 'default', 'random', 'ordered', or custom function
        :param split_params: Additional split parameters
        :return: (train, test) or (train, val, test) datasets
        """
        # Prepare split parameters
        split_params = split_params or {}
        split_params.setdefault('test_size', test_size)
        split_params.setdefault('random_state', random_state)
        current_data = self.raw_data.iloc[self.valid_indices]
        
        # Handle different split methods
        if callable(split_method):
            indices = split_method(current_data, **split_params)
        elif split_by:
            # Extract group split strategy
            strategy = split_params.get('group_strategy', 'random')
            order_by = split_params.get('order_by', split_by)
            
            indices = self._group_split(
                current_data, split_by, test_size, val_size, 
                strategy, order_by, random_state
            )
        elif stratify_by:
            sss = StratifiedShuffleSplit(n_splits=1, **split_params)
            train_idx, test_idx = next(sss.split(current_data, current_data[stratify_by]))
            indices = (train_idx, test_idx)
        elif split_method == 'ordered':
            n = len(current_data)
            split_point = int(n * (1 - test_size))
            indices = (range(split_point), range(split_point, n))
        else:  # Random split
            indices = train_test_split(
                range(len(current_data)), **split_params
            )
            
        # Convert to absolute indices
        absolute_indices = [
            [self.valid_indices[i] for i in idx_set]
            for idx_set in indices
        ]
        
        # Create datasets
        if val_size is None:
            train_ds = self._create_subset(absolute_indices[0])
            test_ds = self._create_subset(absolute_indices[1])
            return train_ds, test_ds
        else:
            train_ds = self._create_subset(absolute_indices[0])
            val_ds = self._create_subset(absolute_indices[1])
            test_ds = self._create_subset(absolute_indices[2])
            return train_ds, val_ds, test_ds

    def _group_split(
        self,
        data: pd.DataFrame,
        group_cols: List[str],
        test_size: float,
        val_size: Optional[float],
        strategy: str = 'random',
        order_by: Optional[List[str]] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, ...]:
        """
        Group-based splitting with multiple strategies and hierarchical support.
        
        :param data: Data subset to split
        :param group_cols: Columns defining groups
        :param test_size: Proportion of groups in test set
        :param val_size: Proportion of groups in validation set
        :param strategy: 'random' or 'ordered'
        :param order_by: Columns for ordering groups
        :param random_state: Random seed
        :return: Tuple of index arrays for each split
        """
        # Create group identifiers
        group_ids = data[group_cols].apply(tuple, axis=1)
        unique_groups = pd.unique(group_ids)
        
        # Order groups if specified
        if order_by:
            # Create ordering keys
            order_keys = data.drop_duplicates(group_cols)
            order_keys = order_keys.sort_values(order_by)[group_cols]
            ordered_groups = [tuple(x) for x in order_keys.to_numpy()]
            unique_groups = [g for g in ordered_groups if g in unique_groups]
        elif strategy == 'ordered':
            # Default ordering by group tuple
            unique_groups = sorted(unique_groups)
        
        # Calculate split sizes
        n_groups = len(unique_groups)
        n_test = max(1, int(n_groups * test_size))
        
        if val_size:
            n_val = max(1, int(n_groups * val_size))
            n_train = n_groups - n_val - n_test
        else:
            n_val = 0
            n_train = n_groups - n_test
        
        # Split groups based on strategy
        if strategy == 'random':
            rng = np.random.RandomState(random_state)
            shuffled_groups = rng.permutation(np.array(unique_groups))
            test_groups = shuffled_groups[:n_test]
            val_groups = shuffled_groups[n_test:n_test+n_val] if val_size else []
            train_groups = shuffled_groups[n_test+n_val:] if val_size else shuffled_groups[n_test:]
        else:  # Ordered strategy
            test_groups = unique_groups[:n_test]
            val_groups = unique_groups[n_test:n_test+n_val] if val_size else []
            train_groups = unique_groups[n_test+n_val:] if val_size else unique_groups[n_test:]
        
        # Create masks for each split
        test_mask = group_ids.isin(test_groups)
        val_mask = group_ids.isin(val_groups) if val_size else pd.Series(False, index=group_ids.index)
        train_mask = group_ids.isin(train_groups)
        
        # Convert to indices
        test_idx = np.where(test_mask)[0]
        val_idx = np.where(val_mask)[0] if val_size else np.array([])
        train_idx = np.where(train_mask)[0]
        
        if val_size:
            return train_idx, val_idx, test_idx
        return train_idx, test_idx

    def apply_scaling(
        self,
        scale_method: str,
        scale_columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> 'BaseDataset':
        """
        Apply scaling to dataset (stateful when fit=True)
        
        :param scale_method: 'standard', 'minmax', 'robust'
        :param scale_columns: Columns to scale (None for all source features)
        :param fit: Whether to fit new scaler (True for train, False for test)
        :return: New dataset with scaling applied
        """
        scale_columns = scale_columns or self.source_feature_names
        scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        if scale_method not in scalers:
            raise ValueError(f"Unsupported scaler: {scale_method}")
        
        # Create new dataset instance
        scaled_ds = self._create_subset(self.valid_indices)
        
        # Initialize or reuse scaler
        if fit or self.scaler is None:
            scaled_ds.scaler = scalers[scale_method]()
            scaled_ds.scale_columns = scale_columns
        else:
            scaled_ds.scaler = self.scaler
            scaled_ds.scale_columns = self.scale_columns
            
        # Apply scaling
        scaled_values = scaled_ds.scaler.fit_transform(
            self.raw_data.loc[self.valid_indices, scale_columns]
        ) if fit else scaled_ds.scaler.transform(
            self.raw_data.loc[self.valid_indices, scale_columns]
        )
        
        # Store scaled values without modifying raw data
        scaled_ds.scaled_values = scaled_values
        return scaled_ds

    def apply_post_filter(self, filter_func: Callable) -> 'BaseDataset | None':
        """
        Apply post-processing filter after transformations
        """
        # Get current data view (with transformations applied)
        current_data = self._get_transformed_data()
        mask = filter_func(current_data)
        new_indices = [self.valid_indices[i] for i in np.where(mask)[0]]
        return self._create_subset(new_indices)

    def _create_subset(self, indices: List[int]) -> 'BaseDataset':
        """Create new dataset with shared state"""
        return BaseDataset(
            self.raw_data,
            source_feature_names=self.source_feature_names,
            target_feature_names=self.target_feature_names,
            indices=indices,
            transform=self.transform
        )

    def _get_transformed_data(self) -> pd.DataFrame:
        """Get current data view with transformations applied"""
        # Start with raw data
        df = self.raw_data.loc[self.valid_indices].copy()
        
        # Apply scaling if available
        if hasattr(self, 'scaled_values') and self.scale_columns is not None:
            for i, col in enumerate(self.scale_columns):
                df[col] = self.scaled_values[:, i] # type: ignore
                
        # Apply feature transformations
        for feature, fn in self.transform.items():
            if feature in df.columns:
                df[feature] = df[feature].apply(fn)
                
        return df

    def __len__(self) -> int:
        """Number of samples in current view"""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve sample with transformations applied.
        Handles NA values by converting to NaN.
        """
        # Get absolute index
        abs_idx = self.valid_indices[idx]
        sample = self.raw_data.iloc[abs_idx]
        
        # Initialize result dict
        result = {}
        
        # Apply scaling if available
        if hasattr(self, 'scaled_values') and self.scaled_values is not None:
            scaled_sample = self.scaled_values[idx]
            for i, col in enumerate(self.scale_columns): # type: ignore
                result[col] = torch.tensor(scaled_sample[i], dtype=torch.float32)
        
        # Apply transformations and handle NA
        for feature in self.all_features:
            # Skip if already scaled
            if feature in result:
                continue
                
            value = sample[feature]
            
            # Convert NA to NaN
            if pd.isna(value):
                value = np.nan
                
            # Apply feature-specific transform
            if feature in self.transform:
                result[feature] = self.transform[feature](value)
            else:
                # Automatic type inference
                if isinstance(value, (float, np.floating)):
                    result[feature] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, (int, np.integer)):
                    result[feature] = torch.tensor(value, dtype=torch.long)
                else:
                    result[feature] = torch.tensor(value)
                    
        return result

    def get_data_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
        """Create DataLoader for current dataset view"""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    @property
    def shape(self) -> Tuple[int, int]:
        """Current dataset dimensions"""
        return len(self), len(self.all_features)

    def __repr__(self) -> str:
        """Informative string representation"""
        return (f"<BaseDataset: samples={len(self)}, "
                f"features={len(self.all_features)}, "
                f"scaled={hasattr(self, 'scaled_values')}>")
       
    
class BaseSlidingWindowDataset(BaseDataset):
    """
    Time-series dataset with sliding window functionality for sequence modeling.
    Inherits from BaseDataset and adds windowing capabilities for temporal data.
    
    Features:
    - Creates sliding windows from time-series data
    - Handles both univariate and multivariate time series
    - Supports various window configurations (fixed, expanding)
    - Maintains temporal ordering
    - Efficient window generation without data duplication
    
    Parameters:
    - timestamp_col: Column name containing timestamps
    - window_size: Number of time steps in each input window
    - horizon: Number of future steps to predict (0 for same-step prediction)
    - step_size: Number of steps between consecutive windows
    - window_type: 'fixed' (same size) or 'expanding' (growing window)
    - stride: Number of steps between consecutive sequence elements (for downsampling)
    - target_cols: Columns to use as prediction targets
    - drop_timestamp: Whether to exclude timestamp from features
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        timestamp_col: str,
        window_size: int,
        horizon: int = 1,
        step_size: int = 1,
        window_type: str = 'fixed',
        stride: int = 1,
        target_cols: Optional[List[str]] = None,
        drop_timestamp: bool = True,
        source_feature_names: Optional[List[str]] = None,
        target_feature_names: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        transform: Optional[Dict[str, Callable]] = None,
        pre_filter_func: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    ):
        """
        Initialize sliding window dataset.
        """
        super().__init__(
            data, 
            source_feature_names, 
            target_feature_names,
            indices,
            transform,
            pre_filter_func
        )
        
        # Validate parameters
        if timestamp_col not in self.raw_data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
            
        if window_size <= 0:
            raise ValueError("Window size must be positive")
            
        if horizon < 0:
            raise ValueError("Horizon must be non-negative")
            
        if step_size <= 0:
            raise ValueError("Step size must be positive")
            
        if window_type not in ['fixed', 'expanding']:
            raise ValueError("Window type must be 'fixed' or 'expanding'")
        
        # Store parameters
        self.timestamp_col = timestamp_col
        self.window_size = window_size
        self.horizon = horizon
        self.step_size = step_size
        self.window_type = window_type
        self.stride = stride
        self.target_cols = target_cols or []
        self.drop_timestamp = drop_timestamp
        
        # Sort data by timestamp
        self.raw_data = self.raw_data.sort_values(timestamp_col).reset_index(drop=True)
        
        # Generate window indices
        self.window_indices = self._generate_window_indices()
        
        # Precompute feature set
        self._precompute_features()
        
    def _precompute_features(self):
        """Determine final feature set after transformations"""
        # Start with all source features
        self.all_features = self.source_feature_names.copy()
        
        # Apply timestamp handling
        if self.drop_timestamp and self.timestamp_col in self.all_features:
            self.all_features.remove(self.timestamp_col)
            
        # Add any new features from transformations
        for feature in self.transform.keys():
            if feature not in self.all_features:
                self.all_features.append(feature)
    
    def _generate_window_indices(self) -> List[Tuple[List[int], int]]:
        """
        Generate window indices for the dataset.
        Returns list of (window_indices, target_index) tuples.
        """
        indices = []
        n = len(self.raw_data)
        sorted_indices = self.raw_data.index.tolist()
        
        for i in range(0, n - self.window_size - self.horizon + 1, self.step_size):
            # Determine window indices
            if self.window_type == 'fixed':
                window_start = i
                window_end = i + self.window_size
                window_indices = sorted_indices[window_start:window_end:self.stride]
            else:  # Expanding window
                window_indices = sorted_indices[:i + self.window_size:self.stride]
            
            # Skip incomplete windows
            if len(window_indices) < self.window_size // self.stride:
                continue
                
            # Determine target index
            target_index = sorted_indices[i + self.window_size + self.horizon - 1]
            
            indices.append((window_indices, target_index))
            
        return indices

    def __len__(self) -> int:
        """Number of windows in the dataset"""
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a window and its corresponding target.
        Returns (window_data, target) tuple.
        """
        window_indices, target_index = self.window_indices[idx]
        
        # Get window data
        window_df = self.raw_data.loc[window_indices]
        
        # Apply transformations to entire window
        transformed_window = []
        for _, row in window_df.iterrows():
            transformed_row = {}
            for feature in self.all_features:
                if feature in self.transform:
                    transformed_row[feature] = self.transform[feature](row)
                else:
                    transformed_row[feature] = row[feature]
            transformed_window.append(transformed_row)
        
        # Convert to tensor (window_size, num_features)
        window_data = []
        for row in transformed_window:
            row_data = [row[feature] for feature in self.all_features]
            window_data.append(row_data)
            
        window_tensor = torch.tensor(window_data, dtype=torch.float32)
        
        # Get target
        target_row = self.raw_data.loc[target_index]
        if self.target_cols:
            target_values = [target_row[col] for col in self.target_cols]
            target_tensor = torch.tensor(target_values, dtype=torch.float32)
        else:
            # Default to last feature if no targets specified
            target_tensor = torch.tensor(target_row[self.all_features[-1]], dtype=torch.float32)
        
        return window_tensor, target_tensor

    def get_sequence_length(self) -> int:
        """Get the sequence length (number of time steps)"""
        return self.window_size // self.stride

    def get_num_features(self) -> int:
        """Get the number of features per time step"""
        return len(self.all_features)

    def get_window(self, idx: int, as_dataframe: bool = False) -> Union[pd.DataFrame, torch.Tensor]:
        """
        Get a window of data as a DataFrame or tensor.
        
        :param idx: Window index
        :param as_dataframe: Return as DataFrame if True, else tensor
        :return: Window data
        """
        window_indices, _ = self.window_indices[idx]
        window_df = self.raw_data.loc[window_indices]
        
        if as_dataframe:
            return window_df
        else:
            return self[idx][0]

    def get_timestamps(self, idx: int) -> List[Any]:
        """
        Get timestamps for a window.
        
        :param idx: Window index
        :return: List of timestamps in the window
        """
        window_indices, _ = self.window_indices[idx]
        return self.raw_data.loc[window_indices, self.timestamp_col].tolist()

    def visualize_window(self, idx: int, feature: str) -> None:
        """
        Simple visualization of a feature across a window.
        
        :param idx: Window index
        :param feature: Feature to visualize
        """
        import matplotlib.pyplot as plt
        
        window_indices, target_index = self.window_indices[idx]
        timestamps = self.raw_data.loc[window_indices, self.timestamp_col]
        values = self.raw_data.loc[window_indices, feature]
        target_ts = self.raw_data.loc[target_index, self.timestamp_col]
        target_val = self.raw_data.loc[target_index, feature]
        
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, values, 'bo-', label=feature)
        plt.plot(target_ts, target_val, 'ro', label='Target')
        plt.axvline(x=target_ts, color='r', linestyle='--', alpha=0.5)
        plt.title(f"Window {idx}: {feature}")
        plt.xlabel("Timestamp")
        plt.ylabel(feature)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        
        
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA
from typing import List, Tuple, Optional
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

class BiomedicalPainDataset(Dataset):
    def __init__(
        self,
        signals_df: pd.DataFrame,
        pain_df: pd.DataFrame,
        window_duration: float,  # in seconds
        step_size: float,         # in seconds
        signal_columns: List[str],
        bandpass_freqs: Tuple[float, float] = (0.5, 40),
        freq_bands: List[Tuple[float, float]] = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)],
        resample_freq: int = 100, # Hz
        apply_ica: bool = False,
        ica_components: Optional[int] = None,
        time_difference:int = 4,
        unit_signals_ts = 'us',
    ):
        """
        Dataset for biomedical signals with asynchronous pain reports.
        
        Args:
            signals_df: DataFrame with columns including 'timestamp' and signal columns
            pain_df: DataFrame with columns 'Timestamp' and 'PainLevel'
            window_duration: Window length in seconds
            step_size: Sliding window step size in seconds
            signal_columns: List of signal column names to process
            bandpass_freqs: Tuple of (low, high) frequencies for bandpass filter
            freq_bands: Frequency bands for power calculation
            resample_freq: Common sampling frequency for resampling
            apply_ica: Whether to apply ICA for artifact removal
            ica_components: Number of ICA components (if None, use all)
        """
        # Convert timestamps to datetime and sort
        self.signals_df = signals_df.copy()
        self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'], unit=unit_signals_ts) 
        self.signals_df['timestamp'] = self.signals_df['timestamp'] - np.timedelta64(time_difference, 'h')
        self.signals_df.sort_values('timestamp', inplace=True)
        
        self.pain_df = pain_df.copy()
        self.pain_df['Timestamp'] = pd.to_datetime(self.pain_df['Timestamp'])
        self.pain_df = self.pain_df.dropna(subset=['PainLevel'])
        self.pain_df.sort_values('Timestamp', inplace=True)
        
        # Store parameters
        self.window_duration = window_duration
        self.step_size = step_size
        self.signal_columns = signal_columns
        self.bandpass_freqs = bandpass_freqs
        self.freq_bands = freq_bands
        self.resample_freq = resample_freq
        self.n_samples_per_window = int(window_duration * resample_freq)
        self.apply_ica = apply_ica
        self.ica_components = ica_components
        self.feature_names = self._generate_feature_names()
        
        # Precompute valid window start times
        self.window_starts = self._precompute_window_starts()
        
    def _generate_feature_names(self):
        """Generate descriptive feature names for ANOVA"""
        names = []
        for col in self.signal_columns:
            for band in self.freq_bands:
                names.append(f"{col}_{band[0]}-{band[1]}Hz")
        return names    
        
    def _precompute_window_starts(self):
        """Generate valid window start times with sufficient signal data coverage"""
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
        
        # Compute pain label
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
        """Apply Butterworth bandpass filter"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def _process_signals(self, start, end):
        """Process signals: resample, filter, (optional ICA), and extract frequency features"""
        # Extract relevant signals in the window
        window_signals = self.signals_df[
            (self.signals_df['timestamp'] >= start) & 
            (self.signals_df['timestamp'] < end)
        ]
        
        # Create common time grid for resampling
        t_start = start.timestamp()
        t_end = end.timestamp()
        t_grid = np.linspace(t_start, t_end, self.n_samples_per_window, endpoint=False)
        
        # Process each signal channel
        processed_signals = []
        for col in self.signal_columns:
            # Extract valid samples for this channel
            col_data = window_signals[['timestamp', col]].dropna()
            
            if col_data.empty:
                # No data - return zeros
                processed_signals.append(np.zeros(self.n_samples_per_window))
                continue
                
            # Convert to numeric timestamps
            t_valid = col_data['timestamp'].astype(np.int64).values / 1e9
            y_valid = col_data[col].values
            
            # Skip if only one data point
            if len(t_valid) < 2:
                processed_signals.append(np.zeros(self.n_samples_per_window))
                continue
                
            # Interpolate to common grid
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
        
        # Stack signals into matrix (channels x time)
        signal_matrix = np.vstack(processed_signals)
        
        # Apply ICA if requested
        if self.apply_ica and signal_matrix.shape[0] > 1:
            ica = FastICA(n_components=self.ica_components, random_state=0)
            try:
                signal_matrix = ica.fit_transform(signal_matrix.T).T
            except Exception as e:
                print(f"ICA failed: {e}")
                # Fall back to original signals if ICA fails
        
        # Compute frequency features
        features = []
        for signal in signal_matrix:
            # Skip if signal is all zeros
            if np.all(signal == 0):
                features.append(np.zeros(len(self.freq_bands)))
                continue
                
            # Compute power spectral density
            freqs, psd = welch(
                signal, 
                fs=self.resample_freq, 
                nperseg=min(256, len(signal)))
            
            # Calculate band powers
            band_powers = []
            for low, high in self.freq_bands:
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_power = np.trapezoid(psd[mask], freqs[mask])
                else:
                    band_power = 0.0
                band_powers.append(band_power)
                
            features.append(band_powers)
        
        # Flatten features and convert to tensor
        features = np.concatenate(features)
        return torch.tensor(features, dtype=torch.float32)

    def _compute_pain_label(self, start, end):
        """Compute weighted average pain level for the window"""
        # Get relevant pain reports
        prev_reports = self.pain_df[self.pain_df['Timestamp'] <= end]
        if prev_reports.empty: return 0.0  # Default if no pain reports
        
        # Include next report after window for boundary calculation
        next_report = self.pain_df[self.pain_df['Timestamp'] > end].head(1)
        all_reports = pd.concat([prev_reports, next_report]).sort_values('Timestamp')
        
        # Create time segments between reports
        segments = []
        for i in range(len(all_reports) - 1):
            seg_start = all_reports.iloc[i]['Timestamp']
            seg_end = all_reports.iloc[i+1]['Timestamp']
            pain_level = all_reports.iloc[i]['PainLevel']
            
            # Clip segment to current window
            seg_start = max(seg_start, start)
            seg_end = min(seg_end, end)
            
            if seg_start < seg_end:
                duration = (seg_end - seg_start).total_seconds()
                segments.append((duration, pain_level))
        
        # Calculate weighted average
        total_duration = 0
        weighted_sum = 0
        for duration, pain in segments:
            total_duration += duration
            weighted_sum += duration * pain
            
        return weighted_sum / total_duration if total_duration > 0 else 0.0
    
    def perform_anova(self, max_samples_per_class=1000):
        """Perform ANOVA on signal features across pain categories"""
        # Collect data for each pain category
        categories = {0: [], 1: [], 2: []}
        category_counts = {0: 0, 1: 0, 2: 0}
        
        for i in range(len(self)):
            features, pain_cat, _ = self[i]
            features = features.numpy()
            
            if category_counts[pain_cat] < max_samples_per_class:
                categories[pain_cat].append(features)
                category_counts[pain_cat] += 1
                
            # Stop if we have enough samples for all categories
            if all(count >= max_samples_per_class for count in category_counts.values()):
                break
        
        # Prepare data for ANOVA
        data_by_category = [np.array(categories[i]) for i in range(3)]
        n_features = data_by_category[0].shape[1]
        
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
        self._plot_anova_results(f_values, p_values, significant_features)
        
        return f_values, p_values, significant_features

    def _plot_anova_results(self, f_values, p_values, significant_features):
        """Visualize ANOVA results"""
        plt.figure(figsize=(15, 10))
        
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
        plt.subplot(2, 1, 2)
        top_features = 20
        sorted_idx = np.argsort(f_values)[::-1]
        top_f = f_values[sorted_idx][:top_features]
        top_names = [self.feature_names[i] for i in sorted_idx[:top_features]]
        
        plt.barh(top_names, top_f, color='skyblue')
        plt.xlabel('F-value')
        plt.title(f'Top {top_features} Features by F-value')
        plt.tight_layout()
        
        plt.savefig('anova_results.png', dpi=300)
        plt.show()