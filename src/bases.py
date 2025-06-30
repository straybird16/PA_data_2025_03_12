from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
from collections.abc import Callable
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset

class BaseDataset(ABC):
    def __init__(self,) -> None:
        super().__init__()
        """
        output dimension: 1-4
        output_type: torch tensor or numpy array
        normalization method: 
            normalize thru 
        sampling strategy: 
            sampling from the dataset directly (up- or down-sample data), which is stable,
            or generate a sampler for the dataloader
        
        """
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        pass
    
    @abstractmethod
    def get_labels(self) -> List[int]:
        """Get all labels for sampling strategies"""
        pass

class BaseDataLoader(ABC):
    def __init__(self, dataset: BaseDataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, window: np.ndarray) -> np.ndarray:
        pass


class SlidingWindowDataset(BaseDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        step_size: int,
        feature_extractor: BaseFeatureExtractor,
        target_col: str|None = None,
        label_mapper: Callable|None = None,
    ):
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.target_col = target_col
        self.label_mapper = label_mapper
        self.indices = self._generate_indices()
        
    def _generate_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for start in range(0, len(self.data) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            indices.append((start, end))
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Union[int, None]]:
        start, end = self.indices[idx]
        window = self.data.iloc[start:end]
        
        # Extract features for each column
        features = []
        for col in self.data.columns:
            if col == self.target_col:
                continue
            col_features = self.feature_extractor.extract(window[col].values)
            features.append(col_features)
        
        features = np.concatenate(features)
        
        # Get target if available
        target = None
        if self.target_col:
            target = window[self.target_col].iloc[-1]  # Last value in window
            if self.label_mapper:
                target = self.label_mapper(target)
        
        return features, target
    
    def get_labels(self) -> List[int]:
        if not self.target_col:
            raise ValueError("Dataset doesn't have target column")
            
        labels = []
        for start, end in self.indices:
            target = self.data[self.target_col].iloc[end-1]
            if self.label_mapper:
                target = self.label_mapper(target)
            labels.append(target)
        return labels
    
    
class TimeSeriesDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        step_size: int,
        feature_extractor: BaseFeatureExtractor,
        target_col: str|None = None,
        label_mapper: Callable|None = None,
    ):
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.target_col = target_col
        self.label_mapper = label_mapper
        self.indices = self._generate_indices()
        
    def _generate_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for start in range(0, len(self.data) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            indices.append((start, end))
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Union[int, None]]:
        start, end = self.indices[idx]
        window = self.data.iloc[start:end]
        
        # Extract features for each column
        features = []
        for col in self.data.columns:
            if col == self.target_col:
                continue
            col_features = self.feature_extractor.extract(window[col].values)
            features.append(col_features)
        
        features = np.concatenate(features)
        
        # Get target if available
        target = None
        if self.target_col:
            target = window[self.target_col].iloc[-1]  # Last value in window
            if self.label_mapper:
                target = self.label_mapper(target)
        
        return features, target
    
    def get_labels(self) -> List[int]:
        if not self.target_col:
            raise ValueError("Dataset doesn't have target column")
            
        labels = []
        for start, end in self.indices:
            target = self.data[self.target_col].iloc[end-1]
            if self.label_mapper:
                target = self.label_mapper(target)
            labels.append(target)
        return labels