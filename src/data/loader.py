from __future__ import annotations

import os
import io
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from src.config import conf


class WISDMSplits:
    """
    Loads WISDM walking ("A") windows and prepares train/val/test splits.
    """

    def __init__(
        self,
        base_path: str,
        sensor_device: str = conf.sensor_device,
        signal_type: str = conf.signal_type,
        window: int = 20,
        step: int = 10,
        test_size: float = 0.1,
        val_size: float = 0.2,
        seed: int = 42,
        activity_code: str = "A",
    ):
        X, y = self.load_signals(
            base_path=base_path,
            sensor_device=sensor_device,
            signal_type=signal_type,
            window=window,
            step=step,
            activity_code=activity_code,
        )

        if X is None or y is None or len(X) == 0:
            raise RuntimeError(
                "No windows were produced. Check dataset path / subject files / window+step settings."
            )

        # split: train vs test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # split: train vs val (from train)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=seed,
            stratify=y_train,
        )

        # torch tensors
        self.train_segments = torch.tensor(X_train, dtype=torch.float32)
        self.val_segments = torch.tensor(X_val, dtype=torch.float32)
        self.test_segments = torch.tensor(X_test, dtype=torch.float32)

        self.train_labels = torch.tensor(y_train, dtype=torch.long)
        self.val_labels = torch.tensor(y_val, dtype=torch.long)
        self.test_labels = torch.tensor(y_test, dtype=torch.long)

        # handy TensorDatasets
        self.train_ds = TensorDataset(self.train_segments, self.train_labels)
        self.val_ds = TensorDataset(self.val_segments, self.val_labels)
        self.test_ds = TensorDataset(self.test_segments, self.test_labels)

    def load_signals(
        self,
        base_path: str,
        sensor_device: str,
        signal_type: str,
        window: int = 20,
        step: int = 10,
        activity_code: str = "A",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          data: (num_windows, window, 3)
          labels: (num_windows,)
        """
        source_dir = os.path.join(base_path, "raw", sensor_device, signal_type)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"directory not found: {source_dir}")

        dtype = [
            ("activity_code", "U1"),
            ("timestamp", int),
            ("x", float),
            ("y", float),
            ("z", float),
        ]

        labels = None
        data = None

        subject_ids = list(conf.subject_ids)

        for subject_id in subject_ids:
            file_path = os.path.join(
                source_dir, f"data_{subject_id}_{signal_type}_{sensor_device}.txt"
            )
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"file not found: {file_path}")

            with open(file_path, "r") as f:
                content = f.read().replace(";", "\n")

            dataframe = np.genfromtxt(
                io.StringIO(content),
                dtype=dtype,
                delimiter=",",
                usecols=[1, 2, 3, 4, 5],
            )
            filtered = dataframe[dataframe["activity_code"] == activity_code]

            xyz = np.stack(
                [filtered["x"], filtered["y"], filtered["z"]], axis=1
            ).astype(np.float32)
            size = xyz.shape[0]
            if size < window:
                continue

            starts = range(0, size - window + 1, step)
            windows = np.stack([xyz[s : s + window] for s in starts], axis=0)

            class_id = subject_ids.index(subject_id)
            new_labels = np.full((windows.shape[0],), class_id, dtype=np.int64)

            data = windows if data is None else np.concatenate([data, windows], axis=0)
            labels = (
                new_labels
                if labels is None
                else np.concatenate([labels, new_labels], axis=0)
            )

        return data, labels
