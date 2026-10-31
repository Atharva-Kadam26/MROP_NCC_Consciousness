"""Harvard BDSP seizure/spike dataset loader, matching TUEV's pickle interface."""

import os
import pickle
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset

from downstream_tasks.position_utils import load_positions


class HarvardSeizure(Dataset):
    def __init__(
        self,
        path: str,
        mode: str,
        positions=None,
        electrodes=None,
        scale_factor: float = 1e4,
    ):
        self.scale_factor = scale_factor

        assert mode in ["train", "val", "test"]
        if mode == "test":
            self.data_path = pjoin(path, "processed_eval")
            self.file_names = sorted([f for f in os.listdir(self.data_path) if not f.startswith(".")])
        elif mode in {"train", "val"}:
            self.data_path = pjoin(path, "processed_train")
            all_files = sorted([f for f in os.listdir(self.data_path) if not f.startswith(".")])

            rng = np.random.default_rng(seed=42)
            rng.shuffle(all_files)

            n_train = int(0.8 * len(all_files))
            if mode == "train":
                self.file_names = all_files[:n_train]
            else:
                self.file_names = all_files[n_train:]

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes).float()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        X = torch.tensor(data["signal"])
        Y = torch.tensor(data["label"]) - 1

        return {
            "sample": X.float() * self.scale_factor,
            "label": Y.long(),
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([x["sample"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "pos": self.positions.unsqueeze(0).repeat(len(batch), 1, 1),
        }
