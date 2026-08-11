import numpy as np
from omegaconf import DictConfig
import os
import pickle as pkl
import torch
from typing import List, Tuple

from vsa_ogm.data.sa.base_sa_dataset import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger

class AblationSingleAgentDataset(BaseSingleAgentDataset):
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, loggers: List[BaseLogger]) -> None:
        """
        Initialize the AblationSingleAgentDataset object.

        Args:
            config (DictConfig): The configuration object containing the
                dataset information.
            loggers (List[BaseLogger]): A list of loggers to log information.
        """
        super(AblationSingleAgentDataset, self).__init__(config, loggers)

        data_dir: str = config.data.data_dir
        prefix: str = config.data.file_prefix
        suffix: str = config.data.file_suffix

        file_path: str = os.path.join(data_dir, f"{prefix}{suffix}")
        with open(file_path, "rb") as f:
            data: dict = pkl.load(f)

        occ_lidar_x: np.ndarray = data["occ_lidar_x"]
        occ_lidar_y: np.ndarray = data["occ_lidar_y"]
        empty_lidar_x: np.ndarray = data["empty_lidar_x"]
        empty_lidar_y: np.ndarray = data["empty_lidar_y"]
        
        points_x = np.concatenate((occ_lidar_x, empty_lidar_x), axis=0)
        points_y = np.concatenate((occ_lidar_y, empty_lidar_y), axis=0)

        # convert points into an arrays
        all_points = np.stack((points_x, points_y), axis=1)
        all_labels = np.concatenate((
            np.ones(occ_lidar_x.shape[0]),
            np.zeros(empty_lidar_x.shape[0])
        ))

        # shuffle the data
        # Seeded from the config so the chunking is reproducible across runs;
        # this used to draw from global numpy state, so every run produced a
        # different partition of points into scans.
        rng = np.random.default_rng(config.mapping.seed)
        shuffled_indices = rng.permutation(all_points.shape[0])
        all_points = all_points[shuffled_indices]
        all_labels = all_labels[shuffled_indices]

        # split the data into point cloud chunks
        self.points = np.array_split(all_points, config.data.num_chunks)
        self.labels = np.array_split(all_labels, config.data.num_chunks)

        self.map_size: list[int] = config.data.world_bounds
        self.step_limit: int = len(self.points) - 1
        self.time_step: int = -1

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Return the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            Tuple: A tuple containing the data and the world size.
        """
        data_batch: dict = {
            "lidar_data": self.points[idx],
            "lidar_distances": None,
            "occupancy": self.labels[idx],
            "max_laser_distance": None,
            "robot_poses": None
        }

        return data_batch