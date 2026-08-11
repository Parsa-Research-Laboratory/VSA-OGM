from glob import glob
import numpy as np
from omegaconf import DictConfig
import os
import torch
from typing import List, Tuple

from vsa_ogm.data.sa.base_sa_dataset import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger

class ToySimSingleAgentDataset(BaseSingleAgentDataset):
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, loggers: List[BaseLogger]) -> None:
        """
        Initialize the ToySimDataset object.

        Args:
            config (DictConfig): The configuration object containing the
                dataset information.
            loggers (List[BaseLogger]): A list of loggers to log information.
        """
        super(ToySimSingleAgentDataset, self).__init__(config, loggers)

        file_path: str = config.data.data_dir
        prefix: str = config.data.file_prefix
        suffix: str = config.data.file_suffix

        mask: str = prefix + "*" + suffix
        file_path_mask: str = os.path.join(file_path, mask)
        files: list[str] = glob(file_path_mask)
        
        # sort the list of filepaths by the digits in their headers
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        print(f"Found {len(files)} files in {file_path} with prefix {prefix} and suffix {suffix}")

        self.files: list[str] = files
        self.map_size: list[int] = config.data.world_bounds
        self.step_limit: int = len(self.files) - 1
        self.time_step: int = -1

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        # NOTE: this used to return self.step_limit (len - 1), which made the
        # final scan unreachable. The dataset has 90 scans, matching Table 1
        # of the paper; returning step_limit silently used only 89 of them.
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Return the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            Tuple: A tuple containing the data and the world size.
        """
        measurement_file = np.load(self.files[idx])

        distance_data: np.ndarray = measurement_file["dist_theta_at_t"]
        laser_data: np.ndarray = measurement_file["laser_data_xy_at_t"]
        max_laser_distance: float = measurement_file["max_laser_distance"]
        pose_data: np.ndarray = measurement_file["all_robot_poses"][idx, :]

        measurement_file.close()

        occupancy = np.zeros((laser_data.shape[0],))
        occupancy[np.where(distance_data != max_laser_distance)] = 1.0

        data_batch: dict = {
            "lidar_data": torch.from_numpy(laser_data),
            "lidar_distances": torch.from_numpy(distance_data),
            "occupancy": torch.from_numpy(occupancy),
            "max_laser_distance": torch.from_numpy(max_laser_distance),
            "robot_poses": torch.from_numpy(pose_data)
        }

        return data_batch