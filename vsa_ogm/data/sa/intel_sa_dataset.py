from glob import glob
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from typing import List, Tuple

from vsa_ogm.data.sa.base_sa_dataset import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger

class IntelSingleAgentDataset(BaseSingleAgentDataset):
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, loggers: List[BaseLogger]) -> None:
        """
        Initialize the IntelSingleAgentDataset object.

        Args:
            config (DictConfig): The configuration object containing the
                dataset information.
            loggers (List[BaseLogger]): A list of loggers to log information.
        """
        super(IntelSingleAgentDataset, self).__init__(config, loggers)

        self.data: pd.DataFrame = pd.read_csv(config.data.data_dir)

        # get all of the unique timestamps with the data
        time_steps: pd.DataFrame = self.data.iloc[:, 0].unique()
        self.num_time_steps: int = time_steps.shape[0]

        self.point_clouds: list = []
        self.time_step: int = -1

        # get the point clouds for each time step
        for ts in range(self.num_time_steps):
            rows: pd.DataFrame = self.data[self.data.iloc[:, 0] == time_steps[ts]]
            pc: np.ndarray = rows.values[:, 1:4]
            self.point_clouds.append(pc)

        self.map_size: list[int] = config.data.world_bounds
        self.step_limit: int = len(self.point_clouds) - 1

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        # NOTE: this used to return self.step_limit (len - 1), which made the
        # final scan unreachable. The dataset has 910 scans, matching Table 1
        # of the paper; returning step_limit silently used only 909 of them.
        return len(self.point_clouds)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Return the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            Tuple: A tuple containing the data and the world size.
        """
        data_batch: dict = {
            "lidar_data": torch.from_numpy(self.point_clouds[idx][:, :2]),
            "lidar_distances": None,
            "occupancy": torch.from_numpy(self.point_clouds[idx][:, 2]),
            "max_laser_distance": None,
            "robot_poses": None
        }

        return data_batch