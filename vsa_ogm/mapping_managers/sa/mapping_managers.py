import numpy as np
from omegaconf import DictConfig
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
from typing import List

from vsa_ogm.data.sa import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger
from vsa_ogm.mappers.sa import BaseSingleAgentMapper

VALID_METRICS: List[str] = ["auc", "f1", "precision", "recall", "accuracy", "nll"]

class SingleAgentMappingManager:
    """
    TODO Finish Documentation
    """

    # annotations only -- these accumulate per run, so a shared class-level
    # list would leak one manager's data into the next manager's metrics.
    all_X_train: List[np.ndarray]
    all_y_train: List[np.ndarray]
    all_X_test: List[np.ndarray]
    all_y_test: List[np.ndarray]
    mapper: BaseSingleAgentMapper = None

    def __init__(self, config: DictConfig, loggers: List[BaseLogger],
                 print_header: str = "(Mapping Manager)") -> None:
        """
        Initializes a MappingManager object.

        Args:
            config (DictConfig): The configuration object containing the
                mapping information.
        """
        self.config: DictConfig = config
        self.loggers: List[BaseLogger] = loggers
        self.print_header: str = print_header
        self.seed: int = config.mapping_manager.seed
        self.test_size: float = config.data.test_split
        self.verbose: bool = config.mapping_manager.verbose

        # extract parameters from the config
        self.metrics: List[str] = config.mapping_manager.metrics

        # check if the metrics are valid
        for metric in self.metrics:
            if metric not in VALID_METRICS:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {VALID_METRICS}")
            
        self.plotting_flags: DictConfig = config.mapping_manager.plotting_flags
        self.saving_flags: DictConfig = config.mapping_manager.saving_flags

        self.all_X_train: List[np.ndarray] = []
        self.all_y_train: List[np.ndarray] = []
        self.all_X_test: List[np.ndarray] = []
        self.all_y_test: List[np.ndarray] = []

        self._initialize_mapper()

    def run(self, dataset: BaseSingleAgentDataset) -> None:
        """
        Run the MappingManager on the given dataset.

        Args:
            dataset (BaseSingleAgentDataset): The dataset to run the
                MappingManager on.
        """

        dataset_length: int = len(dataset)

        # log the start of the mapping manager
        string = f"{self.print_header} Running Mapping Manager on dataset with {dataset_length} steps."
        if self.verbose:
            print(string)
        for logger in self.loggers:
            logger.log_string(string)

        # iterate through the dataset
        for idx in range(dataset_length):
            # initialize per iteration variables
            complete_metric_dict: dict = {}
            complete_images_dict: dict = {}
            complete_figures_dict: dict = {}

            # log the current step
            string = f"{self.print_header} Processing dataset step {idx}."
            if self.verbose:
                print(string)
            for logger in self.loggers:
                logger.log_string(string)

            # get the data at the current index
            data_batch: dict = dataset[idx]
            X: np.ndarray = data_batch["lidar_data"]
            y: np.ndarray = data_batch["occupancy"]

            # split the data into stratified training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=y
            )

            if self.plotting_flags.plot_point_clouds:
                for logger in self.loggers:
                    logger.log_point_cloud(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        title=f"Dataset_Step_{idx}",
                        epoch=idx)

            # store the training and testing data
            self.all_X_train.append(X_train)
            self.all_y_train.append(y_train)
            self.all_X_test.append(X_test)
            self.all_y_test.append(y_test)

            # run the mapper on the data
            start_time = time.time()
            fit_metrics, intermediate_maps = self.mapper.fit(X_train, y_train)
            end_time = time.time()
            fit_time = end_time - start_time
            print(f"Fit time: {fit_time:.2f} seconds")
            complete_metric_dict.update(fit_metrics)

            # predict the testing data
            all_y_test_np = np.concatenate(self.all_y_test)
            all_X_test_np = np.vstack(self.all_X_test)

            
            y_pred, pred_metrics = self.mapper.predict(all_X_test_np)
            complete_metric_dict.update(pred_metrics)

            # normalize the predictions based on the range of the OGM
            ogm_min = np.min(self.mapper.ogm)
            ogm_max = np.max(self.mapper.ogm)
            train_pred = self.mapper.predict(X_train)[0]
            x_train_pred_norm = (train_pred - ogm_min) / (ogm_max - ogm_min)
            x_test_pred_norm = (y_pred - ogm_min) / (ogm_max - ogm_min)
            x_train_pred_norm = np.nan_to_num(x_train_pred_norm)
            x_test_pred_norm = np.nan_to_num(x_test_pred_norm)

            # calculate the performance metrics based on the predictions
            for metric in self.metrics:
                if metric == "auc":
                    train_fpr, train_tpr, _ = metrics.roc_curve(y_train, x_train_pred_norm)
                    test_fpr, test_tpr, _ = metrics.roc_curve(all_y_test_np, x_test_pred_norm)
                    train_auc = metrics.auc(train_fpr, train_tpr)
                    test_auc = metrics.auc(test_fpr, test_tpr)
                    complete_metric_dict["train_auc"] = train_auc
                    complete_metric_dict["test_auc"] = test_auc

                elif metric == "nll":
                    train_nll = metrics.log_loss(y_train, x_train_pred_norm, labels=[0, 1])
                    test_nll = metrics.log_loss(all_y_test_np, x_test_pred_norm, labels=[0, 1])
                    complete_metric_dict["train_nll"] = train_nll
                    complete_metric_dict["test_nll"] = test_nll
                else:
                    raise NotImplementedError(f"Metric {metric} not implemented.")

            if self.plotting_flags.plot_point_clouds:
                for logger in self.loggers:
                    logger.log_point_cloud(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        title=f"TestingPredictions_{idx}",
                        epoch=idx)

            if idx % self.config.logging.log_interval == 0 or idx == dataset_length - 1:
                if self.plotting_flags.plot_images:
                    for logger in self.loggers:
                        logger.log_image(self.mapper.ogm, caption="OGM", epoch=idx)

                    for key, value in intermediate_maps.items():
                        if value is not None:
                            for logger in self.loggers:
                                logger.log_image(value, caption=key, epoch=idx)

                if hasattr(self.mapper, "occupied_quadrant_memory_vectors") and hasattr(self.mapper, "empty_quadrant_memory_vectors"):
                    if self.saving_flags.save_memory_vectors:
                        for logger in self.loggers:
                            logger.log_matrix(self.mapper.occupied_quadrant_memory_vectors.cpu().numpy(), caption="occupied_tile_memories", epoch=idx)
                            logger.log_matrix(self.mapper.empty_quadrant_memory_vectors.cpu().numpy(), caption="empty_tile_memories", epoch=idx)


                if self.saving_flags.save_image_matrices:
                    for logger in self.loggers:
                        logger.log_matrix(self.mapper.ogm, caption="OGM", epoch=idx)

                    for key, value in intermediate_maps.items():
                        if value is not None:
                            for logger in self.loggers:
                                logger.log_matrix(value, caption=key, epoch=idx)

            # log the metrics
            for logger in self.loggers:
                logger.log_metrics(complete_metric_dict, idx)

    def _initialize_mapper(self) -> None:
        """
        Initialize the mapper based on the configuration.
        """
        mapper_type: str = self.config.mapping.mapping_type
        if mapper_type == "SA_VSA_OGM":
            from vsa_ogm.mappers.sa.sa_vsa_mapper import SA_VSA_OGM
            self.mapper = SA_VSA_OGM(self.config, self.loggers)
        elif mapper_type == "SA_BHM_DIAG":
            from vsa_ogm.mappers.sa.sa_bhm_diag_mapper import SA_BHM_DIAG
            self.mapper = SA_BHM_DIAG(self.config, self.loggers)
        else:
            raise ValueError(f"Invalid mapper type: {mapper_type}.")


