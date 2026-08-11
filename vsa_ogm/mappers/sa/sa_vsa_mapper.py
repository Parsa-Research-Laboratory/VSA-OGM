import json
import numpy as np
from omegaconf import DictConfig
import os
from skimage.filters.rank import entropy
from skimage.morphology import disk
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


from .base_sa_mapper import BaseSingleAgentMapper
from ...logging import BaseLogger

def sum_nested_dict(d):
    total = 0
    for value in d.values():
        if isinstance(value, dict):  # If value is a nested dictionary, recurse
            total += sum_nested_dict(value)
        else:  # Otherwise, add the numeric value
            total += value
    return total

def make_good_unitary(num_dims: int, device: str,
        eps: float = 1e-3, rng: np.random.Generator = None) -> torch.tensor:
    """
    create a hyperdimensional vector of unitary length phasers to build the
    quasi-orthogonal algebraic space

    Arguments:
    ----------
    1) num_dims (int): the dimensionality of the vsa
    2) device (str): where to store the tensor
    3) eps (float): the allowable variability in the phase of each phasor
    4) rng (np.random.Generator): the generator used to draw the phasor signs.
        Must be seeded identically across every agent whose memory vectors
        will later be fused, otherwise the agents encode into different
        algebraic spaces and their memories cannot be summed.

    Returns:
    --------
    1) v (torch.tensor): a one dimensional tensor of unitary phasors
    """

    if rng is None:
        rng = np.random.default_rng()

    a = torch.rand((num_dims - 1) // 2)
    sign = rng.choice((-1, +1), len(a))

    sign = torch.from_numpy(sign).to(device)
    a = a.to(device)

    phi = sign * torch.pi * (eps + a * (1 - 2 * eps))

    assert torch.all(torch.abs(phi) >= torch.pi * eps)
    assert torch.all(torch.abs(phi) <= torch.pi * (1 - eps))

    fv = torch.zeros(num_dims, dtype=torch.complex64, device=device)
    fv[0] = 1
    fv[1:(num_dims + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
    fv[(num_dims // 2) + 1:] = torch.flip(torch.conj(fv[1:(num_dims + 1) // 2]), dims=[0])
    
    if num_dims % 2 == 0:
        fv[num_dims // 2] = 1

    assert torch.allclose(torch.abs(fv), torch.ones(fv.shape, device=device))
    
    v = torch.fft.ifft(fv)
    v = v.real
    v = v.to(device)
    
    assert torch.allclose(torch.fft.fft(v), fv)
    assert torch.allclose(torch.linalg.norm(v), torch.ones(v.shape, device=device))

    return v


@torch.jit.script
def compute_local_entropy(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    tensor = tensor.clamp(0, 1)

    bins = 256
    quantized = (tensor * (bins - 1)).long()

    one_hot = F.one_hot(quantized, num_classes=bins).permute(2, 0, 1).float().unsqueeze(0)

    diameter = 2 * radius + 1
    y = torch.arange(diameter, device=tensor.device)
    x = torch.arange(diameter, device=tensor.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    center = radius
    disk_kernel = ((xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2).float()
    disk_kernel = disk_kernel / disk_kernel.sum()

    kernel = disk_kernel.expand(bins, 1, diameter, diameter).contiguous()

    local_hist = F.conv2d(one_hot, kernel, padding=radius, groups=bins)
    local_hist = local_hist.clamp(min=1e-10)

    entropy_map = -(local_hist * torch.log2(local_hist)).sum(dim=1)

    return entropy_map.squeeze(0)

@torch.compile
def compute_mm(norm_qv, xy_axis_matrix):
    return torch.einsum('nm,xym->nxy', norm_qv, xy_axis_matrix)


class SSPGenerator:
    """
    A Utility class to generate arbitrary numbers of hyper-vectors with the
    same shape so they can be binded and bundled together
    """
    def __init__(self, dimensionality: int, device: str, length_scale: float = 1,
                 seed: int = 0) -> None:
        """
        Init SSP Generator

        Arguments:
        ----------
        1) dimensionality (int): the number of dimensions contained within
            each hypervector
        2) device (str): a string representing the device to load, store,
            and operate
        3) length_scale (float): adjust the width of the kernel
        4) seed (int): seeds both the torch and numpy draws so that the same
            axis vectors are produced on every call, in every process. Agents
            whose memory vectors will be fused must share this seed.

        Returns:
        --------
        None
        """
        self.dimensionality: int = dimensionality
        self.device: str = device
        self.length_scale: float = length_scale
        self.seed: int = seed

    def generate(self, n: int) -> torch.tensor:
        """
        Randomly create a series of n hypervectors

        Arguments:
        ----------
        1) n (int): the number of vectors to generate

        Returns:
        --------
        1) ssp_matrix (torch.tensor): a matrix of random hypervectors of
            shape [n, self.dimensionality]
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ssp_matrix = torch.zeros((n, self.dimensionality), device=self.device)

        for i in range(n):
            ssp_matrix[i, :] = make_good_unitary(
                num_dims=self.dimensionality,
                device=self.device,
                rng=rng
            )

        return ssp_matrix


class SA_VSA_OGM(BaseSingleAgentMapper):
    """
    TODO Finish Documentation
    """
    # class variables and objects
    num_observations: int = 0
    ogm: np.ndarray = None
    pairwaise_distance = nn.PairwiseDistance()

    # NOTE: these are annotations only. They must NOT be given mutable
    # defaults here -- build_quadrant_level() appends to them, so a shared
    # class-level list would be mutated by every mapper in the process and
    # every agent would silently read agent 0's geometry from index [0].
    quadrant_axis_bounds: List[Tuple[torch.tensor, torch.tensor]]
    quadrant_centers: List[torch.tensor]
    xy_axis_linspace: Tuple[torch.tensor, torch.tensor]
    occupied_quadrant_memory_vectors: torch.tensor = None
    empty_quadrant_memory_vectors: torch.tensor = None
    xy_axis_vectors: torch.tensor = None
    xy_axis_matrix: torch.tensor = None
    xy_axis_global_heatmap: torch.tensor = None
    xy_axis_occupied_heatmap: torch.tensor = None
    xy_axis_empty_heatmap: torch.tensor = None
    xy_axis_class_matrix: torch.tensor = None


    def __init__(self, config: DictConfig, loggers: List[BaseLogger],
                 print_header: str = ("(SA VSA-OGM")) -> None:
        """
        Initialize the VSA_OGM object.

        Args:
            config (DictConfig): The configuration object containing the
                mapping information
            loggers (List[BaseLogger]): A list of loggers to log information.
            print_header (str): The header to print when logging information

        Returns:
            None
        """
        super(SA_VSA_OGM, self).__init__(config, loggers, print_header)

        # -----------------------------------------------
        # extract parameters from the config
        # -----------------------------------------------
        self.axis_resolution: int = config.mapping.axis_resolution
        self.decoding_method: str = config.mapping.decoding.method
        self.decoding_alpha: float = config.mapping.decoding.alpha
        self.decoding_disk_radii_1: int = config.mapping.decoding.disk_radii_1
        self.decoding_disk_radii_2: int = config.mapping.decoding.disk_radii_2
        self.device: str = config.mapping.device
        self.num_tiles: int = config.mapping.num_tiles
        self.seed: int = config.mapping.seed
        self.vector_dimensionality: int = config.mapping.vector_dimensionality
        self.vector_length_scale: float = config.mapping.vector_length_scale
        self.world_bounds: List[int] = config.data.world_bounds
        self.world_bounds_tensor: torch.tensor = torch.tensor(
            self.world_bounds,
            device=self.device
        )
        self.verbose: bool = config.mapping.verbose

        # -----------------------------------------------
        # initialize class variables based on the config
        # -----------------------------------------------
        
        # normalize the world bounds to start at (0, 0)
        self.world_bounds_norm: List[int] = [
            self.world_bounds[1] - self.world_bounds[0],
            self.world_bounds[3] - self.world_bounds[2]
        ]

        self.environment_dimensionality: int = 2
        self.boolean_results_mask = None

        # --------------------------
        # Dependency Initialization
        # --------------------------
        self.pdist = torch.nn.PairwiseDistance()

        self.ssp_generator = SSPGenerator(
            dimensionality=self.vector_dimensionality,
            device=self.device,
            length_scale=self.vector_length_scale,
            seed=self.seed
        )

        # per-instance geometry containers (see the note on the class body)
        self.quadrant_axis_bounds: List[Tuple[torch.tensor, torch.tensor]] = []
        self.quadrant_centers: List[torch.tensor] = []

        self.build_quadrant_level(0, self.num_tiles)
        self._build_quadrant_indices()
        self._build_xy_axis_linspace()
        self._build_xy_axis_vectors()

        # vector_path = "/home/ssnyde9/axis_vectors.pt"
        # if not os.path.exists(vector_path):
        #     torch.save(self.xy_axis_vectors, vector_path)
        # else:
        #     print("Loading axis vectors from file")
        #     self.xy_axis_vectors = torch.load(vector_path)

        # Memory Caching for Repeated Operations
        self.x_axis_fd = torch.fft.fft(self.xy_axis_vectors[0])[None, :]
        self.y_axis_fd = torch.fft.fft(self.xy_axis_vectors[1])[None, :]

        self._build_xy_axis_matrix()
        self._build_xy_axis_heatmaps()
        self._build_xy_axis_class_matrices()

        self.occupied_quadrant_memory_vectors = torch.zeros(
            size=(
                self.num_tiles ** self.environment_dimensionality,
                self.vector_dimensionality
            ),
            device=self.device
        )
        self.empty_quadrant_memory_vectors = torch.clone(self.occupied_quadrant_memory_vectors)

        self.bounds_X = self.quadrant_axis_bounds[0][0][1]
        self.bounds_Y = self.quadrant_axis_bounds[0][1][1]
        self.num_tiles = int(self.quadrant_centers[0].shape[0] ** (1/2))

        self.xy_axis_matrix = self.xy_axis_matrix.contiguous()

    def fit(self, X: List[np.ndarray], y: List[np.ndarray]) -> None:
        """
        Fit the VSA_OGM model to the given data.

        Args:
            X (List[np.ndarray]): The input data to fit the model to.
            y (List[np.ndarray]): The target data to fit the model to.

        Returns:
            None
        """
        fit_metrics: dict = {}

        if len(X) != len(y):
            raise ValueError("The number of input and target data must match.")
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
            y = torch.tensor(y)
        
        # load the data to the GPUs
        cpu2gpu_start_time: float = time.time()
        X = X.to(self.device)
        cpu2gpu_end_time: float = time.time()
        fit_metrics["cpu2gpu_time"] = cpu2gpu_end_time - cpu2gpu_start_time

        # split the data based on class labels
        split_start_time: float = time.time()
        X_occupied = X[y == 1]
        X_empty = X[y == 0]
        split_end_time: float = time.time()
        fit_metrics["split_time"] = split_end_time - split_start_time

        if len(X_occupied) > 0:
            occ_encoding_metrics: dict = self.encode_observation(X_occupied, occupied=True)
            fit_metrics["occupied"] = occ_encoding_metrics
        if len(X_empty) > 0:
            empty_encoding_metrics: dict = self.encode_observation(X_empty, occupied=False)
            fit_metrics["empty"] = empty_encoding_metrics

        occupied_heatmap = self.xy_axis_occupied_heatmap
        empty_heatmap = self.xy_axis_empty_heatmap

        occupied_heatmap, empty_heatmap, decoding_metrics, intermediate_maps = self.decode_heatmaps(
            occupied_heatmap, empty_heatmap
        )

        fit_metrics["decoding"] = decoding_metrics

        if self.device.startswith("cuda"):
            ogm_conversion_start = torch.cuda.Event(enable_timing=True)
            ogm_conversion_end = torch.cuda.Event(enable_timing=True)
            ogm_conversion_start.record()
        else:
            ogm_conversion_start = time.time()

        ogm = occupied_heatmap - empty_heatmap
        # ogm = ogm.T

        if self.device.startswith("cuda"):
            ogm_conversion_end.record()
            torch.cuda.synchronize()
            fit_metrics["ogm_conversion"] = ogm_conversion_start.elapsed_time(
                ogm_conversion_end
            )
        else:
            ogm_conversion_end = time.time()
            fit_metrics["ogm_conversion"] = ogm_conversion_end - ogm_conversion_start
        
        total_time = sum_nested_dict(fit_metrics)
        if not self.device.startswith("cuda"):
            # python time module returns time in seconds so convert to milliseconds
            total_time *= 1000
        fit_metrics["total_time"] = total_time

        self.ogm = ogm.cpu().numpy()
        self.num_observations += 1

        if self.verbose:
            print(json.dumps(fit_metrics, indent=4))

        return fit_metrics, intermediate_maps
    
    def predict(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict the output from the input data.

        Args:
            X (List[np.ndarray]): The input data to predict the output from.

        Returns:
            List[np.ndarray]: The predicted output data.
        """
        predictions: List[np.ndarray] = []
        prediction_metrics: dict = {}

        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        else:
            X = torch.clone(X)
        
        X = X.to("cpu")

        X[:, 0] -= self.world_bounds[0]
        X[:, 1] -= self.world_bounds[2]
        X = X / self.axis_resolution
        X = torch.round(X)
        X = X.long()

        # filter all points outside of the world bounds
        X = X[(X[:, 0] >= 0) & (X[:, 0] < self.ogm.shape[0]) & (X[:, 1] >= 0) & (X[:, 1] < self.ogm.shape[1])]
        
        predictions: np.ndarray = self.ogm[X[:, 0], X[:, 1]]

        return predictions, prediction_metrics

    def decode_heatmaps(self, occupied_heatmap: torch.tensor,
            empty_heatmap: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        TODO Finish Documentation
        """
        decoding_metrics: dict = {}
        intermediate_maps: dict = {
            "occupied": occupied_heatmap.cpu().numpy(),
            "empty": empty_heatmap.cpu().numpy(),
            "occupied_entropy": None,
            "empty_entropy": None,
            "occupied_entropy_prob": None,
            "empty_entropy_prob": None
        }

        if self.device.startswith("cuda"):
            decoding_start = torch.cuda.Event(enable_timing=True)
            decoding_end = torch.cuda.Event(enable_timing=True)
            decoding_start.record()
        else:
            decoding_start = time.time()

        # -----------------------------------------------
        # Decoding Approach 1: this is the default we have
        #   been using in all tests
        # -----------------------------------------------
        if self.decoding_method == "default":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)
            occ_data = occupied_heatmap.cpu().numpy()
            empty_data = empty_heatmap.cpu().numpy()
            occ_data *= 255
            empty_data *= 255
            occ_data = occ_data.astype(np.uint8)
            empty_data = empty_data.astype(np.uint8)
            occ_data = entropy(occ_data, disk(self.decoding_disk_radii_1))
            empty_data = entropy(empty_data, disk(self.decoding_disk_radii_2))
            occupied_heatmap = torch.tensor(occ_data, device=self.device)
            empty_heatmap = torch.tensor(empty_data, device=self.device)

        # -----------------------------------------------
        # Decoding Approach 2: normalize the individual
        #   heatmaps
        # -----------------------------------------------
        elif self.decoding_method == "normalize":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 3: square the individual
        #   heatmaps
        # -----------------------------------------------
        elif self.decoding_method == "squaring":
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 4: cubic the individual
        #   heatmaps
        # -----------------------------------------------
        elif self.decoding_method == "cubic":
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)

        # -----------------------------------------------
        # Decoding Approach 5: normalize and square
        #   the individual heatmaps
        # -----------------------------------------------
        elif self.decoding_method == "normalize_squaring":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 6: normalize and cubic
        #   the individual heatmaps
        # -----------------------------------------------
        elif self.decoding_method == "normalize_cubic":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)

        # -----------------------------------------------
        # Decoding Approach 7: renyi entropy
        # -----------------------------------------------
        elif self.decoding_method == "renyi":
            occ_data = occupied_heatmap.cpu().numpy()
            empty_data = empty_heatmap.cpu().numpy()
            occ_data *= 255
            empty_data *= 255
            occ_data = occ_data.astype(np.uint8)
            empty_data = empty_data.astype(np.uint8)
            occ_data = entropy(occ_data, disk(self.decoding_disk_radii_1))
            empty_data = entropy(empty_data, disk(self.decoding_disk_radii_2))
            occupied_heatmap = torch.tensor(occ_data, device=self.device)
            empty_heatmap = torch.tensor(empty_data, device=self.device)

        # -----------------------------------------------
        # Decoding Approach 8: renyi entropy with
        #   squaring
        # -----------------------------------------------
        elif self.decoding_method == "renyi_squaring":
            occupied_heatmap = torch.pow(occupied_heatmap, 2)
            empty_heatmap = torch.pow(empty_heatmap, 2)
            occ_data = occupied_heatmap.cpu().numpy()
            empty_data = empty_heatmap.cpu().numpy()
            occ_data *= 255
            empty_data *= 255
            occ_data = occ_data.astype(np.uint8)
            empty_data = empty_data.astype(np.uint8)
            occ_data = entropy(occ_data, disk(self.decoding_disk_radii_1))
            empty_data = entropy(empty_data, disk(self.decoding_disk_radii_2))
            occupied_heatmap = torch.tensor(occ_data, device=self.device)
            empty_heatmap = torch.tensor(empty_data, device=self.device)

        # -----------------------------------------------
        # Decoding Approach 9: renyi entropy with
        #   cubic
        # -----------------------------------------------
        elif self.decoding_method == "renyi_cubic":
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occ_data = occupied_heatmap.cpu().numpy()
            empty_data = empty_heatmap.cpu().numpy()
            occ_data *= 255
            empty_data *= 255
            occ_data = occ_data.astype(np.uint8)
            empty_data = empty_data.astype(np.uint8)
            occ_data = entropy(occ_data, disk(self.decoding_disk_radii_1))
            empty_data = entropy(empty_data, disk(self.decoding_disk_radii_2))
            occupied_heatmap = torch.tensor(occ_data, device=self.device)
            empty_heatmap = torch.tensor(empty_data, device=self.device)

        # -----------------------------------------------
        # Decoding Approach 10: renyi entropy with
        #   normalization
        # -----------------------------------------------
        elif self.decoding_method == "renyi_normalize":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occ_data = occupied_heatmap.cpu().numpy()
            empty_data = empty_heatmap.cpu().numpy()
            occ_data *= 255
            empty_data *= 255
            occ_data = occ_data.astype(np.uint8)
            empty_data = empty_data.astype(np.uint8)
            occ_data = entropy(occ_data, disk(self.decoding_disk_radii_1))
            empty_data = entropy(empty_data, disk(self.decoding_disk_radii_2))
            occupied_heatmap = torch.tensor(occ_data, device=self.device)
            empty_heatmap = torch.tensor(empty_data, device=self.device)

        # -----------------------------------------------
        # Decoding Approach 11: renyi entropy with
        #   normalization and squaring
        # -----------------------------------------------
        elif self.decoding_method == "renyi_normalize_squaring":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.pow(occupied_heatmap, 2)
            empty_heatmap = torch.pow(empty_heatmap, 2)
            occupied_heatmap = compute_local_entropy(occupied_heatmap, self.decoding_disk_radii_1)
            empty_heatmap = compute_local_entropy(empty_heatmap, self.decoding_disk_radii_2)

        # -----------------------------------------------
        # Decoding Approach 12: renyi entropy with
        #   normalization and cubic
        # -----------------------------------------------
        elif self.decoding_method == "renyi_normalize_cubic":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occupied_heatmap = compute_local_entropy(occupied_heatmap, self.decoding_disk_radii_1)
            empty_heatmap = compute_local_entropy(empty_heatmap, self.decoding_disk_radii_2)
            # occ_data = occupied_heatmap.cpu().numpy()
            # empty_data = empty_heatmap.cpu().numpy()
            # occ_data *= 255
            # empty_data *= 255
            # occ_data = occ_data.astype(np.uint8)
            # empty_data = empty_data.astype(np.uint8)
            # occ_data = entropy(occ_data, disk(self.decoding_disk_radii_1))
            # empty_data = entropy(empty_data, disk(self.decoding_disk_radii_2))

            # save to intermediate representations
            # intermediate_maps["occupied_entropy"] = occ_data
            # intermediate_maps["empty_entropy"] = empty_data

            # occupied_heatmap = torch.tensor(occ_data, device=self.device)
            # empty_heatmap = torch.tensor(empty_data, device=self.device)

        # -----------------------------------------------
        # Decoding Approach 13: ReLU
        # -----------------------------------------------
        elif self.decoding_method == "relu":
            occupied_heatmap = torch.relu(occupied_heatmap)
            empty_heatmap = torch.relu(empty_heatmap)
        
        # -----------------------------------------------
        # Decoding Approach 14: ReLU with squaring
        # -----------------------------------------------
        elif self.decoding_method == "relu_squaring":
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)
            occupied_heatmap = torch.relu(occupied_heatmap)
            empty_heatmap = torch.relu(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 15: ReLU with cubic
        # -----------------------------------------------
        elif self.decoding_method == "relu_cubic":
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occupied_heatmap = torch.relu(occupied_heatmap)
            empty_heatmap = torch.relu(empty_heatmap)
        
        # -----------------------------------------------
        # Decoding Approach 16: ReLU with normalization
        # -----------------------------------------------
        elif self.decoding_method == "relu_normalize":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.relu(occupied_heatmap)
            empty_heatmap = torch.relu(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 17: ReLU with normalization
        #   and squaring
        # -----------------------------------------------
        elif self.decoding_method == "relu_normalize_squaring":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)
            occupied_heatmap = torch.relu(occupied_heatmap)
            empty_heatmap = torch.relu(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 18: ReLU with normalization
        #   and cubic
        # -----------------------------------------------
        elif self.decoding_method == "relu_normalize_cubic":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occupied_heatmap = torch.relu(occupied_heatmap)
            empty_heatmap = torch.relu(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 19: TanH
        # -----------------------------------------------
        elif self.decoding_method == "tanh":
            occupied_heatmap = torch.tanh(occupied_heatmap)
            empty_heatmap = torch.tanh(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 20: TanH with squaring
        # -----------------------------------------------
        elif self.decoding_method == "tanh_squaring":
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)
            occupied_heatmap = torch.tanh(occupied_heatmap)
            empty_heatmap = torch.tanh(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 21: TanH with cubic
        # -----------------------------------------------
        elif self.decoding_method == "tanh_cubic":
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occupied_heatmap = torch.tanh(occupied_heatmap)
            empty_heatmap = torch.tanh(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 22: TanH with normalization
        # -----------------------------------------------
        elif self.decoding_method == "tanh_normalize":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.tanh(occupied_heatmap)
            empty_heatmap = torch.tanh(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 23: TanH with normalization
        #   and squaring
        # -----------------------------------------------
        elif self.decoding_method == "tanh_normalize_squaring":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)
            occupied_heatmap = torch.tanh(occupied_heatmap)
            empty_heatmap = torch.tanh(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 24: TanH with normalization
        #   and cubic
        # -----------------------------------------------
        elif self.decoding_method == "tanh_normalize_cubic":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occupied_heatmap = torch.tanh(occupied_heatmap)
            empty_heatmap = torch.tanh(empty_heatmap)
        
        # -----------------------------------------------
        # Decoding Approach 25: Sigmoid
        # -----------------------------------------------
        elif self.decoding_method == "sigmoid":
            occupied_heatmap = torch.sigmoid(occupied_heatmap)
            empty_heatmap = torch.sigmoid(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 26: Sigmoid with squaring
        # -----------------------------------------------
        elif self.decoding_method == "sigmoid_squaring":
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)
            occupied_heatmap = torch.sigmoid(occupied_heatmap)
            empty_heatmap = torch.sigmoid(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 27: Sigmoid with cubic
        # -----------------------------------------------
        elif self.decoding_method == "sigmoid_cubic":
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occupied_heatmap = torch.sigmoid(occupied_heatmap)
            empty_heatmap = torch.sigmoid(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 28: Sigmoid with normalization
        # -----------------------------------------------
        elif self.decoding_method == "sigmoid_normalize":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.sigmoid(occupied_heatmap)
            empty_heatmap = torch.sigmoid(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 29: Sigmoid with normalization
        #   and squaring
        # -----------------------------------------------
        elif self.decoding_method == "sigmoid_normalize_squaring":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.square(occupied_heatmap)
            empty_heatmap = torch.square(empty_heatmap)
            occupied_heatmap = torch.sigmoid(occupied_heatmap)
            empty_heatmap = torch.sigmoid(empty_heatmap)

        # -----------------------------------------------
        # Decoding Approach 30: Sigmoid with normalization
        #   and cubic
        # -----------------------------------------------
        elif self.decoding_method == "sigmoid_normalize_cubic":
            occupied_heatmap /= torch.max(occupied_heatmap)
            empty_heatmap /= torch.max(empty_heatmap)
            occupied_heatmap = torch.pow(occupied_heatmap, 3)
            empty_heatmap = torch.pow(empty_heatmap, 3)
            occupied_heatmap = torch.sigmoid(occupied_heatmap)
            empty_heatmap = torch.sigmoid(empty_heatmap)

        # -----------------------------------------------
        # Unknown Decoding Method
        # -----------------------------------------------
        else:
            raise ValueError(f"Unknown decoding method: {self.decoding_method}")
        
        # intermediate_maps["occupied_entropy"] = torch.clone(occupied_heatmap)
        # intermediate_maps["empty_entropy"] = torch.clone(empty_heatmap)

        stacked_heatmap = torch.stack((occupied_heatmap, empty_heatmap), dim=0)
        stacked_heatmap = torch.softmax(stacked_heatmap, dim=0)
        occupied_heatmap_softmax = stacked_heatmap[0]
        empty_heatmap_softmax = stacked_heatmap[1]
        
        if self.device.startswith("cuda"):
            decoding_end.record()
            torch.cuda.synchronize()
            decoding_metrics["decoding_time"] = decoding_start.elapsed_time(
                decoding_end
            )
        else:
            decoding_end = time.time()
            decoding_metrics["decoding_time"] = decoding_end - decoding_start

        intermediate_maps["occupied_entropy_prob"] = occupied_heatmap_softmax.cpu().numpy()
        intermediate_maps["empty_entropy_prob"] = empty_heatmap_softmax.cpu().numpy()
        intermediate_maps["ogm_testing"] = (occupied_heatmap_softmax - empty_heatmap_softmax).cpu().numpy()
        
        # intermediate_maps["occupied_entropy"] = occupied_heatmap.cpu().numpy()
        # intermediate_maps["empty_entropy"] = empty_heatmap.cpu().numpy()
        
        return occupied_heatmap_softmax, empty_heatmap_softmax, decoding_metrics, intermediate_maps
    
    def encode_observation(self, point_cloud: Union[np.ndarray, torch.tensor],
            occupied: bool = True) -> None:
        """
        Processes an observation represented as a point cloud and the
        corresponding labels for each point.

        Args:
            point_cloud (Union[np.ndarray, torch.tensor]): A 2D tensor of points
            labels (Union[np.ndarray, torch.tensor]): A 1D tensor of labels

        Returns:
            None
        """

        encode_metrics: dict = {}

        # Timing (Start)
        if self.device.startswith("cuda"):
            world_bound_norm_start = torch.cuda.Event(enable_timing=True)
            world_bound_norm_end = torch.cuda.Event(enable_timing=True)
            world_bound_norm_start.record()
        else:
            world_bound_norm_start = time.time()

        # Computations
        point_cloud[:, :2] -= self.world_bounds_tensor[[0, 2]]
        
        # Timing (End)
        if self.device.startswith("cuda"):
            world_bound_norm_end.record()
        else:
            world_bound_norm_end = time.time()
            encode_metrics["world_bound_norm"] = (world_bound_norm_end \
                - world_bound_norm_start)

        ups = point_cloud

        # -----------------------------------------------
        # Calculate quadrant memories for each new point
        # using a multipoint L2 distance calculation
        # -----------------------------------------------
        # Timing (Start)
        if self.device.startswith("cuda"):
            tile_memory_calculation_start = torch.cuda.Event(enable_timing=True)
            tile_memory_calculation_end = torch.cuda.Event(enable_timing=True)
            tile_memory_calculation_start.record()
        else:
            tile_memory_calculation_start = time.time()

        # Computation
        ups: torch.tensor = ups.unsqueeze(1)
        qcm: torch.tensor = self.quadrant_centers[0]
        qcm: torch.tensor = qcm.unsqueeze(0)
        dists: torch.tensor = self.pdist(ups, qcm)
        closest_quads: torch.tensor = torch.argmin(dists, dim=1)
        ups = ups.squeeze(1)

        # Timing (End)
        if self.device.startswith("cuda"):
            tile_memory_calculation_end.record()
        else:
            tile_memory_calculation_end = time.time()
            encode_metrics["tile_memory_calculation"] = tile_memory_calculation_end \
                - tile_memory_calculation_start

        # -----------------------------------------------
        # normalize all points to the dimensions of a single quadrant
        # -----------------------------------------------
        # Timing (Start)
        if self.device.startswith("cuda"):
            quadrant_norm_start = torch.cuda.Event(enable_timing=True)
            quadrant_norm_end = torch.cuda.Event(enable_timing=True)
            quadrant_norm_start.record()
        else:
            quadrant_norm_start = time.time()

        # Computation
        ups[:, 0].remainder_(self.bounds_X)
        ups[:, 1].remainder_(self.bounds_Y)

        # Timing (End)
        if self.device.startswith("cuda"):
            quadrant_norm_end.record()
        else:
            quadrant_norm_end = time.time()
            encode_metrics["quadrant_norm"] = quadrant_norm_end - quadrant_norm_start

        # ---------------------------
        # Matrix Encoding Approach
        # ---------------------------
        # Timing (Start)
        if self.device.startswith("cuda"):
            powers_start = torch.cuda.Event(enable_timing=True)
            powers_end = torch.cuda.Event(enable_timing=True)
            powers_start.record()
        else:
            powers_start = time.time()

        # Computation
        ups = ups / self.vector_length_scale
        x_powers = ups[:, 0]
        y_powers = ups[:, 1]

        # Timing (End)
        if self.device.startswith("cuda"):
            powers_end.record()
        else:
            powers_end = time.time()
            encode_metrics["powers"] = powers_end - powers_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            axis_fd_power_matrix_start = torch.cuda.Event(enable_timing=True)
            axis_fd_power_matrix_end = torch.cuda.Event(enable_timing=True)
            axis_fd_power_matrix_start.record()
        else:
            axis_fd_power_matrix_start = time.time()

        # Computation
        # x_axis_fd_matrix = self.x_axis_fd ** x_powers[:, None]
        # y_axis_fd_matrix = self.y_axis_fd ** y_powers[:, None]

        x_axis_fd_matrix = torch.exp(x_powers[:, None] * torch.log(self.x_axis_fd))
        y_axis_fd_matrix = torch.exp(y_powers[:, None] * torch.log(self.y_axis_fd))


        # Timing (End)
        if self.device.startswith("cuda"):
            axis_fd_power_matrix_end.record()
        else:
            axis_fd_power_matrix_end = time.time()
            encode_metrics["axis_fd_power_matrix"] = axis_fd_power_matrix_end \
                - axis_fd_power_matrix_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            axis_fd_power_unsqueeze_start = torch.cuda.Event(enable_timing=True)
            axis_fd_power_unsqueeze_end = torch.cuda.Event(enable_timing=True)
            axis_fd_power_unsqueeze_start.record()
        else:
            axis_fd_power_unsqueeze_start = time.time()

        # Computation
        x_axis_fd_matrix = x_axis_fd_matrix.unsqueeze(0)
        y_axis_fd_matrix = y_axis_fd_matrix.unsqueeze(0)

        # Timing (End)
        if self.device.startswith("cuda"):
            axis_fd_power_unsqueeze_end.record()
        else:
            axis_fd_power_unsqueeze_end = time.time()
            encode_metrics["axis_fd_power_unsqueeze"] = axis_fd_power_unsqueeze_end \
                - axis_fd_power_unsqueeze_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            xy_axis_fd_matrix_start = torch.cuda.Event(enable_timing=True)
            xy_axis_fd_matrix_end = torch.cuda.Event(enable_timing=True)
            xy_axis_fd_matrix_start.record()
        else:
            xy_axis_fd_matrix_start = time.time()
        
        # Computation
        xy_axis_fd_matrix = torch.concatenate((x_axis_fd_matrix, y_axis_fd_matrix), dim=0)
        xy_axis_fd_matrix = torch.prod(xy_axis_fd_matrix, dim=0)

        # Timing (End)
        if self.device.startswith("cuda"):
            xy_axis_fd_matrix_end.record()
        else:
            xy_axis_fd_matrix_end = time.time()
            encode_metrics["xy_axis_fd_matrix"] = xy_axis_fd_matrix_end - xy_axis_fd_matrix_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            xy_axis_ifft_start = torch.cuda.Event(enable_timing=True)
            xy_axis_ifft_end = torch.cuda.Event(enable_timing=True)
            xy_axis_ifft_start.record()
        else:
            xy_axis_ifft_start = time.time()

        # Computation
        xy_axis_fd_matrix = torch.fft.ifft(xy_axis_fd_matrix, dim=1)
        xy_axis_fd_matrix = xy_axis_fd_matrix.real

        # Timing (End)
        if self.device.startswith("cuda"):
            xy_axis_ifft_end.record()
        else:
            xy_axis_ifft_end = time.time()
            encode_metrics["xy_axis_ifft"] = xy_axis_ifft_end - xy_axis_ifft_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            index_add_start = torch.cuda.Event(enable_timing=True)
            index_add_end = torch.cuda.Event(enable_timing=True)
            index_add_start.record()
        else:
            index_add_start = time.time()

        # Computation
        if occupied:
            self.occupied_quadrant_memory_vectors.index_add_(
                0,
                closest_quads,
                xy_axis_fd_matrix.float()
            )
        else:
            self.empty_quadrant_memory_vectors.index_add_(
                0,
                closest_quads,
                xy_axis_fd_matrix.float()
            )

        # Timing (End)
        if self.device.startswith("cuda"):
            index_add_end.record()
        else:
            index_add_end = time.time()
            encode_metrics["index_add"] = index_add_end - index_add_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            qv_norm_start = torch.cuda.Event(enable_timing=True)
            qv_norm_end = torch.cuda.Event(enable_timing=True)
            qv_norm_start.record()
        else:
            qv_norm_start = time.time()

        updated_indices = torch.unique(closest_quads)

        if occupied:
            norm_qv = self.occupied_quadrant_memory_vectors[updated_indices] / torch.norm(
                self.occupied_quadrant_memory_vectors[updated_indices], dim=1, keepdim=True
            )
        else:
            norm_qv = self.empty_quadrant_memory_vectors[updated_indices] / torch.norm(
                self.empty_quadrant_memory_vectors[updated_indices], dim=1, keepdim=True
            )

        # Timing (End)
        if self.device.startswith("cuda"):
            qv_norm_end.record()
        else:
            qv_norm_end = time.time()
            encode_metrics["qv_norm"] = qv_norm_end - qv_norm_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            hm_clone_start = torch.cuda.Event(enable_timing=True)
            hm_clone_end = torch.cuda.Event(enable_timing=True)
            hm_clone_start.record()
        else:
            hm_clone_start = time.time()

        if occupied:
            temp_xy_axis_heatmap = torch.clone(self.xy_axis_occupied_heatmap)
        else:
            temp_xy_axis_heatmap = torch.clone(self.xy_axis_empty_heatmap)

        # Timing (End)
        if self.device.startswith("cuda"):
            hm_clone_end.record()
        else:
            hm_clone_end = time.time()
            encode_metrics["hm_clone"] = hm_clone_end - hm_clone_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            dot_product_start = torch.cuda.Event(enable_timing=True)
            dot_product_end = torch.cuda.Event(enable_timing=True)
            dot_product_start.record()
        else:
            dot_product_start = time.time()

        if not hasattr(self, "occupied_results"):
            self.occupied_results = torch.zeros(
                (self.occupied_quadrant_memory_vectors.shape[0],) + self.xy_axis_matrix.shape[:2], 
                device=self.device
            )
            self.empty_results = torch.zeros(
                (self.empty_quadrant_memory_vectors.shape[0],) + self.xy_axis_matrix.shape[:2], 
                device=self.device
            )

        if occupied:
            result = self.occupied_results
        else:
            result = self.empty_results

        partial_result = torch.einsum('nm,xym->nxy', norm_qv, self.xy_axis_matrix)
        # partial_result = compute_mm(norm_qv, self.xy_axis_matrix)
        result[updated_indices] = partial_result

        if occupied:
            self.occupied_results = result
        else:
            self.empty_results = result

        # Timing (End)
        if self.device.startswith("cuda"):
            dot_product_end.record()
        else:
            dot_product_end = time.time()
            encode_metrics["dot_product"] = dot_product_end - dot_product_start

        # Timing (Start)
        if self.device.startswith("cuda"):
            hm_decoding_start = torch.cuda.Event(enable_timing=True)
            hm_decoding_end = torch.cuda.Event(enable_timing=True)
            hm_decoding_start.record()
        else:
            hm_decoding_start = time.time()

        if self.num_tiles > 1:
            result = result.view(self.num_tiles, self.num_tiles, self.quadrant_indices_y[1], self.quadrant_indices_x[1])
            result = result.permute(1, 2, 0, 3)
            result = result.reshape(self.num_tiles * self.quadrant_indices_y[1], self.num_tiles * self.quadrant_indices_x[1])
        else:
            result = result.squeeze(0)

        temp_xy_axis_heatmap = result

        temp_xy_axis_heatmap = torch.nan_to_num(temp_xy_axis_heatmap)
        if occupied:
            self.xy_axis_occupied_heatmap = temp_xy_axis_heatmap
        else:
            self.xy_axis_empty_heatmap = temp_xy_axis_heatmap 

        # Timing (End)
        if self.device.startswith("cuda"):
            hm_decoding_end.record()
        else:
            hm_decoding_end = time.time()
            encode_metrics["hm_decoding"] = hm_decoding_end - hm_decoding_start

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

            encode_metrics["quadrant_norm"] = quadrant_norm_start.elapsed_time(
                quadrant_norm_end
            )

            encode_metrics["world_bound_norm"] = world_bound_norm_start.elapsed_time(
                world_bound_norm_end
            )
            encode_metrics["tile_memory_calculation"] = tile_memory_calculation_start.elapsed_time(
                tile_memory_calculation_end
            )
            encode_metrics["axis_fd_power_matrix"] = axis_fd_power_matrix_start.elapsed_time(
                axis_fd_power_matrix_end
            )
            encode_metrics["axis_fd_power_unsqueeze"] = axis_fd_power_unsqueeze_start.elapsed_time(
                axis_fd_power_unsqueeze_end
            )
            encode_metrics["xy_axis_fd_matrix"] = xy_axis_fd_matrix_start.elapsed_time(
                xy_axis_fd_matrix_end
            )
            encode_metrics["xy_axis_ifft"] = xy_axis_ifft_start.elapsed_time(
                xy_axis_ifft_end
            )
            encode_metrics["index_add"] = index_add_start.elapsed_time(
                index_add_end
            )
            encode_metrics["qv_norm"] = qv_norm_start.elapsed_time(
                qv_norm_end
            )
            encode_metrics["hm_clone"] = hm_clone_start.elapsed_time(
                hm_clone_end
            )
            encode_metrics["dot_product"] = dot_product_start.elapsed_time(
                dot_product_end
            )
            encode_metrics["hm_decoding"] = hm_decoding_start.elapsed_time(
                hm_decoding_end
            )

        return encode_metrics


    def query_point_thetas(self, points: Union[np.ndarray, torch.tensor],
                return_as_numpy: bool = True) -> torch.tensor:
        """
        Queries the memory for the given point and returns the theta value.

        Args:
            - points: A numpy array or torch tensor representing the points to
                query.
            - return_as_numpy: A boolean indicating whether to return the
                results as a numpy array (default: True).

        Returns:
            - results: A torch tensor or numpy array containing the theta
                values for the queried points.
        """
        # assert isinstance(points, np.ndarray) or isinstance(points, torch.Tensor)
        # assert len(points.shape) == 2
        # assert points.shape[1] == self.environment_dimensionality
        # assert isinstance(return_as_numpy, bool)

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        
        if points.device != self.device:
            points = points.to(self.device)

        # assert torch.min(points[:, 0]) >= self.world_bounds[0]
        # assert torch.max(points[:, 0]) <= self.world_bounds[1]
        # assert torch.min(points[:, 1]) >= self.world_bounds[2]
        # assert torch.max(points[:, 1]) <= self.world_bounds[3]

        points[:, 0] -= self.world_bounds[0]
        points[:, 1] -= self.world_bounds[2]
        points = points / self.axis_resolution
        points = torch.round(points)
        points = points.long()
        
        results: torch.tensor = self.xy_axis_global_heatmap[points[:, 0], points[:, 1]]

        # assert len(results.shape) == 1
        # assert results.shape[0] == points.shape[0]

        if return_as_numpy:
            results = results.detach().cpu().numpy()
        
        return results
            
    def query_point_classes(self, points: Union[np.ndarray, torch.tensor],
                return_as_numpy: bool = True) -> torch.tensor:
        """
        Queries the memory for the given point and returns the class.

        Args:
            point (torch.tensor): The point to query.

        Returns:
            torch.tensor: The class for the given point.
        """
        # assert isinstance(points, np.ndarray) or isinstance(points, torch.Tensor)
        # assert len(points.shape) == 2
        # assert points.shape[1] == self.environment_dimensionality
        # assert isinstance(return_as_numpy, bool)

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        
        if points.device != self.device:
            points = points.to(self.device)

        # assert torch.min(points[:, 0]) >= self.world_bounds[0]
        # assert torch.max(points[:, 0]) <= self.world_bounds[1]
        # assert torch.min(points[:, 1]) >= self.world_bounds[2]
        # assert torch.max(points[:, 1]) <= self.world_bounds[3]

        points[:, 0] -= self.world_bounds[0]
        points[:, 1] -= self.world_bounds[2]
        points = points / self.axis_resolution
        points = torch.round(points)
        points = points.long()
        
        results: torch.tensor = self.xy_axis_class_matrix[points[:, 0], points[:, 1]]

        # assert len(results.shape) == 1
        # assert results.shape[0] == points.shape[0]

        if return_as_numpy:
            results = results.detach().cpu().numpy()
        
        return results
    
    def _build_xy_axis_class_matrices(self) -> None:
        """
        Builds the XY axis class matrices.

        This method asserts that the `xy_axis_matrix` attribute is not None, is
        of type `torch.tensor`, and has a shape with three dimensions. It then
        initializes the `xy_axis_heatmap` attribute as a tensor of zeros with
        the same shape as `xy_axis_matrix`.

        Args:
            None
            
        Returns:
            None
        
        Raises:
            AssertionError: If xy_axis_matrix is None, not a torch.tensor,
            or has an invalid shape.
        """
        # assert self.xy_axis_matrix is not None
        # assert isinstance(self.xy_axis_matrix, torch.Tensor)
        # assert len(self.xy_axis_matrix.shape) == 3

        if self.verbose:
            print("Building XY axis class matrices...")

        self.xy_axis_class_matrix = torch.ones(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )
        self.xy_axis_class_matrix *= -2

        if self.verbose:
            print("Finished building XY axis class matrices.")

    def _build_xy_axis_heatmaps(self) -> None:
        """
        Builds the XY axis class heatmaps based on the xy_axis_matrix.

        Args:
            None
        
        Returns:
            None
        
        Raises:
            AssertionError: If xy_axis_matrix is None, not a torch.tensor,
            or has an invalid shape.
        """
        # assert self.xy_axis_matrix is not None
        # assert isinstance(self.xy_axis_matrix, torch.Tensor)
        # assert len(self.xy_axis_matrix.shape) == 3

        if self.verbose:
            print("Building XY axis heatmaps...")

        self.xy_axis_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )
    
        self.xy_axis_occupied_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )

        self.xy_axis_empty_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )

        if self.verbose:
            print("Finished building XY axis heatmaps.")

    def _build_xy_axis_linspace(self) -> None:
        """
        Build the x and y axis linspace for the XY axis.

        This method calculates the x and y axis linspace based on the world
        bounds and axis resolution. It also extracts the horizontal and
        vertical boundaries, as well as the centers, from the axis linspace.
        Finally, it plots the quadrant boundaries, quadrant centers, and voxels
        for the XY axis.

        Args:
            None

        Returns:
            None
            
        Raises:
            NotImplementedError: If the environment dimensionality != 2.
        """

        if self.verbose:
            print("Building XY axis linspace...")

        # if self.environment_dimensionality != 2:
        #     raise NotImplementedError

        # assert self.world_bounds_norm[0] / self.axis_resolution == \
        #     int(self.world_bounds_norm[0] / self.axis_resolution)
        # assert self.world_bounds_norm[1] / self.axis_resolution == \
        #     int(self.world_bounds_norm[1] / self.axis_resolution)
        
        xal_steps: int = int(self.world_bounds_norm[0] / self.axis_resolution)
        yal_steps: int = int(self.world_bounds_norm[1] / self.axis_resolution)
        
        xa = torch.linspace(
            start=0,
            end=self.world_bounds_norm[0],
            steps=(2 * xal_steps + 1),
            device=self.device
        )
        ya = torch.linspace(
            start=0,
            end=self.world_bounds_norm[1],
            steps=(2 * yal_steps + 1),
            device=self.device
        )

        # extract the horizontal and vertical boundaries from the axis linspace
        xab = xa[::2]
        yab = ya[::2]

        # assert len(xab.shape) == 1
        # assert len(yab.shape) == 1
        # assert torch.min(xab) == 0
        # assert torch.min(yab) == 0
        # assert torch.max(xab) == self.world_bounds_norm[0]
        # assert torch.max(yab) == self.world_bounds_norm[1]
    
        # extract the centers from the axis linspace
        xac = xa[1::2]
        yac = ya[1::2]

        # assert len(xac.shape) == 1
        # assert len(yac.shape) == 1
        # assert torch.min(xac) == self.axis_resolution / 2
        # assert torch.min(yac) == self.axis_resolution / 2
        # assert torch.max(xac) == self.world_bounds_norm[0] - self.axis_resolution / 2
        # assert torch.max(yac) == self.world_bounds_norm[1] - self.axis_resolution / 2

        self.xy_axis_linspace = (xac, yac)

        if self.verbose:
            print("Finished building XY axis linspace.")

        # if self.plotting_flags["plot_xy_voxels"]:
        #     if self.verbose:
        #         print("Plotting XY axis boundaries, centers, and voxels...")

        #     vbp_sp: str = os.path.join(
        #         self.log_dir,
        #         "xy_voxel_boundaries.png"
        #     )
        #     spp.plot_quadrant_boundaries(
        #         qb_x=xab,
        #         qb_y=yab,
        #         world_bounds_norm=self.world_bounds_norm,
        #         save_path=vbp_sp,
        #         title_header="XY Axis Voxel Boundaries",
        #     )

        #     vcmg = torch.meshgrid(xac, yac, indexing="xy")
        #     vcmg = torch.stack(vcmg, dim=2)
        #     vcmg = vcmg.reshape((vcmg.shape[0] * vcmg.shape[1], 2))
        #     vcmg = vcmg.to(self.device)

        #     vcp_sp: str = os.path.join(
        #         self.log_dir,
        #         "xy_voxel_centers.png"
        #     )
        #     spp.plot_quadrant_centers(
        #         qcs=vcmg,
        #         world_bounds_norm=self.world_bounds_norm,
        #         save_path=vcp_sp,
        #         title_header="XY Axis Voxel Centers",
        #     )

        #     vp_sp: str = os.path.join(
        #         self.log_dir,
        #         "xy_voxels.png"
        #     )
        #     spp.plot_quadrants_and_centers(
        #         qcs=vcmg,
        #         qb_x=xab,
        #         qb_y=yab,
        #         world_bounds_norm=self.world_bounds_norm,
        #         save_path=vp_sp,
        #         title_header="XY Axis Voxels",
        #     )

        #     if self.verbose:
        #         print("Finished plotting XY axis boundaries, centers, and voxels.")


    def _build_xy_axis_vectors(self) -> None:
        """
        Builds the XY axis vectors using the SSP generator.

        This method generates the XY axis vectors based on the environment
        dimensionality using the SSP generator. It ensures that the generated
        vectors have the correct shape and dimensions.

        Args:
            None
        
        Returns:
            None

        Raises:
            AssertionError: If the generated vectors have an incorrect shape
            or dimension.
        """

        if self.verbose:
            print("Building XY axis vectors...")

        self.xy_axis_vectors = self.ssp_generator.generate(
            self.environment_dimensionality
        )

        # axis_vector_sp: str = os.path.join(
        #     self.log_dir,
        #     "xy_axis_vectors.npy"
        # )
        # with open(axis_vector_sp, "wb") as f:
        #     np.save(f, self.xy_axis_vectors.detach().cpu().numpy())

        assert len(self.xy_axis_vectors.shape) == 2
        assert self.xy_axis_vectors.shape[0] == self.environment_dimensionality
        assert self.xy_axis_vectors.shape[1] == self.vector_dimensionality

        if self.verbose:
            print("Finished building XY axis vectors.")
    
    def _build_xy_axis_matrix(self) -> None:
        """
        Build the XY axis matrix using the xy_axis_linspace and xy_axis_vectors.

        This method constructs a matrix representing the XY axis by iterating
        over the xy_axis_linspace and xy_axis_vectors. For each combination of
        x and y values, it calculates the corresponding vector using the power
        function and binds them together using fractional binding.

        Returns:
            None
        """

        if self.verbose:
            print("Building XY axis matrix...")

        x_shape: tuple = self.quadrant_indices_x[1]
        y_shape: tuple = self.quadrant_indices_y[1]

        self.xy_axis_matrix = torch.zeros(
            (x_shape, y_shape, self.vector_dimensionality),
            device=self.device
        )

        x_axis_fd_matrix = self.x_axis_fd.repeat(x_shape, 1)
        y_axis_fd_matrix = self.y_axis_fd.repeat(y_shape, 1)

        x_powers = (self.xy_axis_linspace[0][:x_shape] / self.vector_length_scale)
        y_powers = (self.xy_axis_linspace[1][:y_shape] / self.vector_length_scale)

        x_power_matrix = x_powers.repeat(self.vector_dimensionality, 1).T
        y_power_matrix = y_powers.repeat(self.vector_dimensionality, 1).T

        x_axis_fd_matrix = x_axis_fd_matrix ** x_power_matrix
        y_axis_fd_matrix = y_axis_fd_matrix ** y_power_matrix

        x_axis_fd_matrix = x_axis_fd_matrix.unsqueeze(1)
        y_axis_fd_matrix = y_axis_fd_matrix.unsqueeze(0)

        self.xy_axis_matrix = x_axis_fd_matrix * y_axis_fd_matrix
        self.xy_axis_matrix = torch.fft.ifftn(self.xy_axis_matrix, dim=-1).real

        if self.verbose:
            print("Finished building XY axis matrix.")

    def _build_quadrant_indices(self) -> None:
        """
        Builds the quadrant indices based on the quadrant axis bounds and axis
        resolution. The quadrant indices are calculated by dividing the
        quadrant axis bounds by the axis resolution.
        """

        if self.verbose:
            print("Building quadrant indices...")

        quadrant_indices_x = self.quadrant_axis_bounds[0][0] / self.axis_resolution
        quadrant_indices_y = self.quadrant_axis_bounds[0][1] / self.axis_resolution

        self.quadrant_indices_x = quadrant_indices_x.to(torch.int)
        self.quadrant_indices_y = quadrant_indices_y.to(torch.int)

        if self.verbose:
            print("Finished building quadrant indices.")

    def _build_quadrant_memory_hierarchy(self) -> None:
        """
        Builds the memory hierarchy for the quadrants.

        This method constructs the memory hierarchy for the quadrants based on
        the specified VSA dimensionality. It initializes the quadrant memory
        vectors as torch tensors with zeros.

        Args:
            None
        
        Returns:
            None

        Raises:
            NotImplementedError: If the hierarchy has more than one level.
        """

        if self.verbose:
            print("Building quadrant memory hierarchy...")

        # if len(self.quadrant_hierarchy) > 1:
        #     raise NotImplementedError

        # Shape = {
        #   0 = number of quadrants by number of quadrants flattened,
        #   1 = number of dimensions in the vsa
        # }
        self.occupied_quadrant_memory_vectors = torch.zeros(
            size=(
                self.num_tiles ** self.environment_dimensionality,
                self.vector_dimensionality
            ),
            device=self.device
        )
        self.empty_quadrant_memory_vectors = torch.clone(self.occupied_quadrant_memory_vectors)

        if self.verbose:
            print("Finished building quadrant memory hierarchy.")

    def build_quadrant_level(self, level: int, size: int) -> None:
        """
        Build the quadrant level based on the given level and size.

        Args:
            level (int): The level of the quadrant.

        Returns:
            None

        Raises:
            NotImplementedError: If the level is not 0.
        """

        if self.verbose:
            print(f"Building quadrant level [{level}]...")
            print("World Bounds Norm: ", self.world_bounds_norm)
            print("World Bounds Norm Quadrant X: ", self.world_bounds_norm[0] / size)
            print("World Bounds Norm Quadrant X: ", self.world_bounds_norm[1] / size)
            print("World Bounds Norm Quadrant X: ", self.world_bounds_norm[0] / self.axis_resolution / size)
            print("World Bounds Norm Quadrant X: ", self.world_bounds_norm[1] / self.axis_resolution / size)

        new_upper_bounds = [
            size * np.ceil(self.world_bounds_norm[0] / size),
            size * np.ceil(self.world_bounds_norm[1] / size)
        ]

        if self.verbose:
            print("Num Quadrants = ", size * np.ceil(self.world_bounds_norm[0] / size))
            print("Num Quadrants = ", size * np.ceil(self.world_bounds_norm[1] / size))

            print("New Bounds Norm: ", new_upper_bounds)
            print("New Bounds Norm Quadrant X: ", new_upper_bounds[0] / size)
            print("New Bounds Norm Quadrant X: ", new_upper_bounds[1] / size)
            print("New Bounds Norm Quadrant X: ", new_upper_bounds[0] / self.axis_resolution / size)
            print("New Bounds Norm Quadrant X: ", new_upper_bounds[1] / self.axis_resolution / size)
        
        # Calculate the quadrant size across x and y axes in meters
        # while also taking into account the level of the quadrant
        if level == 0:
            size_x_meters: float = new_upper_bounds[0] / size
            size_y_meters: float = new_upper_bounds[1] / size
        
        size_x_bins = int(size_x_meters / self.axis_resolution)
        size_y_bins = int(size_y_meters / self.axis_resolution)

        if self.verbose:
            print(f"Quadrant Level [{level}] - Number of Quadrants: {size ** 2}")
            print(f"Quadrant Level [{level}] - Quadrant Size (Bins): {size_x_bins}, {size_y_bins}")
            print(f"Quadrant Level [{level}] - Quadrant Size (Meters): {size_x_meters}, {size_y_meters}")
            print(f"Quadrant Level [{level}] - Level Size (Bins): {size_x_bins * size}, {size_y_bins * size}")
            print(f"Quadrant Level [{level}] - Level Size (Meters): {size_x_meters * size}, {size_y_meters * size}")

        # Verify that the level size calculated based on all quadrant sizes
        if level == 0:
            assert size_x_meters * size == new_upper_bounds[0]
            assert size_y_meters * size == new_upper_bounds[1]
        else:
            raise NotImplementedError

        # calculate the vertical and horizontal bounds of the quadrants
        qb_x = torch.linspace(0, new_upper_bounds[0], size + 1)
        qb_y = torch.linspace(0, new_upper_bounds[1], size + 1)
            
        # calculate the quadrant centers
        qcs_x = torch.linspace(0, new_upper_bounds[0], 2 * size + 1)[1::2]
        qcs_y = torch.linspace(0, new_upper_bounds[1], 2 * size + 1)[1::2]

        self.quadrant_axis_bounds.append((qb_x, qb_y))
        
        qcmg = torch.meshgrid(qcs_x, qcs_y, indexing="xy")
        qcs = torch.stack(qcmg, dim=2)
        qcs = qcs.reshape((size ** 2, 2))
        qcs = qcs.to(self.device)

        self.quadrant_centers.append(qcs)



