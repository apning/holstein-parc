# Standard library imports
import argparse
from contextlib import contextmanager
import heapq
import pathlib
import random
import time
import warnings
from collections import deque
from datetime import datetime, timedelta
from typing import Iterable

# Third-party imports
import torch

try:
    import pynvml

    HAS_PNYVML = True
except ImportError:
    HAS_PNYVML = False


def str_formatted_datetime():
    """
    Get current datetime as a formatted string.

    Returns:
        str: Datetime string in format 'YYYYMMDD-HHMMSS'.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def select_best_device(mode):
    """
    Select the best available GPU device based on specified criteria.

    Args:
        mode (str): Selection mode - 'm' for most free memory, 'u' for least utilization. 'u' requires pynvml package to be installed.

    Returns:
        torch.device: Selected device (GPU or CPU if no GPU available).

    Raises:
        Exception: If mode is not 'm' or 'u'.
    """
    if mode not in ["m", "u"]:
        raise ValueError(
            f'select_best_device: Acceptable inputs for mode are "m" (most free memory) and "u" (least utilization). You specified: {mode}'
        )

    if not torch.cuda.is_available():
        return torch.device("cpu")

    indices = list(range(torch.cuda.device_count()))
    random.shuffle(
        indices
    )  # shuffle the indices we iterate through so that, if, say, a bunch of processes scramble for GPUs at once, the first one won't get them all

    if mode == "m":
        max_free_memory = 0
        device_index = 0
        for i in indices:
            free_memory = torch.cuda.mem_get_info(i)[0]
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                device_index = i
        return torch.device(f"cuda:{device_index}")

    elif mode == "u":
        if HAS_PNYVML:
            pynvml.nvmlInit()
            min_util = 100
            device_index = 0
            for i in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # Get the handle for the target GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu  # GPU utilization percentage (integer)
                if gpu_utilization < min_util:
                    min_util = gpu_utilization
                    device_index = i
            pynvml.nvmlShutdown()

            # If all the GPUs are basically at max util, then make choice via memory availiability
            if min_util > 95:
                return select_best_device(mode="m")

            return torch.device(f"cuda:{device_index}")
        else:
            warnings.warn(
                "Utilization 'u' based selection is only available if pnyvml is available, but it is not. Please install pnyvml to use mode 'u'. Switching to mode 'm' (memory-based device selection)"
            )
            return select_best_device("m")


class Min_n_Items:
    def __init__(self, n: int):
        """
        Keeps a running record of the n lowest values so far and their associated epoch number
        Values are something like loss values, etc
        """

        self.n = n
        self.heap = []

    def record_epoch(self, val: float, epoch_num: int):
        """
        Records a value with associated epoch number
        If the value is smaller than the existing n values already in the heap, it replaces the largest value in the heap with the new value
        Else, the value is not recorded
        """

        # when given tuple, heapq sorts using first element. We negate value because heapq is a min-heap
        epoch_tuple = (-val, epoch_num)
        if len(self.heap) < self.n:
            heapq.heappush(self.heap, epoch_tuple)
        else:
            heapq.heappushpop(self.heap, epoch_tuple)

    def get_smallest_val(self, include_epoch=False):
        """
        Returns the smallest recorded value

        include_epoch (bool): Include epoch number of smallest value too
        """

        if include_epoch:
            return -max(self.heap)[0], max(self.heap)[1]

        return -max(self.heap)[0]

    def get_dict(self):
        """
        Returns a dict with the recorded values and their associated epoch values
        """
        return {epoch_tuple[1]: -epoch_tuple[0] for epoch_tuple in self.heap}

    def __str__(self):
        """
        Prints out a string with information about recorded values
        As you can see it is currently tailored for validation loss
        """
        min_str = f"Top {self.n} Epochs Info:\n"
        min_str += f"Average Val Loss Across Lowest {self.n} epochs: {self.avg_min_n_val()}\n"
        min_n_dict = self.get_dict()
        sorted_items_by_value = sorted(min_n_dict.items(), key=lambda item: item[1])
        for key, value in sorted_items_by_value:
            min_str += f"\tEpoch: {key}\tVal Loss: {value}\n"
        return min_str

    def avg_min_n_val(self, n=None):
        """
        Return the average value across the n smallest values
        If n is None then uses self.n (all tracked values)
        """
        if n is None:
            n = self.n
        elif n > self.n:
            warnings.warn(
                f"Min_n_Items average_min_n_val: Specified value n cannot be greater than tracked number of items. You specified: {n}. Defaulting to {self.n}, the number of all tracked items."
            )
            n = self.n
        min_dict = self.get_dict()
        sorted_min_vals = sorted(min_dict.values())
        return sum(sorted_min_vals[:n]) / n


class Running_Avg:
    """
    Maintain a running average of values.

    Efficiently computes average without storing all values.
    """

    """
    Maintains a running average of values added
    """

    def __init__(self):
        self.running_avg = 0.0
        self.counter = 0

    def add(self, val: float):
        """
        Add a value to the running average.

        Args:
            val (float): Value to add.
        """
        self.running_avg = (self.running_avg * self.counter + val) / (self.counter + 1)
        self.counter += 1

    def get_avg(self):
        """
        Get the current running average.

        Returns:
            float: Current average value.
        """
        return self.running_avg


class Multi_Running_Avgs:
    """
    Manage multiple running averages simultaneously.

    Can handle either list/tuple or dictionary of values.
    The structure is determined on first add() call.
    """

    def __init__(self):
        self.running_avgs = None

    def add(self, vals: Iterable[float] | dict[float]):
        # Initialize self.running_avgs if this is the first call to .add()
        if self.running_avgs is None:
            if isinstance(vals, dict):
                self.is_dict = True
                self.running_avgs = {k: Running_Avg() for k in vals}
            else:
                self.is_dict = False
                self.running_avgs = tuple(Running_Avg() for _ in range(len(vals)))
            self.n = len(self.running_avgs)

        if len(vals) != self.n:
            raise ValueError(
                f"Multi_Running_Avgs.add(): Length of iterable given ({len(vals)}) does not match expected length ({self.n})"
            )

        if self.is_dict:
            [self.running_avgs[k].add(v) for k, v in vals.items()]
        else:
            [running_avg.add(val) for running_avg, val in zip(self.running_avgs, vals)]

    def get_running_avgs(self) -> tuple[float] | dict[float]:
        """
        If self.running_avgs is a dict, return a dict with each of the running averages as floats. Otherwise, returns a tuple containing n floats representing each of the running averages
        """
        if self.running_avgs is None:
            raise ValueError("running_avgs is not initialized yet! Must call .add() at least once")

        if self.is_dict:
            return {k: running_avg.get_avg() for k, running_avg in self.running_avgs.items()}
        return tuple(running_avg.get_avg() for running_avg in self.running_avgs)

    def get_summed_avgs(self) -> float:
        """
        Returns the sum of all running averages
        """
        if self.running_avgs is None:
            raise ValueError("running_avgs is not initialized yet! Must call .add() at least once")

        if self.is_dict:
            return sum(running_avg.get_avg() for running_avg in self.running_avgs.values())
        return sum(self.get_running_avgs())

    def __len__(self):
        if hasattr(self, "n"):
            return self.n
        return 0


def get_off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Extract off-diagonal elements from square matrices.

    Args:
        x (torch.Tensor): Input tensor of shape (..., L, L).

    Returns:
        torch.Tensor: Off-diagonal elements of shape (..., L, L-1).
    """
    L = x.shape[-1]
    # Create a mask that is True everywhere except on the diagonal
    mask = ~torch.eye(L, dtype=bool, device=x.device)
    # Select off-diagonal elements, then reshape into (..., L, L-1)
    return x[..., mask].view(*x.shape[:-2], L, L - 1)


class MovingAverage:
    """
    Fixed-window moving average calculator.

    Maintains last n values and computes their average.
    Uses deque for O(1) add/remove operations.

    Args:
        n (int): Window size for moving average.
    """

    def __init__(self, n):
        if n <= 0:
            raise ValueError("n must be a positive integer")
        self.n = n
        self.values = deque(maxlen=n)
        self.total = 0.0  # running sum for O(1) average calculation

    def record(self, value):
        """
        Record a new value in the moving average window.

        Args:
            value: Value to add to the window.
        """
        if len(self.values) == self.n:
            oldest = self.values.popleft()
            self.total -= oldest
        self.values.append(value)
        self.total += value

    def get_avg(self):
        """
        Calculate the average of values in the current window.

        Returns:
            float: Average of values in the window.

        Raises:
            ValueError: If no values have been recorded yet.
        """
        if not self.values:
            raise ValueError("No values recorded yet")
        return self.total / len(self.values)


def get_project_root() -> pathlib.Path:
    """
    Determines the project root using a fixed relative path from this file.
    It assumes this file is located within the 'project_root/src' directory.
    Validates the assumed root by checking for an 'src' directory within it.

    Returns:
        pathlib.Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the 'src' directory is not found at the
                           assumed project root, indicating a potential
                           misconfiguration or change in directory structure.
        RuntimeError: If this utility file's location has changed such that
                      the fixed relative path logic is no longer valid.
    """
    try:
        # Get the absolute path of this file
        this_file_path = pathlib.Path(__file__).resolve()

        # Assumed structure: project_root/src/<this file>.py
        assumed_project_root = this_file_path.parent.parent
    except IndexError:
        # This would happen if Path(__file__).parent goes above filesystem root
        raise RuntimeError(
            f"The utility file '{__file__}' seems to be located too high in the "
            f"directory tree for the fixed relative path logic to apply. "
            f"Expected 'project_root/src/<this file>.py'. "
            f"Instead got {this_file_path}."
        )

    # Validate: Check for the presence of an 'src' directory in the assumed root.
    # This 'src' directory is the one directly under the project_root.
    expected_src_dir = assumed_project_root / "src"

    if not (expected_src_dir.exists() and expected_src_dir.is_dir()):
        raise FileNotFoundError(
            f"Validation failed: An 'src' directory was not found at the assumed "
            f"project root '{assumed_project_root}'.\n"
            f"This function expects the project structure to be 'project_root/src/...', "
            f"and this utility file ('{__file__}') to be at a certain fixed location "
            f"within 'src/'. If the structure or file location has changed, "
            f"this function may need an update."
        )

    return assumed_project_root


def str2bool(v: str) -> bool:
    """
    Accept common bool spellings and return Python bools.
    Returns None unchanged so the default can stay None.
    """
    if v is None or isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


@contextmanager
def timer(message: bool = "Time taken: {}", disable: bool = False):
    start = time.perf_counter()
    yield
    elapsed = timedelta(seconds=time.perf_counter() - start)
    if not disable:
        print(message.format(elapsed))
