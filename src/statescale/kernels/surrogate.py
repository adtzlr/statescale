import numpy as np
from dataclasses import dataclass


@dataclass
class SurrogateKernelParameters:
    "Surrogate kernel parameters."

    means: np.array
    U: np.array
    alpha: np.array
    modes: int


@dataclass
class SurrogateKernelData:
    "Surrogate kernel data."

    point_data: dict
    cell_data: dict


class SurrogateKernel:
    """A surrogate kernel.
    
    Parameters
    ----------
    point_data : dict
        A dict of point data.
    cell_data : dict
        A dict of cell data.
    **kwargs
        Additional keyword arguments for the calibration of the kernel.

        modes : 2-tuple of int or None, optional
            Mode-range for the surrogate model. Default is (2, 10). If None, the modes
            are chosen in such a way that cumsum(S) / S >= threshold of the singular
            values are included.
        threshold : float, optional
            Default threshold to evaluate the number of modes for the surrogate model.
            Default is 0.995.
    """

    def __init__(self, point_data, cell_data, **kwargs):

        self.kernel_data = SurrogateKernelData(
            point_data=self._calibrate(
                data=point_data,
                **kwargs,
            ),
            cell_data=self._calibrate(
                data=cell_data,
                **kwargs,
            ),
        )

    def _calibrate(self, data, modes=(2, 10), threshold=0.995):
        """Calibrate the surrogate kernel.
        
        Parameters
        ----------
        data : dict
            Dict with snapshot data.
        modes : 2-tuple of int or None, optional
            Mode-range for the surrogate model. Default is (2, 10). If None, the modes
            are chosen in such a way that cumsum(S) / S >= threshold of the singular
            values are included.
        threshold : float, optional
            Default threshold to evaluate the number of modes for the surrogate model.
            Default is 0.995.

        Returns
        -------
        SurrogateKernelParameters
            A dict with the surrogate kernel parameters.
        """
        out = dict()

        for label, values in data.items():
            means = values.mean(axis=0, keepdims=True)
            centered = (values - means).reshape(len(values), -1)

            U, S, Vh = np.linalg.svd(centered.T, full_matrices=False)

            S2 = S**2
            modes_calc = np.argwhere(np.cumsum(S2) / np.sum(S2) > threshold).min()

            # min_modes <= modes <= max_modes
            modes_used = np.maximum(modes[0], np.minimum(modes_calc, modes[1]))

            U = U[:, :modes_used]

            alpha = centered @ U

            out[label] = SurrogateKernelParameters(
                means=means,
                U=U,
                alpha=alpha,
                modes=modes_used,
            )

        return out

    @staticmethod
    def evaluate(
        snapshots,
        values,
        xi,
        upscale,
        kernel_parameters,
        indices=None,
        axis=None,
        **kwargs,
    ):

        alpha = upscale(
            points=snapshots,
            values=kernel_parameters.alpha,
            xi=xi,
            **kwargs,
        )

        means_taken = kernel_parameters.means
        U_taken = kernel_parameters.U.T.reshape(-1, *values.shape[1:])

        if indices is not None:
            U_taken = U_taken.take(indices=indices, axis=axis)
            means_taken = means_taken.take(indices=indices, axis=axis)

        centered_taken = np.einsum("am,m...->a...", alpha, U_taken)
        return means_taken + centered_taken
