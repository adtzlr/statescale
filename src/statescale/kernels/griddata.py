from dataclasses import dataclass

import numpy as np


@dataclass
class GriddataKernelParameters:
    "Griddata kernel parameters."

    points: np.array
    values: np.array


@dataclass
class GriddataKernelData:
    "Griddata kernel data."

    point_data: dict
    cell_data: dict
    field_data: dict


class GriddataKernel:
    """A griddata kernel.

    Parameters
    ----------
    point_data : dict
        A dict of point data.
    cell_data : dict
        A dict of cell data.
    field_data : dict
        A dict of field data.
    **kwargs
        Additional keyword arguments (not used).

    See Also
    --------
    statescale.SnapshotModel : A model with point-, cell- and field-data at snapshots
        and with methods to interpolate the data at points of interest.
    statescale.kernels.SurrogateKernel : A surrogate kernel.
    """

    def __init__(self, snapshots, point_data, cell_data, field_data, **kwargs):
        self.kernel_data = GriddataKernelData(
            point_data=self._calibrate(snapshots=snapshots, data=point_data),
            cell_data=self._calibrate(snapshots=snapshots, data=cell_data),
            field_data=field_data,
        )

    def _calibrate(self, snapshots, data):
        """Calibrate the surrogate kernel.

        Parameters
        ----------
        snapshots : np.ndarray
            Snapshot points.
        data : dict
            Dict with snapshot data.

        Returns
        -------
        GriddataKernelParameters
            A dict with the griddata kernel parameters.
        """
        out = dict()

        for label, values in data.items():
            out[label] = GriddataKernelParameters(
                points=snapshots,
                values=values,
            )

        return out

    @staticmethod
    def evaluate(
        xi,
        upscale,
        kernel_parameters,
        indices=None,
        axis=None,
        **kwargs,
    ):

        values_taken = kernel_parameters.values

        if indices is not None:
            values_taken = values_taken.take(indices=indices, axis=axis)

        return upscale(kernel_parameters.points, values_taken, xi, **kwargs)
