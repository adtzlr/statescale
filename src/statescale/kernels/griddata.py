import numpy as np
from dataclasses import dataclass


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
    **kwargs
        Additional keyword arguments (not used).
    """

    def __init__(self, point_data, cell_data, **kwargs):
        self.kernel_data = GriddataKernelData(
            point_data={},
            cell_data={},
            field_data={},
        )

    @staticmethod
    def evaluate(
        snapshots,
        values,
        xi,
        interp,
        kernel_data=None,
        indices=None,
        axis=None,
        **kwargs,
    ):

        values_taken = values

        if indices is not None:
            values_taken = values_taken.take(indices=indices, axis=axis, **kwargs)

        return interp(snapshots, values_taken, xi)
