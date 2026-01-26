from dataclasses import dataclass
from functools import wraps

import numpy as np


@dataclass
class SurrogateParameters:
    "Surrogate model parameters."

    means: np.array
    U: np.array
    alpha: np.array
    modes: int


@dataclass
class ModelResult:
    "Snapshot model result data."

    point_data: list | dict | None = None
    cell_data: list | dict | None = None
    field_data: dict | None = None

    def __len__(self):
        if self.point_data:
            return next(iter(self.point_data.values())).shape[0]
        elif self.cell_data:
            return next(iter(self.cell_data.values())).shape[0]
        else:
            return 0

    def __getitem__(self, t):
        return ModelResult(
            point_data={k: v[t] for k, v in self.point_data.items()},
            cell_data={k: v[t] for k, v in self.cell_data.items()},
            field_data=self.field_data,
        )

    def __iter__(self):
        for t in range(len(self)):
            yield self[t]

    @property
    def T(self):
        return ModelResult(
            point_data={k: v.T for k, v in self.point_data.items()},
            cell_data={k: v.T for k, v in self.cell_data.items()},
            field_data=self.field_data,
        )

    def as_mesh(self, points=None, cells=None, cell_type=None, mesh=None):
        """Return a PyVista unstructured grid mesh with the model result data.

        Parameters
        ----------
        points : np.ndarray, optional
            The points of the mesh. Required if no mesh is provided. Default is None.
        cells : np.ndarray, optional
            The cells of the mesh. Required if no mesh is provided. Default is None.
        cell_type : str, optional
            The cell type of the mesh. Required if no mesh is provided. Default is None.
        mesh : felupe.Mesh or None, optional
            An existing mesh to attach the model result data to. Default is None.

        Returns
        -------
        mesh_with_data : pyvista.UnstructuredGrid
            The mesh with the model result data attached.

        """
        import felupe

        if mesh is None:
            mesh = felupe.Mesh(
                points=points,
                cells=cells,
                cell_type=cell_type,
            )

        view = mesh.view(point_data=self.point_data, cell_data=self.cell_data)
        return view.mesh

    def mean(self, *args, **kwargs):
        "Compute the arithmetic :func:`~numpy.mean` along the specified axis."
        return self.apply(np.mean)(*args, **kwargs)

    def apply(self, func, on_point_data=True, on_cell_data=True, on_field_data=False):
        """Apply any function to the arrays of all or selected data dicts.

        Parameters
        ----------
        func : callable
            The function to be applied.
        on_point_data : bool, optional
            A flag to apply the function on the point data. Default is True.
        on_cell_data : bool, optional
            A flag to apply the function on the cell data. Default is True.
        on_field_data : bool, optional
            A flag to apply the function on the field data. Default is False.

        Returns
        -------
        apply_func : callable
            A transformed function, which applies the given function to all arrays.

            Parameters
            ----------
            *args : tuple, optional
                Additional arguments are passed to the given function.
            **kwargs : dict, optional
                Additional arguments are passed to the given function.

        Notes
        -----
        By default, only the time-dependent point- and cell-data arrays are modified.

        Examples
        --------

        ..  code-block::

            >>> import numpy as np
            >>> import statescale
            >>>
            >>> res = statescale.Modelresult(point_data={"u": np.random.rand(25, 100, 8, 3)})
            >>> out = res.apply(np.mean)(axis=-2)
            >>>
            >>> out.point_data["u"].shape
            (25, 100, 3)
        """

        @wraps(func)
        def apply_func(*args, **kwargs):

            point_data = self.point_data
            cell_data = self.cell_data
            field_data = self.field_data

            if on_point_data and self.point_data is not None:
                point_data = {
                    k: func(v, *args, **kwargs) for k, v in self.point_data.items()
                }

            if on_cell_data and self.cell_data is not None:
                cell_data = {
                    k: func(v, *args, **kwargs) for k, v in self.cell_data.items()
                }

            if on_field_data and self.field_data is not None:
                field_data = {
                    k: func(v, *args, **kwargs) for k, v in self.field_data.items()
                }

            return ModelResult(
                point_data=point_data,
                cell_data=cell_data,
                field_data=field_data,
            )

        return apply_func
