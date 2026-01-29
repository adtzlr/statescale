import numpy as np

from ._containers import ModelResult
from ._evaluate import evaluate_data
from .kernels import GriddataKernel, SurrogateKernel


class SnapshotModel:
    r"""A model with point-, cell- and field-data at snapshots and with methods to
    interpolate the data at points of interest.

    Parameters
    ----------
    snapshots : np.array
        Snapshots of shape (n_snapshots, n_dim). Note that a signal, used for the
        evaluation of the data, must have equal n_dim.
    point_data : list of dict or dict or None, optional
        A dict of point data. The lengths of all arrays must be equal and of shape
        (n_snapshots, ...). If a list of dict is given, each item must hold the dict of
        a single snapshot.
    cell_data : list of dict or dict or None, optional
        A dict of cell data. The lengths of all arrays must be equal and of shape
        (n_snapshots, ...). If a list of dict is given, each item must hold the dict of
        a single snapshot.
    field_data : dict or None, optional
        A dict of time-independent field data.
    kernel : str, optional
        The kernel to be used for interpolation. Either ``"surrogate"`` or
        ``"griddata"``. Default is ``"griddata"``.
    **kwargs : dict, optional
        Additional keyword arguments for the calibration of the kernel. See
        :class:`~statescale.kernels.SurrogateKernel` and
        :class:`~statescale.kernels.GriddataKernel` for more details.

    Notes
    -----

    ..  list-table:: Naming conventions
        :header-rows: 1
        :widths: 25 75

        * - Symbol
          - Description
        * - ``x_si``
          - Snapshots
        * - ``d_s...``
          - Time-dependent data at snapshots with arbitrary trailing axes
        * - ``x_ai``
          - Signal
        * - ``d_a...``
          - Data for the signal (with arbitrary trailing axes)

    ..  list-table:: Indices
        :header-rows: 1
        :widths: 25 75

        * - Index
          - Description
        * - ``s``
          - s-th snapshot
        * - ``i``
          - i-th vector component of snapshot / signal
        * - ``a``
          - a-th (time-) step of signal
        * - ``...``
          - optional arbritrary trailing axes

    Examples
    --------
    A minimal example. Snapshots must have shapes ``(n_snapshots, n_dim)``, point- and
    cell-data ``(n_snapshots, ...)`` and the dimension of the signal must be compatible
    with snapshots, i.e. ``(n_steps, n_dim)``. The second dimension of snapshots and the
    signal are optional, 1d-arrays are also supported. The model result will be of shape
    ``(n_steps, ...)``.

    1.  Array-based input data

        ..  plot::
            :context:

            import numpy as np
            import statescale

            snapshots = np.linspace(0, 1, num=3).reshape(-1, 1)  # 3 snapshots, 1 parameter
            point_data = {"displacement": np.random.rand(3, 9, 3)}  # 3 snapshots, 9 points, 3 dim
            cell_data = {"strain": np.random.rand(3, 4, 6)}  # 3 snapshots, 4 cells, 6 dim
            field_data = {"id": 1001}  # time-independent data

            model = statescale.SnapshotModel(
                snapshots=snapshots,
                point_data=point_data,
                cell_data=cell_data,
                field_data=field_data,
                # kernel="surrogate",  # use a POD surrogate model
                # modes=(2, 10),  # min- and max no. of modes for surrogate model
                # threshold=0.999,  # min. energy threshold for surrogate model
            )

            signal = np.linspace(0, 1, num=20).reshape(-1, 1)  # 20 items, 1 parameter

            # a `ModelResult` object with `point_data`, `cell_data` and `field_data`.
            res = model.evaluate(signal)

    2.  List-based input data

        If the data is list-based, the model can also import lists of dicts, with per-
        snapshot list items. Model results also support indexing and a conversion to
        lists of dicts.

        ..  plot::
            :context:

            import numpy as np
            import statescale

            point_data = [
                {"displacement": np.random.rand(6, 2)},  # 1. snapshot, 6 points, 2 dim
                {"displacement": np.random.rand(6, 2)},  # 2. snapshot, 6 points, 2 dim
                {"displacement": np.random.rand(6, 2)},  # 3. snapshot, 6 points, 2 dim
            ]
            cell_data = [
                {"strain": np.random.rand(4, 2, 2)},  # 1. snapshot, 4 cells, (2, 2) dim
                {"strain": np.random.rand(4, 2, 2)},  # 2. snapshot, 4 cells, (2, 2) dim
                {"strain": np.random.rand(4, 2, 2)},  # 3. snapshot, 4 cells, (2, 2) dim
            ]

            model = statescale.SnapshotModel(
                snapshots=snapshots,
                point_data=point_data,
                cell_data=cell_data,
                field_data=field_data,
            )

            # `point_data`, `cell_data` and `field_data` for step 5 of the signal.
            res_5 = model.evaluate(signal)[5]

    Any NumPy-function may be applied to the model result data on all time-dependent
    arrays. E.g., the mean over all cells (here, the first axis) of the cell-data is
    evaluated by:

    ..  plot::
        :context:

        res_5_mean = res_5.apply(
            np.mean, on_point_data=False, on_cell_data=True
        )(axis=0)

    See Also
    --------
    statescale.kernels.SurrogateKernel : A surrogate kernel.
    statescale.kernels.GriddataKernel : A griddata kernel.

    """

    def __init__(
        self,
        snapshots,
        point_data=None,
        cell_data=None,
        field_data=None,
        kernel="griddata",
        **kwargs,
    ):

        if point_data is None:
            point_data = dict()

        if cell_data is None:
            cell_data = dict()

        if field_data is None:
            field_data = dict()

        if isinstance(point_data, list):
            point_data = self._from_list(point_data)

        if isinstance(cell_data, list):
            cell_data = self._from_list(cell_data)

        self.snapshots = snapshots
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data

        Kernel = {
            "surrogate": SurrogateKernel,
            "griddata": GriddataKernel,
        }[kernel]
        self.kernel = Kernel(
            snapshots=self.snapshots,
            point_data=self.point_data,
            cell_data=self.cell_data,
            field_data=self.field_data,
            **kwargs,
        )

    @staticmethod
    def _from_list(data):
        "Return a data dict from given per-snapshot list of data dicts."

        new_data = {}
        for label in data[0].keys():
            list_of_data = []
            for d in data:
                list_of_data.append(d[label])
            new_data[label] = np.array(list_of_data)
        return new_data

    def save_model(self, filename="model.npz"):
        "Save the model to a file."

        np.savez(filename, model=self)

    def save_kernel(self, filename="kernel.npz"):
        "Save the kernel to a file."
        np.savez(filename, kernel=self.kernel)

    @classmethod
    def load_model(cls, filename):
        "Return a model, loaded from a file."
        return np.load(filename, allow_pickle=True)["model"].item()

    @classmethod
    def load_kernel(cls, filename):
        "Return a model with the loaded kernel file."

        kernel = np.load(filename, allow_pickle=True)["kernel"].item()
        return cls.from_kernel(kernel)

    @classmethod
    def from_kernel(cls, kernel):
        "Return a model with the loaded kernel."

        model = cls(
            snapshots=np.zeros(0),
            point_data=None,
            cell_data=None,
            field_data=kernel.kernel_data.field_data,
        )
        model.kernel = kernel
        return model

    def evaluate(self, xi, method="griddata", **kwargs):
        r"""Evaluate the point- and cell-data at xi.

        Parameters
        ----------
        xi : np.array
            Points at which to interpolate data.
        method : str, optional
            Use ``"griddata"`` or ``"rbf"``. Default is ``"griddata"``.
        **kwargs : dict, optional
            Optional keyword-arguments are passed to the :meth:`evaluate` method of the
            kernel. See :class:`~statescale.kernels.SurrogateKernel` and
            :class:`~statescale.kernels.GriddataKernel` for more details.

        Returns
        -------
        ModelResult
            The model result with attributes for time-dependent ``point_data``,
            ``cell_data`` and time-independent ``field_data``.
        """

        point_data = self.evaluate_point_data(xi=xi, method=method, **kwargs).point_data
        cell_data = self.evaluate_cell_data(xi=xi, method=method, **kwargs).cell_data

        return ModelResult(
            point_data=point_data, cell_data=cell_data, field_data=self.field_data
        )

    def evaluate_point_data(
        self, xi, indices=None, axis=None, method="griddata", **kwargs
    ):
        r"""Evaluate point-data at xi.

        Parameters
        ----------
        xi : np.array
            Points at which to interpolate data.
        indices : array_like or None, optional
            The indices of the values to extract. Also allow scalars for indices.
            Default is None.
        axis : int or None, optional
            The axis over which to select values. By default, the flattened input
            array is used. Default is None.
        method : str, optional
            Use ``"griddata"`` or ``"rbf"``. Default is ``"griddata"``.
        **kwargs : dict, optional
            Optional keyword-arguments are passed to the :meth:`evaluate` method of the
            kernel. See :class:`~statescale.kernels.SurrogateKernel` and
            :class:`~statescale.kernels.GriddataKernel` for more details.

        Returns
        -------
        ModelResult
            The model result with attributes for time-dependent ``point_data``,
            (empty) ``cell_data`` and time-independent ``field_data``.
        """

        point_data = evaluate_data(
            xi=xi,
            indices=indices,
            axis=axis,
            kernel_evaluate=self.kernel.evaluate,
            kernel_data=self.kernel.kernel_data.point_data,
            method=method,
            **kwargs,
        )

        return ModelResult(
            point_data=point_data, cell_data={}, field_data=self.field_data
        )

    def evaluate_cell_data(
        self, xi, indices=None, axis=None, method="griddata", **kwargs
    ):
        r"""Evaluate cell-data at xi.

        Parameters
        ----------
        xi : np.array
            Points at which to interpolate data.
        indices : array_like or None, optional
            The indices of the values to extract. Also allow scalars for indices.
            Default is None.
        axis : int or None, optional
            The axis over which to select values. By default, the flattened input
            array is used. Default is None.
        method : str, optional
            Use ``"griddata"`` or ``"rbf"``. Default is ``"griddata"``.
        **kwargs : dict, optional
            Optional keyword-arguments are passed to the :meth:`evaluate` method of the
            kernel. See :class:`~statescale.kernels.SurrogateKernel` and
            :class:`~statescale.kernels.GriddataKernel` for more details.

        Returns
        -------
        ModelResult
            The model result with attributes for time-dependent (empty) ``point_data``,
            ``cell_data`` and time-independent ``field_data``.
        """

        cell_data = evaluate_data(
            xi=xi,
            indices=indices,
            axis=axis,
            kernel_evaluate=self.kernel.evaluate,
            kernel_data=self.kernel.kernel_data.cell_data,
            method=method,
            **kwargs,
        )

        return ModelResult(
            point_data={}, cell_data=cell_data, field_data=self.field_data
        )
