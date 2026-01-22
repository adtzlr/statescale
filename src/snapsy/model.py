import numpy as np

from .evaluate import evaluate_data
from .surrogate import fit_surrogate_parameters
from .typing import ModelResult


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
    use_surrogate : bool, optional
        Use a surrogate model for interpolation. Default is False.
    modes : 2-tuple of int or None, optional
        Mode-range for the surrogate model. Default is (2, 10). If None, the modes are
        chosen in such a way that cumsum(S) / S >= threshold of the singular values are
        included. Only considered if use_surrogate is True.
    threshold : float, optional
        Default threshold to evaluate the number of modes for the surrogate model. Only
        considered if use_surrogate is True. Default is 0.995.

    Notes
    -----
    The surrogate model is a snapshot-based POD surrogate with interpolation of modal
    coefficients and is based on [1]_, with a selection of the maximum number of
    modes as outlined in [2]_.

    ..  list-table:: Naming conventions
        :header-rows: 1
        :widths: 25 75

        * - Symbol
          - Description
        * - ``x_si``
          - Snapshots
        * - ``d_s...``
          - Time-dependent data at snapshots with arbitrary trailing axes
        * - ``mean(d)_...``
          - Mean over all snapshots of data
        * - ``Δd_s...``
          - Centered data at snapshots
        * - ``U_...m``
          - Unitary matrix of U S Vh = svd(Δd_...s)
        * - ``α_sm``
          - Factors at snapshots to obtain the centered data
        * - ``x_ai``
          - Signal
        * - ``α_am``
          - Factors for the signal to obtain the centered data
        * - ``Δd_a...``
          - Centered data for the signal
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
        * - ``m``
          - m-th mode of surrogate model
        * - ``a``
          - a-th timestep of signal
        * - ``...``
          - optional arbritrary trailing axes

    Examples
    --------
    First, let's create and run a simple FEM model with FElupe. A record-function
    collects the point- and cell-data dicts at the snapshots.

    ..  code-block::

        import felupe as fem

        mesh = fem.Rectangle(n=6)
        region = fem.RegionQuad(mesh)
        field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

        boundaries = fem.dof.uniaxial(field, clamped=True)
        solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=2), field=field)

        snapshots = fem.math.linsteps([0, 1], num=3)
        ramp = {boundaries["move"]: snapshots}
        step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)

        point_data = []
        cell_data = []


        def record(*args, **kwargs):
            point_data.append(dict(u=field[0].values))
            cell_data.append(dict(E=field.evaluate.log_strain()))


        job = fem.Job(steps=[step], callback=record)
        job.evaluate()

    Then, use the lists of point- and cell-data at the snapshots to create a
    :class:`~snapsy.SnapshotModel`.

    ..  code-block::

        import snapsy

        model = snapsy.SnapshotModel(
            snapshots=snapshots,
            point_data=point_data,
            cell_data=cell_data,
            use_surrogate=False,
            # modes = (2, 10),  # choose min-max mode-range for surrogate model
            # threshold = 0.999,  # ratio of included singular values for surrogate model
        )

    A signal will be used to interpolate (evaluate) the point and cell data. The result
    can be converted to a list and supports iteration.

    ..  code-block::

        signal = fem.math.linsteps([0, 1], num=500)

        out = model.evaluate(signal)
        data = out[-5]

    The results are used to plot the deformed FEM model along with a chosen cell-data.
    Basic math, like transpose, can be applied to the model result. Any custom math-
    function can also be applied on the arrays of the dicts by
    :meth:`~snapsy.SnapshotModel.apply`.

    ..  code-block::

        import numpy as np

        field[0].values[:] = data.point_data["u"]
        view = field.view(
            point_data=data.point_data,
            cell_data=data.apply(np.mean)(axis=-2).T.cell_data,
        )
        view.plot("E", component=0).show()

    References
    ----------
    ..  [1] L. Sirovich, "Turbulence and the dynamics of coherent structures. I.
        Coherent structures", Quart. Appl. Math., vol. 45, no. 3, pp. 561–571, Oct.
        1987, doi: https://doi.org/10.1090/qam/910462.

    ..  [2] P. Holmes, J. L. Lumley, and G. Berkooz, Turbulence, Coherent Structures,
        Dynamical Systems and Symmetry, Cambridge, U.K.: Cambridge University Press,
        1996. doi: https://doi.org/10.1017/CBO9780511622700.
    """

    def __init__(
        self,
        snapshots,
        point_data=None,
        cell_data=None,
        field_data=None,
        use_surrogate=False,
        modes=(2, 10),
        threshold=0.995,
    ):

        if point_data is None:
            point_data = dict()

        if cell_data is None:
            cell_data = dict()

        if isinstance(point_data, list):
            point_data = self.from_list(point_data)

        if isinstance(cell_data, list):
            cell_data = self.from_list(cell_data)

        self.snapshots = snapshots
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data

        self.use_surrogate = use_surrogate
        self.surrogate = dict()
        self.modes = modes
        self.threshold = threshold

        if self.use_surrogate:
            self.surrogate = self._fit_surrogate(self.point_data, self.cell_data)
        else:
            self.surrogate = ModelResult(
                point_data=None, cell_data=None, field_data=self.field_data
            )

    @staticmethod
    def from_list(data):
        new_data = {}
        for label in data[0].keys():
            list_of_data = []
            for d in data:
                list_of_data.append(d[label])
            new_data[label] = np.array(list_of_data)
        return new_data

    def save(self, filename):
        np.save(filename, self)

    @classmethod
    def load(cls, filename):
        return np.load(filename, allow_pickle=True).item()

    def evaluate(self, xi, method="griddata", **kwargs):
        r"""Evaluate the point- and cell-data at xi.

        Parameters
        ----------
        xi : np.array
            Points at which to interpolate data.
        method : str, optional
            Use ``"griddata"`` or ``"rbf"``. Default is ``"griddata"``.
        **kwargs : dict
            Optional keyword-arguments are passed to griddata.

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
        **kwargs : dict
            Optional keyword-arguments are passed to griddata.

        Returns
        -------
        ModelResult
            The model result with attributes for time-dependent ``point_data``,
            (empty) ``cell_data`` and time-independent ``field_data``.
        """

        point_data = evaluate_data(
            snapshots=self.snapshots,
            data=self.point_data,
            xi=xi,
            use_surrogate=self.use_surrogate,
            surrogate=self.surrogate.point_data,
            indices=indices,
            axis=axis,
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
        **kwargs : dict
            Optional keyword-arguments are passed to griddata.

        Returns
        -------
        ModelResult
            The model result with attributes for time-dependent (empty) ``point_data``,
            ``cell_data`` and time-independent ``field_data``.
        """

        cell_data = evaluate_data(
            snapshots=self.snapshots,
            data=self.cell_data,
            xi=xi,
            use_surrogate=self.use_surrogate,
            surrogate=self.surrogate.cell_data,
            indices=indices,
            axis=axis,
            method=method,
            **kwargs,
        )

        return ModelResult(
            point_data={}, cell_data=cell_data, field_data=self.field_data
        )

    def _fit_surrogate(self, point_data, cell_data, **kwargs):

        return ModelResult(
            point_data=fit_surrogate_parameters(
                data=point_data,
                modes=self.modes,
                threshold=self.threshold,
                **kwargs,
            ),
            cell_data=fit_surrogate_parameters(
                data=cell_data,
                modes=self.modes,
                threshold=self.threshold,
                **kwargs,
            ),
            field_data=self.field_data,
        )
