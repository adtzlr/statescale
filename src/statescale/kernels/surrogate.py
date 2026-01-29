from dataclasses import dataclass

import numpy as np


@dataclass
class SurrogateKernelParameters:
    "Surrogate kernel parameters."

    points: np.array
    U: np.array
    alpha: np.array
    alpha_mean: np.array
    modes: int
    shape: tuple


@dataclass
class SurrogateKernelData:
    "Surrogate kernel data."

    point_data: dict
    cell_data: dict
    field_data: dict


class SurrogateKernel:
    r"""A surrogate kernel.

    Parameters
    ----------
    point_data : dict
        A dict of point data.
    cell_data : dict
        A dict of cell data.
    field_data : dict
        A dict of field data.
    **kwargs
        Additional keyword arguments for the calibration of the kernel.

        modes : 2-tuple of int or None, optional
            Mode-range for the surrogate model. Default is (2, 10). If None, the modes
            are chosen in such a way that cumsum(S) / S >= threshold of the singular
            values are included.
        threshold : float, optional
            Default threshold to evaluate the number of modes for the surrogate model.
            Default is 0.995.

    Notes
    -----
    The surrogate kernel is a snapshot-based proper orthogonal decomposition (POD)
    surrogate with interpolation of modal coefficients and is based on [1]_, with a
    selection of the maximum number of modes as outlined in [2]_.

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
          - j-th vector component of flattened data
        * - ``m``
          - m-th mode of surrogate model
        * - ``a``
          - a-th timestep of signal

    First, the centered data at the snapshots is computed by subtracting the mean over
    all snapshots.

    ..  math::

        \Delta d_{sj} = d_{sj} - \operatorname{mean}(d)_j

    Then, a singular value decomposition (SVD) of the centered data is performed to
    obtain the principal components (modes) of the data. The number of modes to be used
    in the surrogate model is determined based on the provided mode range and the
    threshold for the cumulative sum of singular values.

    ..  math::

        \Delta d_{js} = U_{jm}\ S_{mm}\ V^H_{ms}

    ..  math::

        \alpha_{sm} &= \Delta d_{sj}\ U_{jm}

        \bar{\alpha}_{sm} &= \bar{d}_{sj}\ U_{jm}

    Next, the factors at the signal points are obtained by interpolating the modal
    coefficients from the snapshots to the signal points using the provided upscale
    function.

    ..  math::

        \alpha_{am} &= \text{upscale}(x_{si}, \alpha_{sm}, x_{ai})

        \beta_{am} &= \alpha_{am} + \bar{\alpha}_{am}

    The factors at the snapshots are computed by projecting the data onto the
    selected modes.

    ..  math::

        d_{aj} = \beta_{am}\ U_{mj}

    Finally, the kernel parameters are stored in
    :class:`~statescale.kernels.SurrogateKernelParameters` for later use in the
    surrogate model.

    See Also
    --------
    statescale.SnapshotModel : A model with point-, cell- and field-data at snapshots
        and with methods to interpolate the data at points of interest.
    statescale.kernels.GriddataKernel : A griddata kernel.

    References
    ----------
    .. [1] J. S. Hesthaven and S. Ubbiali, "Non-intrusive reduced order modeling of
            nonlinear problems using neural networks," Journal of Computational
            Physics, vol. 363, pp. 55-78, 2018.
    .. [2] P. Benner, S. Gugercin, and K. Willcox, "A survey of projection-based
            model reduction methods for parametric dynamical systems," SIAM Review,
            vol. 57, no. 4, pp. 483-531, 2015.
    """

    def __init__(self, snapshots, point_data, cell_data, field_data, **kwargs):

        self.kernel_data = SurrogateKernelData(
            point_data=self._calibrate(
                snapshots=snapshots,
                data=point_data,
                **kwargs,
            ),
            cell_data=self._calibrate(
                snapshots=snapshots,
                data=cell_data,
                **kwargs,
            ),
            field_data=field_data,
        )

    def _calibrate(self, snapshots, data, modes=(2, 10), threshold=0.995):
        r"""Calibrate the surrogate kernel.

        Parameters
        ----------
        snapshots : np.ndarray
            Snapshot points.
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
            alpha_mean = means.reshape(1, -1) @ U

            out[label] = SurrogateKernelParameters(
                points=snapshots,
                U=U,
                alpha=alpha,
                alpha_mean=alpha_mean,
                modes=modes_used,
                shape=values.shape[1:],
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

        alpha = upscale(
            points=kernel_parameters.points,
            values=kernel_parameters.alpha,
            xi=xi,
            **kwargs,
        )

        U_taken = kernel_parameters.U.T.reshape(-1, *kernel_parameters.shape)

        if indices is not None:
            U_taken = U_taken.take(indices=indices, axis=axis)

        beta = kernel_parameters.alpha_mean + alpha
        values_taken = np.einsum("am,m...->a...", beta, U_taken)

        return values_taken
