import numpy as np


def evaluate_data(
    snapshots,
    data,
    xi,
    use_surrogate,
    surrogate,
    indices=None,
    axis=None,
    method="griddata",
    **kwargs,
):
    r"""Evaluate the data at xi. Selected indices may be provided for a given
    axis.

    Parameters
    ----------
    snapshots : np.array
        Snapshots of shape (n_snapshots, n_dim). Note that a signal, used for the
        evaluation of the data, must have equal n_dim.
    data : dict
        Dict with snapshot data.
    xi : np.array
        Points at which to interpolate data.
    use_surrogate : bool
        Use a surrogate model for interpolation.
    surrogate : SGParameters
        Surrogate model parameters.
    indices : array_like or None, optional
        The indices of the values to extract. Also allow scalars for indices.
        Default is None.
    axis : int or None, optional
        The axis over which to select values. By default, the flattened input
        array is used. Default is None.
    method : str, optional
        Use ``"griddata"`` or ``"rbf"``. Default is ``"griddata"``.
    **kwargs : dict
        Optional keyword-arguments are passed to the interpolation method.

    Returns
    -------
    dict
        A dict with the interpolated data.
    """

    out = dict()

    if method == "griddata":
        from scipy.interpolate import griddata

        def interp(points, values, xi, **kwargs):

            # griddata requires points and xi as 1d array for dim=1
            if len(points.shape) == 2 and points.shape[1] == 1:
                points = points.ravel()

            if len(xi.shape) == 2 and xi.shape[1] == 1:
                xi = xi.ravel()

            return griddata(points, values, xi, **kwargs)

    elif method == "rbf":
        from scipy.interpolate import RBFInterpolator

        def interp(points, values, xi, **kwargs):

            # RBFInterpolator requires points and xi as 2d array, also for dim=1
            if len(points.shape) == 1:
                points = points.reshape(-1, 1)

            if len(xi.shape) == 1:
                xi = xi.reshape(-1, 1)
            return RBFInterpolator(y=points, d=values, **kwargs)(x=xi)

    else:
        raise ValueError("Method not supported.")

    if use_surrogate:

        for label, values in data.items():

            alpha = interp(
                points=snapshots,
                values=surrogate[label].alpha,
                xi=xi,
                **kwargs,
            )

            means_taken = surrogate[label].means
            U_taken = surrogate[label].U.T.reshape(-1, *values.shape[1:])

            if indices is not None:
                U_taken = U_taken.take(indices=indices, axis=axis, **kwargs)
                means_taken = means_taken.take(indices=indices, axis=axis, **kwargs)

            centered_taken = np.einsum("am,m...->a...", alpha, U_taken)
            out[label] = means_taken + centered_taken

    else:
        for label, values in data.items():
            values_taken = values

            if indices is not None:
                values_taken = values_taken.take(indices=indices, axis=axis, **kwargs)

            out[label] = interp(snapshots, values_taken, xi)

    return out
