import numpy as np

from .models import SurrogateParameters


def fit_surrogate_parameters(data, modes, threshold, **kwargs):

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

        out[label] = SurrogateParameters(
            means=means,
            U=U,
            alpha=alpha,
            modes=modes_used,
        )

    return out
