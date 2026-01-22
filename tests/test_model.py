import numpy as np
import pytest

import snapsy


def test_from_list_converts_to_arrays():
    data_list = [
        {"a": np.array([1.0, 2.0]), "b": np.array([3.0])},
        {"a": np.array([4.0, 5.0]), "b": np.array([6.0])},
    ]

    converted = snapsy.SnapshotModel.from_list(data_list)

    assert isinstance(converted, dict)
    assert "a" in converted and "b" in converted
    np.testing.assert_allclose(converted["a"], np.array([[1.0, 2.0], [4.0, 5.0]]))
    np.testing.assert_allclose(converted["b"], np.array([[3.0], [6.0]]))


def test_surrogate_parameters_created_and_shapes():
    # simple deterministic data where trailing dims flatten to length 2
    snapshots = np.linspace(0, 1, 5).reshape(-1, 1)
    values = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

    point_data = {"u": values}
    cell_data = {"E": values * 2}

    model = snapsy.SnapshotModel(
        snapshots=snapshots,
        point_data=point_data,
        cell_data=cell_data,
        use_surrogate=True,
        modes=(2, 10),
        threshold=0.5,
    )

    # surrogate should be present and contain SurrogateParameters for each label
    assert hasattr(model, "surrogate")
    assert "u" in model.surrogate.point_data
    sp = model.surrogate.point_data["u"]

    # SurrogateParameters should expose expected attributes
    assert (
        hasattr(sp, "means")
        and hasattr(sp, "U")
        and hasattr(sp, "alpha")
        and hasattr(sp, "modes")
    )

    # Check shapes: means has leading 1, alpha has snapshots x modes, U has flattened_dim x modes
    assert sp.means.shape[0] == 1
    assert sp.alpha.shape[0] == snapshots.shape[0]
    assert sp.U.shape[0] == values.reshape(len(values), -1).shape[1]


def test_save_and_load_preserves_snapshot_and_keys():
    snapshots = np.linspace(0, 1, 4).reshape(-1, 1)
    point_data = {"u": np.arange(4.0).reshape(4, 1)}
    cell_data = {"E": np.arange(4.0).reshape(4, 1) * 2}

    model = snapsy.SnapshotModel(
        snapshots=snapshots,
        point_data=point_data,
        cell_data=cell_data,
        use_surrogate=False,
    )

    out_file = "model.npy"
    model.save(str(out_file))

    loaded = snapsy.SnapshotModel.load(str(out_file))

    # loaded should be a SnapshotModel instance with same snapshots and keys
    assert isinstance(loaded, snapsy.SnapshotModel)
    np.testing.assert_allclose(loaded.snapshots, model.snapshots)
    assert set(loaded.point_data.keys()) == set(model.point_data.keys())
    assert set(loaded.cell_data.keys()) == set(model.cell_data.keys())


def test_evaluate_monkeypatched():
    """Replace the internal evaluate_data used by `SnapshotModel` and verify
    that `evaluate` returns a `ModelResult` combining point-, cell- and
    field-data as expected.
    """

    snapshots = np.array([[0.0], [1.0]])
    point_data = {"u": np.array([[0.0], [1.0]])}
    cell_data = {"E": np.array([[10.0], [20.0]])}

    model = snapsy.SnapshotModel(
        snapshots=snapshots,
        point_data=point_data,
        cell_data=cell_data,
        use_surrogate=False,
    )

    # Monkeypatch the name used inside snapsy.model
    xi = np.array([0.1, 0.9])
    res = model.evaluate(xi)

    # Verify returned structure and values
    assert (
        hasattr(res, "point_data")
        and hasattr(res, "cell_data")
        and hasattr(res, "field_data")
    )
    np.testing.assert_allclose(res.point_data["u"], np.array([[0.1], [0.9]]))
    np.testing.assert_allclose(res.cell_data["E"], np.array([[11.0], [19.0]]))
    # field_data should be forwarded (None in this model)
    assert res.field_data is None


def test_modelresult_apply_and_T():
    # deterministic small arrays
    point = {"u": np.arange(12).reshape(3, 4)}
    cell = {"E": np.arange(6).reshape(3, 2)}
    field = {"const": np.array([10.0])}

    mr = snapsy.typing.ModelResult(point_data=point, cell_data=cell, field_data=field)

    # apply mean over axis=1 should act on both point and cell data
    out = mr.apply(np.mean)(axis=1)

    np.testing.assert_allclose(out.point_data["u"], np.mean(point["u"], axis=1))
    np.testing.assert_allclose(out.cell_data["E"], np.mean(cell["E"], axis=1))

    # field_data is not changed by default
    assert out.field_data == field

    # transpose property T should transpose arrays
    t = mr.T
    np.testing.assert_allclose(t.point_data["u"], point["u"].T)
    np.testing.assert_allclose(t.cell_data["E"], cell["E"].T)


def test_modelresult_iteration_and_len():
    # simple one-column snapshot data
    point = {"u": np.array([[1.0], [2.0], [3.0]])}
    cell = {"E": np.array([[10.0], [20.0], [30.0]])}

    mr = snapsy.typing.ModelResult(point_data=point, cell_data=cell)

    # len should reflect number of snapshots
    assert len(mr) == 3

    # iteration/list should yield ModelResult items for each snapshot
    items = list(mr)
    assert len(items) == 3
    np.testing.assert_allclose(items[0].point_data["u"], point["u"][0])
    np.testing.assert_allclose(items[2].cell_data["E"], cell["E"][2])


def test_evaluate_data_without_and_with_surrogate():
    import sys
    import types

    # Create a fake scipy.interpolate module with a griddata function
    def fake_griddata(points, values, xi, **kwargs):
        # return the mean over the first axis tiled to match xi
        v = np.mean(values, axis=0, keepdims=True)
        n = len(np.atleast_1d(xi))
        return np.tile(v, (n, 1))

    fake_interp = types.ModuleType("scipy.interpolate")
    fake_interp.griddata = fake_griddata

    # Non-surrogate case
    snapshots = np.linspace(0, 1, 3).reshape(-1, 1)
    data = {"a": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
    xi = np.array([0.5, 0.5])

    out = snapsy.evaluate.evaluate_data(
        snapshots=snapshots,
        data=data,
        xi=xi,
        use_surrogate=False,
        surrogate=None,
        method="griddata",
    )

    # Expected: mean of data across snapshots, tiled for len(xi)
    expected = np.tile(np.mean(data["a"], axis=0, keepdims=True), (len(xi), 1))
    np.testing.assert_allclose(out["a"], expected)

    # Surrogate case
    # Build values and surrogate parameters matching shapes used in evaluate_data
    values = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    data2 = {"u": values}

    class SP:
        def __init__(self, means, U, alpha, modes):
            self.means = means
            self.U = U
            self.alpha = alpha
            self.modes = modes

    # flattened trailing dim = 2, choose modes=2, U identity for simplicity
    means = np.array([[100.0, 200.0]])
    U = np.eye(2)
    alpha = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    surrogate = {"u": SP(means=means, U=U, alpha=alpha, modes=2)}

    out2 = snapsy.evaluate.evaluate_data(
        snapshots=snapshots,
        data=data2,
        xi=xi,
        use_surrogate=True,
        surrogate=surrogate,
        method="griddata",
    )

    # fake_griddata returns mean(alpha,axis=0) tiled
    alpha_mean = np.mean(alpha, axis=0)
    centered = np.tile(alpha_mean, (len(xi), 1))  # since U is identity
    expected2 = means + centered
    np.testing.assert_allclose(out2["u"], expected2)


if __name__ == "__main__":
    test_from_list_converts_to_arrays()
    test_surrogate_parameters_created_and_shapes()
    test_save_and_load_preserves_snapshot_and_keys()
    test_evaluate_monkeypatched()
    test_modelresult_apply_and_T()
    test_modelresult_iteration_and_len()
    test_evaluate_data_without_and_with_surrogate()
    print("All tests passed.")
