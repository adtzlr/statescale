import numpy as np
import pytest
import numpy as np

import statescale


def test_snapshot_model():

    snapshots = np.linspace(0, 1, num=3).reshape(-1, 1)  # 3 snapshots, 1 parameter
    point_data = {
        "displacement": np.random.rand(3, 9, 3)
    }  # 3 snapshots, 9 points, 3 dim
    cell_data = {"strain": np.random.rand(3, 4, 6)}  # 3 snapshots, 4 cells, 6 dim
    field_data = {"id": 1001}  # time-independent data

    model = statescale.SnapshotModel(
        snapshots=snapshots,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        # kernel="surrogate",  # use a POD surrogate model
        # modes=(2, 10),  # min- and max no. of modes for surrogate model
    )

    signal = np.linspace(0, 1, num=20).reshape(-1, 1)  # 20 items, 1 parameter

    # a `ModelResult` object with `point_data`, `cell_data` and `field_data`.
    res = model.evaluate(signal)


def test_snapshot_model_list():

    snapshots = np.linspace(0, 1, num=3).reshape(-1, 1)  # 3 snapshots, 1 parameter
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
    field_data = {"id": 1001}  # time-independent data

    model = statescale.SnapshotModel(
        snapshots=snapshots,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )

    signal = np.linspace(0, 1, num=20).reshape(-1, 1)  # 20 items, 1 parameter

    # `point_data`, `cell_data` and `field_data` for step 5 of the signal.
    res_5 = model.evaluate(signal)[5]

    res_5_mean = res_5.apply(np.mean, on_point_data=False, on_cell_data=True)(axis=0)


if __name__ == "__main__":
    test_snapshot_model()
    test_snapshot_model_list()
