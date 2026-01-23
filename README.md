<p align="center">
  <a href="https://felupe.readthedocs.io/en/latest/?badge=latest"><img src="https://github.com/user-attachments/assets/d1bc153c-b597-4961-839f-ed24de096747" height="160"></a>
  <p align="center"><i>snapsy - Snapshot-based interpolation of simulation data</i></p>
</p>

[![codecov](https://codecov.io/gh/adtzlr/snapsy/graph/badge.svg?token=doCYWavVbp)](https://codecov.io/gh/adtzlr/snapsy)
[![readthedocs](https://app.readthedocs.org/projects/snapsy/badge/?version=latest&style=flat)](https://snapsy.readthedocs.io/en/latest/)

snapsy is a Python package for snapshot-based interpolation of high-dimensional
simulation data along arbitrary parameter or signal paths. 


## ✨ Overview
`snapsy` provides a lightweight API to manage time-dependent point- and cell-data
and time-independent field-data across snapshots and to interpolate that data on new
signals. The central class is `SnapshotModel`.

**Highlights**

- [x] Snapshot-based interpolation
- [x] Apply signals on existing simulation data at snapshots
- [x] Efficient handling of high-dimensional data
- [x] Easy-to-use API

<p align="center">
  <img width="371" height="256" alt="Image" src="https://github.com/user-attachments/assets/0ab590ab-4c91-4093-a36c-22662a604401" />
</p>

## 📦 Installation
Install from PyPI:

```
pip install snapsy
```

Development install (from source):

1. Clone the repository:

```
git clone https://github.com/adtzlr/snapsy.git
cd snapsy
```

2. Install in editable mode:

```
pip install --editable .
```

Dependencies: `numpy` and `scipy` (and `pytest` for running tests).

## 🚀 Quickstart
A minimal example. Snapshots must have shapes `(n_snapshots, n_dim)`, point- and cell-
data `(n_snapshots, ...)` and the dimension of the signal must be compatible with
snapshots, i.e. `(n_steps, n_dim)`. The second dimension of snapshots and the signal
are optional, 1d-arrays are also supported. The model result will be of shape
`(n_steps, ...)`.

### Array-based input data
```python
import numpy as np
import snapsy

snapshots = np.linspace(0, 1, num=3).reshape(-1, 1)  # 3 snapshots, 1 parameter
point_data = {"displacement": np.random.rand(3, 9, 3)}  # 3 snapshots, 9 points, 3 dim
cell_data = {"strain": np.random.rand(3, 4, 6)}  # 3 snapshots, 4 cells, 6 dim
field_data = {"id": 1001}  # time-independent data

model = snapsy.SnapshotModel(
    snapshots=snapshots,
    point_data=point_data,
    cell_data=cell_data,
    field_data=field_data,
    # use_surrogate=False,  # use a POD surrogate model
    # modes=(2, 10),  # min- and max no. of modes for surrogate model
)

signal = np.linspace(0, 1, num=20).reshape(-1, 1)  # 20 items, 1 parameter

# a `ModelResult` object with `point_data`, `cell_data` and `field_data`.
res = model.evaluate(signal)
```

### List-based input data
If the data is list-based, the model can also import lists of dicts, with per-snapshot
list items. Model results also support indexing and a conversion to lists of dicts.

```python
import numpy as np
import snapsy

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

model = snapsy.SnapshotModel(
    snapshots=snapshots,
    point_data=point_data,
    cell_data=cell_data,
    field_data=field_data,
)

# `point_data`, `cell_data` and `field_data` for step 5 of the signal.
res_5 = model.evaluate(signal)[5]
```

Any NumPy-function may be applied to the model result data on all time-dependent arrays.
E.g., the mean over all cells (here, the first axis) of the cell-data is evaluated by:

```python
res_5_mean = res_5.apply(np.mean, on_point_data=False, on_cell_data=True)(axis=0)
```

More details can be found in the [documentation](https://snapsy.readthedocs.io/).

## 🛠️ Tests
The tests are located in `tests`. Run them locally with:

```
tox
```

## ➕ Contributing
Bug reports and pull requests are welcome. Please open an issue or PR in
the repository. The package is in early development, expect breaking API changes until
version 1.0.0.

## 📄 Changelog
All notable changes to this project will be documented in [this file](CHANGELOG.md). The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 🔓 License
See the `LICENSE` file.
