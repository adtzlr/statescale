# snapsy
snapsy is a Python package for snapshot-based interpolation of high-dimensional
simulation data along arbitrary parameter or signal paths. 

[![codecov](https://codecov.io/gh/adtzlr/snapsy/graph/badge.svg?token=doCYWavVbp)](https://codecov.io/gh/adtzlr/snapsy)
[![readthedocs](https://app.readthedocs.org/projects/snapsy/badge/?version=latest&style=flat)](https://snapsy.readthedocs.io/en/latest/)

## Overview
`snapsy` provides a lightweight API to manage time-dependent point- and cell-data
and time-independent field-data across snapshots and to interpolate that data on new
signals. The central class is `SnapshotModel`.

<p align="center">
  <img width="371" height="256" alt="Image" src="https://github.com/user-attachments/assets/0ab590ab-4c91-4093-a36c-22662a604401" />
</p>

## Installation
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

## Quickstart
A minimal example:

```python
import numpy as np
import snapsy

snapshots = np.linspace(0, 1, num=5).reshape(-1, 1)
point_data = {"u": np.random.rand(5, 9, 3)}  # 5 snapshots, 9 points, 3 dim
cell_data = {"E": np.random.rand(5, 4, 6)}  # 5 snapshots, 4 cells, 6 dim
field_data = {"id": 1001}  # time-independent data

model = snapsy.SnapshotModel(
    snapshots=snapshots,
    point_data=point_data,
    cell_data=cell_data,
    field_data=field_data,
)

signal = np.linspace(0, 1, num=20).reshape(-1, 1)

# `res` is a `ModelResult` object with `point_data`, `cell_data` and `field_data`.
res = model.evaluate(signal)

```

## Tests
The tests are located in `tests`. Run them locally with:

```
tox
```

## Contributing
Bug reports and pull requests are welcome. Please open an issue or PR in
the repository. The package is in early development, expect breaking API changes until
version 1.0.0.

## License
See the `LICENSE` file.
