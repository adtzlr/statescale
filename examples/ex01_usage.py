r"""
FEM result interpolation
------------------------
First, let's create and run a simple FEM model with FElupe. A record-function collects
the point- and cell-data dicts at the snapshots.
"""

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

# %%
# Then, use the lists of point- and cell-data at the snapshots to create a
# :class:`~statescale.SnapshotModel`.
import statescale

model = statescale.SnapshotModel(
    snapshots=snapshots,
    point_data=point_data,
    cell_data=cell_data,
    # use_surrogate=True,  # use a surrogate model for interpolation
    # modes = (2, 10),  # choose min-max mode-range for surrogate model
    # threshold = 0.999,  # ratio of included singular values for surrogate model
)

# %%
# A signal will be used to interpolate (evaluate) the point and cell data. The result
# can be converted to a list and supports iteration.
signal = fem.math.linsteps([0, 1], num=500)

out = model.evaluate(signal)
data = out[-5]

# %%
# The results are used to plot the deformed FEM model along with a chosen cell-data.
# Basic math, like transpose, can be applied to the model result. Any custom math-
# function can also be applied on the arrays of the dicts by
# :meth:`~statescale.ModelResult.apply`.
import numpy as np

field[0].values[:] = data.point_data["u"]
view = field.view(
    point_data=data.point_data,
    cell_data=data.apply(np.mean, on_point_data=False)(axis=-2).T.cell_data,
)
view.plot("E", component=0).show()

data = data.apply(np.mean, on_point_data=False)(axis=-2)
data = data.apply(np.transpose, on_point_data=False)()
mesh = data.as_mesh(mesh=mesh)