# ISS profile collection

The numbered files in `startup/` run in a shared IPython namespace and configure
the ISS beamline, including EPICS devices, data services, plans, and the GUI.

Use the root Pixi workspace on a beamline workstation:

```sh
pixi run -e terminal start
```

`xas` and `isstools` intentionally use editable checkouts under
`/nsls2/data/iss/shared/config/repos/`. Keep these checkouts available when
installing or updating either the terminal or QueueServer environment.

To refresh the shared profile lockfile on a workstation with those checkouts:

```sh
pixi update
```

The manifest permits patch updates for Bluesky, the Tiled plugins, and nslsii,
and updates within Tiled 0.2. NumPy 2 and pandas 2 are shared profile dependencies
for both terminal and QueueServer. Review the resolved changes and validate
beamline startup before deploying an updated environment. The root lockfile has
not been refreshed by this refactor because the editable checkouts were absent
in the development workspace.

Run the hardware-independent tests anywhere using their separately locked
workspace:

```sh
pixi run --manifest-path tests/pixi.toml --locked test
```

This workspace uses Python 3.12–3.13, NumPy 2, pandas 2, and pytest 8–9 without
resolving the beamline packages. Tests cover bulk sample position insertion,
serialization, coordinate and exposure updates with pandas copy-on-write, and
scan deduplication and persistence. They extract the relevant class definitions
from the startup files without executing device initialization.

Bulk sample insertion now concatenates once per batch. A local comparison of
5,000 positions took 12.98 seconds before and 0.031 seconds after (median of
three runs, Python 3.13.16 / pandas 2.3.3). Data and dtypes matched apart from
generated point IDs. This measures sample construction, not scan acquisition.

The older scripts directly in `tests/` require live data services and a loaded
profile; they are not standalone unit tests. The existing Azure jobs also use
historical Python 3.9 beamline templates, so they do not validate the current
Pixi environment. Hardware startup and acquisition still need beamline testing.
