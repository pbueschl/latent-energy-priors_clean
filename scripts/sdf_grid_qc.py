#!/usr/bin/env python
"""Wrapper for `python -m lep.sdf_grid_qc`."""

from _repo import bootstrap

bootstrap()

from lep.sdf_grid_qc import main


if __name__ == "__main__":
    main()
