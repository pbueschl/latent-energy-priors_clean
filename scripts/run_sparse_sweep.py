#!/usr/bin/env python
"""Wrapper for `python -m lep.sparse_sweep`."""

from _repo import bootstrap

bootstrap()

from lep.sparse_sweep import main


if __name__ == "__main__":
    main()
