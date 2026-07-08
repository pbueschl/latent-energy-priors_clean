#!/usr/bin/env python
"""Wrapper for `python -m lep.shapenet`."""

from _repo import bootstrap

bootstrap()

from lep.shapenet import main


if __name__ == "__main__":
    main()
