#!/usr/bin/env python
"""Wrapper for `python -m lep.bootstrap`."""

from _repo import bootstrap

bootstrap()

from lep.bootstrap import main


if __name__ == "__main__":
    main()
