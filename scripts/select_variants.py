#!/usr/bin/env python
"""Wrapper for `python -m lep.select_variants`."""

from _repo import bootstrap

bootstrap()

from lep.select_variants import main


if __name__ == "__main__":
    main()
