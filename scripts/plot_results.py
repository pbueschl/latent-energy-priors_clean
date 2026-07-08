#!/usr/bin/env python
"""Wrapper for `python -m lep.plot_results`."""

from _repo import bootstrap

bootstrap()

from lep.plot_results import main


if __name__ == "__main__":
    main()
