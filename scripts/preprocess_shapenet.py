#!/usr/bin/env python
"""Wrapper for `python -m lep.preprocess`."""

from _repo import bootstrap

bootstrap()

from lep.preprocess import main


if __name__ == "__main__":
    main()
