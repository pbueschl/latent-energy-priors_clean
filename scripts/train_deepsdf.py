#!/usr/bin/env python
"""Wrapper for `python -m lep.train_deepsdf`."""

from _repo import bootstrap

bootstrap()

from lep.train_deepsdf import main


if __name__ == "__main__":
    main()
