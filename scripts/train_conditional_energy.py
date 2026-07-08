#!/usr/bin/env python3
"""Wrapper for ``python -m lep.train_conditional_energy``."""

from _repo import bootstrap

bootstrap()

from lep.train_conditional_energy import main


if __name__ == "__main__":
    main()
