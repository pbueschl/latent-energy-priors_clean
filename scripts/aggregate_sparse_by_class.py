#!/usr/bin/env python
"""Wrapper for sparse-sweep class-wise aggregation."""

from _repo import bootstrap

bootstrap()

from lep.medshape import aggregate_main


if __name__ == "__main__":
    aggregate_main()
