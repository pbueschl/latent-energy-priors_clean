#!/usr/bin/env python
"""Wrapper for MedShape split creation."""

from _repo import bootstrap

bootstrap()

from lep.medshape import split_main


if __name__ == "__main__":
    split_main()
