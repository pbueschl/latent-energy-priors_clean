#!/usr/bin/env python3
"""Wrapper for ``python -m lep.latent_gmm_prior``."""

from _repo import bootstrap

bootstrap()

from lep.latent_gmm_prior import main


if __name__ == "__main__":
    main()
