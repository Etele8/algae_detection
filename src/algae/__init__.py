"""Unsupervised phytoplankton detection & morphotype clustering.

Pipeline stages (each is a module + a script under ``scripts/``):

    manifest -> detect -> embed -> cluster -> review -> classify

The goal is a human-in-the-loop classifier: the model finds every organism
and groups look-alikes; a researcher labels *groups* instead of individual
cells, and those labels propagate to the full dataset.
"""

__version__ = "0.1.0"
