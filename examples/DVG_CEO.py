# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:47:40 2022

@author: mafal
"""

import contextlib
from pathlib import Path

from adaptvqe import molecules
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import DVG_CEO


def main(mol_name: str, r: float):
    mol_factory = getattr(molecules, "create_" + mol_name)
    assert mol_factory is not None, f"No method found to make molecule {mol_name}"
    assert callable(mol_factory)

    logfile = Path("logs") / f"{mol_name}-{r:.3f}.txt"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    # Redirect all the print statements to our log file
    with open(logfile, "w") as f:
        with contextlib.redirect_stdout(f):
            molecule = mol_factory(r)
            pool = DVG_CEO(molecule)

            my_adapt = LinAlgAdapt(
                pool=pool,
                molecule=molecule,
                max_adapt_iter=1000,
                recycle_hessian=True,
                tetris=True,
                verbose=True,
                threshold=1e-3,
                convergence_criterion="max_g",
            )
            my_adapt.run()
            data = my_adapt.data

            # Access the final ansatz indices and coefficients
            print("Evolution of ansatz indices: ", data.evolution.indices)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mol", choices=["h4", "h6", "lih", "beh2"], required=True)
    parser.add_argument("--r", type=float, required=True)
    args = parser.parse_args()
    main(args.mol, args.r)
