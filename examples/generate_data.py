import argparse
import itertools
from multiprocessing import Pool
import numpy as np

import DVG_CEO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes to use",
    )
    args = parser.parse_args()

    desired_data_points = 1000
    molecules = ["lih", "h4", "beh2", "h6"]
    num_r = desired_data_points // len(molecules)
    r_points = np.linspace(0.5, 5, num_r)

    with Pool(args.num_processes) as p:
        results = p.starmap(DVG_CEO.main, itertools.product(molecules, r_points))
