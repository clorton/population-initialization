"""main script for agent initialization"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

import abm_initialization_tools.kmcurve as kmcurve
import abm_initialization_tools.priqueue as priqueue
import abm_initialization_tools.pyramid as pyramid


def main(args):
    print("Hello World!")

    dobs = np.empty(args.count, dtype=np.int32)  # dates of birth
    dods = np.empty(args.count, dtype=np.int32)  # dates of death

    prng = np.random.default_rng(args.seed)

    # initialize the agents' dates of birth
    popdata = pyramid.load_pyramid_csv(args.pyramid)
    agedist = pyramid.AliasedDistribution(
        popdata[:, 4], prng=prng
    )  # ignore sex for now
    indices = agedist.sample(args.count)
    minage = popdata[:, 0] * 365  # closed interval (include this value)
    limage = (popdata[:, 1] + 1) * 365  # open interval (do not include this value)
    print("Converting age-bin indices to dates of birth...")
    for i in tqdm(range(len(popdata))):
        mask = indices == i
        dobs[mask] = prng.integers(low=minage[i], high=limage[i], size=mask.sum())
    print("Converting dates of birth to dates of death...")
    for i in tqdm(range(len(dobs))):
        dods[i] = kmcurve.predicted_day_of_death(dobs[i])
    dods -= dobs  # renormalize to be relative to _now_ (t=0)
    dobs = -dobs  # all _living_ agents have dates of birth before now (t=0)

    # populate the priority queue with agents by their dates of death
    pq = priqueue.PriorityQueue(args.count)
    print("Populating the priority queue with agents by their dates of death...")
    for i in tqdm(range(len(dods))):
        pq.push(
            i, dods[i]
        )  # payload is the agent's index, priority is the date of death

    return


if __name__ == "__main__":
    parser = ArgumentParser(description="main script for agent initialization")
    script_dir = Path(__file__).parent.absolute()
    parser.add_argument(
        "--pyramid",
        type=Path,
        default=script_dir / "data" / "USA-2023.csv",
        help="path to the population pyramid data file",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=20240607,
        help="seed for the random number generator",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=1_000_000,
        help="number of agents to instantiate and initialize",
    )
    args = parser.parse_args()
    main(args)
