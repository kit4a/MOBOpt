#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

import mobopt as mo
import deap.benchmarks as db
import argparse

SEED = 1234


def target(x):
    return np.asarray(db.schaffer_mo(x))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="NI", type=int, metavar="NI",
                        help="Number of iterations of the method",
                        required=True)
    parser.add_argument("-r", dest="Prob", type=float, default=0.1,
                        help="Probability of random jumps",
                        required=False)
    parser.add_argument("-q", dest="Q", type=float, default=1.0,
                        help="Weight in factor",
                        required=False)
    parser.add_argument("-ni", dest="NInit", type=int, metavar="NInit",
                        help="Number of initialization points",
                        required=False, default=5)
    parser.add_argument("-v", dest="verbose", action='store_true',
                        help="Verbose")
    parser.add_argument("--outdir", dest="outdir", type=str,
                        default="schaffer/",
                        help="Outpu directory for saving data")
    parser.add_argument("--rprob", dest="Reduce", action="store_true",
                        help="If present reduces prob linearly" +
                        " along simmulation")
    parser.set_defaults(Reduce=False)

    args = parser.parse_args()

    NParam = 1 # nb of decision vars
    NIter = args.NI
    if 0 <= args.Prob <= 1.0:
        Prob = args.Prob
    else:
        raise ValueError("Prob must be between 0 and 1")
    N_init = args.NInit
    verbose = args.verbose
    Q = args.Q

    PB = np.asarray([[-1000.0, 1000.0]]*NParam) # search space dim

    Optimize = mo.MOBayesianOpt(target=target,
                                NObj=2,
                                pbounds=PB,
                                verbose=verbose,
                                outdir=args.outdir,
                                max_or_min='min',
                                RandomSeed=SEED)

    Optimize.initialize(init_points=N_init) # launch N_init evaluations at random points

    front, pop = Optimize.maximize(n_iter=NIter,
                                   prob=Prob,
                                   q=Q,
                                   ReduceProb=args.Reduce)

    PF = np.asarray([np.asarray(y) for y in Optimize.y_Pareto])
    PS = np.asarray([np.asarray(x) for x in Optimize.x_Pareto])


    fig, ax = pl.subplots(1, 1)
    ax.scatter(front[:, 0], front[:, 1], label=r"$\chi$")
    ax.scatter(-PF[:, 0], -PF[:, 1], label="F", color='red')
    ax.grid()
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.legend()
    fig.savefig(args.outdir+f"final_{SEED}.png", dpi=300)


if __name__ == '__main__':
    main()
