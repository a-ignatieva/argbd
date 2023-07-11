#!/usr/bin/env python
# coding: utf-8

# Script to concatenate results for running ARG-Needle in parallel

import stdpopsim
import sys
import os
from os.path import exists
import random
import pickle
import tskit
import tsinfer
import tsdate

sys.path.append("..")
import argbd.branchdurations
import argbd.cladedurations

chromosome = sys.argv[1]
sample_size = int(sys.argv[2])
num_chunks = int(sys.argv[3])

simulation_loc = ("./simulations/branch-durations/" + chromosome + "/")

argn_handle = "argneedle_smc_prime_" + str(sample_size)

results = [None] * num_chunks
for i in range(num_chunks):
    results[i] = pickle.load(
        open(
            simulation_loc
            + "argneedle_smc_prime_"
            + str(sample_size)
            + "_"
            + str(i)
            + ".results.pickle",
            "rb",
        )
    )
results_concat = argbd.branchdurations.merge_results(results)
with open(
    simulation_loc + "argneedle_smc_prime_" + str(sample_size) + ".results.pickle", "wb"
) as file:
    pickle.dump(results_concat, file)
