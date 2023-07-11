#!/usr/bin/env python
# coding: utf-8

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

chromosome = sys.argv[1]  # e.g. 'chr21' or 'None'
sample_size = int(sys.argv[2])  # haploids
num_sample_branches = int(sys.argv[3])  # number of branches to test
sample_start = int(
    sys.argv[4]
)  # left endpoint of interval of edge ids to select from (for running in parallel)
sample_end = int(
    sys.argv[5]
)  # right endpoint of interval of edge ids to select from (for running in parallel)
sample_index = int(sys.argv[6])  #  chunk number (for running in parallel)

simulation_loc = ("./simulations/branch-durations/" + chromosome + "/")

species = stdpopsim.get_species("HomSap")
if chromosome == "None":
    contig = species.get_contig(length=5e6)
else:
    contig = species.get_contig(chromosome=chromosome, genetic_map="HapMapII_GRCh38")
mutation_rate = contig.mutation_rate
recombination_map = contig.recombination_map
Ne = species.population_size  # diploids

argn_handle = "argneedle_smc_prime_" + str(sample_size)

if exists(simulation_loc + argn_handle + ".trees"):
    argn_trees = tskit.load(simulation_loc + argn_handle + ".trees")
else:
    sys.exit("No ARG-Needle .trees file found")

argn_results = argbd.branchdurations.gof_calc(
    argn_trees,
    recombination_map,
    Ne,
    top_only=False,
    argn_trees=True,
    argn_norm=False,
    num_sample_branches=num_sample_branches,
    # sample_range=(sample_start, sample_end),
)
with open(
    simulation_loc + argn_handle + "_" + str(sample_index) + ".results.pickle",
    "wb",
) as file:
    print("Writing ARG-Needle results to file...")
    pickle.dump(argn_results, file)
del argn_results
