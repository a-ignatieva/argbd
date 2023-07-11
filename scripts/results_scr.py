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

chromosome = sys.argv[1]  # e.g. 'chr21' or 'None'
sample_size = int(sys.argv[2])  # haploids
sim_model = sys.argv[3]  # 'hudson', 'smc_prime', 'smc'
# whether to use only topology-disrupting events, 0=False or 1=True
top_only = int(sys.argv[4])
num_sample_branches = int(sys.argv[6])  # number of branches to sample

top_only = top_only == 1

relate_loc = "./relate-devel"
relate_lib_loc = "./relate_lib-devel"
simulation_loc = ("./simulations/branch-durations/" + chromosome + "/")
resource_loc = "./resource/"

species = stdpopsim.get_species("HomSap")
if chromosome == "None":
    contig = species.get_contig(length=5e6)
else:
    contig = species.get_contig(chromosome=chromosome, genetic_map="HapMapII_GRCh38")
mutation_rate = contig.mutation_rate
recombination_map = contig.recombination_map
Ne = species.population_size  # diploids

sim_handle = "simulated_data_" + str(sim_model) + "_" + str(sample_size)
relate_handle = "relate_" + str(sim_model) + "_" + str(sample_size)
tsdate_handle = "tsdate_" + str(sim_model) + "_" + str(sample_size)
argneedle_handle = "argneedle_" + str(sim_model) + "_" + str(sample_size)

if top_only:
    print(sim_handle, top_only)
    sim_trees = tskit.load(simulation_loc + sim_handle + ".trees")
    sim_results = argbd.branchdurations.gof_calc(
        sim_trees,
        recombination_map,
        Ne,
        top_only=False,
        num_sample_branches=num_sample_branches,
    )
    with open(simulation_loc + sim_handle + ".results.pickle", "wb") as file:
        print("Writing results to file...")
        pickle.dump(sim_results, file)
    del sim_results

if exists(simulation_loc + relate_handle + ".anc"):
    if not exists(simulation_loc + relate_handle + ".trees"):
        os.system(
            relate_loc
            + "/bin/RelateFileFormats --mode ConvertToTreeSequence -i "
            + simulation_loc
            + relate_handle
            + " -o "
            + simulation_loc
            + relate_handle
        )
    relate_trees = tskit.load(simulation_loc + relate_handle + ".trees")
    relate_results = argbd.branchdurations.gof_calc(
        relate_trees,
        recombination_map,
        Ne,
        relate_trees=True,
        relate_anc_file=simulation_loc + relate_handle,
        top_only=top_only,
        supported_only=True,
        mutation_rate=mutation_rate,
        num_sample_branches=num_sample_branches,
    )
    with open(
        simulation_loc + relate_handle + "_TO_" + str(top_only) + ".results.pickle",
        "wb",
    ) as file:
        print("Writing Relate results to file...")
        pickle.dump(relate_results, file)
    del relate_results
if exists(simulation_loc + tsdate_handle + ".trees"):
    print(tsdate_handle, top_only)
    tsdate_trees = tskit.load(simulation_loc + tsdate_handle + ".trees")
    tsdate_results = argbd.branchdurations.gof_calc(
        tsdate_trees,
        recombination_map,
        Ne,
        tsdate_trees=True,
        top_only=top_only,
        num_sample_branches=num_sample_branches,
    )
    with open(
        simulation_loc + tsdate_handle + "_TO_" + str(top_only) + ".results.pickle",
        "wb",
    ) as file:
        print("Writing tsdate results to file...")
        pickle.dump(tsdate_results, file)
    del tsdate_results

# Call squashed trees from argneedle and use within computations
if exists(simulation_loc + argneedle_handle + "_squashed.trees"):
    argneedle_trees = tskit.load(simulation_loc + argneedle_handle + "_squashed.trees")
    argneedle_results = argbd.branchdurations.gof_calc(
        argneedle_trees,
        recombination_map,
        Ne,
        top_only=False,
        argn_trees=True,
        argn_norm=False,
        num_sample_branches=num_sample_branches,
    )
    with open(
        simulation_loc + argneedle_handle + "_TO_" + str(top_only) + ".results.pickle",
        "wb",
    ) as file:
        print("Writing argneedle results to file...")
        pickle.dump(argneedle_results, file)
    del argneedle_results
