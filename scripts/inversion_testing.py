#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import exists
import random
import pickle
import tskit
import msprime
import pyslim
import numpy as np

sys.path.append("..")
import argbd.cladedurations
import argbd.simulations

filename = str(sys.argv[1])  # .trees file
inversion = int(sys.argv[2])  # 0 if no inversion, 1 if with inversion

relate_loc = "./relate-devel"
output_loc = (
    "./simulations/branch-durations/slim_testing/results/"
)
trees_loc = (
    "./simulations/branch-durations/slim_testing/trees/"
)

ts = tskit.load(trees_loc + filename + ".trees")

Ne = 10000  # diploids
mutation_rate = 1e-8
recombination_rate = 1e-8

demography = msprime.Demography.from_tree_sequence(ts)
for pop in demography.populations:
    pop.initial_size = Ne
recap = pyslim.recapitate(
    ts, 
    demography=demography,
    recombination_rate=recombination_rate, 
    random_seed=1,
)

seed = random.randrange(100001)
print("Random seed:", seed)
rng = np.random.default_rng(seed = seed)
alive_inds = pyslim.individuals_alive_at(recap, 0)
keep_indivs = rng.choice(alive_inds, 50, replace=False)
keep_nodes = []
for i in keep_indivs:
    keep_nodes.extend(recap.individual(i).nodes)

recap = recap.simplify(keep_nodes, keep_input_roots=False)
if inversion and recap.num_mutations != 2:
    sys.exit("No inversions here!")
else:
    print("Inversion is above nodes:")
    for m in recap.mutations():
        print(m.node)
    print("="*10)
# WARNING: need keep=False (otherwise will have an issue with running Relate later)
recap = msprime.sim_mutations(recap, rate=mutation_rate, random_seed=11, discrete_genome=True, keep=False)
sites_to_delete = [s.id for s in recap.sites() if len(s.mutations) > 1]
print("Deleting homoplasic sites:", len(sites_to_delete), flush=True)
recap = recap.delete_sites(sites_to_delete)
recap.dump(trees_loc + filename + "_recap.trees")
print("Number of mutations:", recap.num_mutations)

recombination_map = msprime.RateMap(
    position=[0, ts.sequence_length],
    rate=[recombination_rate],
)

argbd.simulations.flat_recombination_map(recombination_rate, recap.sequence_length)

argbd.simulations.run_relate(
    recap,
    rec_map_file="dummy_map.txt",
    mutation_rate=mutation_rate,
    Ne=Ne*2,
    relate_loc=relate_loc,
    quiet=False,
)

os.system(relate_loc + '/bin/RelateFileFormats --mode ConvertToTreeSequence -i relate -o slim_relate')
os.system("mv slim_relate.trees " + trees_loc + filename + "_relate.trees")
slim_relate = tskit.load(trees_loc + filename + "_relate.trees")

results_slimrelate = argbd.cladedurations.clade_duration(slim_relate, Ne, polytomies = True)
results_slimrelate.merge_clades(recombination_map, cM_limit=0.01)
results_slimrelate.calculate_q(recombination_map, use_muts=True, muts_per_kb=0.05)
with open(output_loc + filename + "_relate.results.pickle", "wb") as file:
    pickle.dump(results_slimrelate, file)

results_slim = argbd.cladedurations.clade_duration(recap, Ne, polytomies = True)
results_slim.calculate_q(recombination_map, use_muts=False)
with open(output_loc + filename + "_sim.results.pickle", "wb") as file:
    pickle.dump(results_slim, file)











