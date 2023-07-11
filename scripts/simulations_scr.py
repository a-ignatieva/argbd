#!/usr/bin/env python
# coding: utf-8

import stdpopsim
import sys
import os
import random
import tsinfer
import tsdate

sys.path.append("..")
import argbd.simulations

chromosome = sys.argv[1]  # e.g. chr21
sample_size = int(sys.argv[2])  # haploids
sim_model = sys.argv[3]  # hudson, smc, or smc_prime
memory = int(sys.argv[4])  # total memory (will be divided by threads)
threads = int(sys.argv[5])  # number of threads for Relate

relate_loc = "./relate-devel"
relate_lib_loc = "./relate_lib-devel"
simulation_loc = "./simulations/branch-durations/" + chromosome + "/"
resource_loc = "./resource/"
# Recombination maps are obtained from stdpopsim (presuming placed into the resource folder)
# https://stdpopsim.s3.us-west-2.amazonaws.com/genetic_maps/HomSap/HapMapII_GRCh38.tar.gz
rec_map_file = resource_loc + "HapMapII_GRCh38/genetic_map_Hg38_" + chromosome + ".txt"

species = stdpopsim.get_species("HomSap")
if chromosome == "None":
    contig = species.get_contig(length=5e6)
else:
    contig = species.get_contig(chromosome=chromosome, genetic_map="HapMapII_GRCh38")
mutation_rate = contig.mutation_rate
recombination_map = contig.recombination_map
model = stdpopsim.PiecewiseConstantSize(species.population_size)
samples = {"pop_0": int(sample_size / 2)}

Ne = species.population_size  # diploids

print("Simulating:", sample_size, flush=True)
ts, rs = argbd.simulations.simulate_data(
    contig,
    model,
    samples,
    sim_model=sim_model,
)
print(
    ts.num_trees,
    ts.num_samples,
    ts.first().interval,
    ts.last().interval,
    mutation_rate,
    rs,
    flush=True,
)

# Delete homoplasic sites
sites_to_delete = [s.id for s in ts.sites() if len(s.mutations) > 1]
print("Deleting homoplasic sites:", len(sites_to_delete), flush=True)
ts = ts.delete_sites(sites_to_delete)
ts.dump(
    simulation_loc + "simulated_data_" + sim_model + "_" + str(sample_size) + ".trees"
)

if sim_model == "smc_prime":
    if chromosome == "None":
        with open("dummy_map.txt", "w") as file:
            file.write("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n")
            file.write(
                "0 "
                + str(contig.recombination_map.mean_rate * ts.sequence_length / 0.01)
                + " 0.0\n"
            )
            file.write(str(int(ts.sequence_length)) + " 0.0 1.0\n")
        rec_map_file = "dummy_map.txt"

    print("Running Relate", flush=True)
    argbd.simulations.run_relate(
        ts,
        rec_map_file,
        mutation_rate,
        2 * Ne,
        relate_loc,
        memory=int(memory / threads),
        consistency=False,
        postprocess=False,
        randomise=False,
        threads=threads,
        quiet=False,
    )
    os.system(
        "mv relate.anc "
        + simulation_loc
        + "relate_"
        + sim_model
        + "_"
        + str(sample_size)
        + ".anc"
    )
    os.system(
        "mv relate.mut "
        + simulation_loc
        + "relate_"
        + sim_model
        + "_"
        + str(sample_size)
        + ".mut"
    )

    print("Running final cleanup", flush=True)
    argbd.simulations.clean_relate()
    print("Done", flush=True)

    ts_tsdate = argbd.simulations.run_tsinfer(ts, Ne, contig)
    ts_tsdate.dump(
        simulation_loc + "tsdate_" + sim_model + "_" + str(sample_size) + ".trees"
    )

    # demo file for argneedle, subsequently invoking argneedle reconstruction
    demo_loc = resource_loc + "const10k.demo"
    ts_argneedle = argbd.simulations.run_argneedle(
        ts,
        recombination_map,
        mutation_rate,
        chromosome,
        demo_loc,
        50,
        sample_size,
        sim_model,
        resource_loc,
        simulation_loc,
        "argneedle",
    )
