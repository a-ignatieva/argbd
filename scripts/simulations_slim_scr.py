#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import tskit

invlength = int(sys.argv[1])  # length of inverted region

seed = str(random.randrange(10000000000000000))

slim_loc = "./SLiM/bin/slim"
trees_loc = ("./trees/")

if invlength != 0:
    os.system(slim_loc + " -d invlength=" + str(invlength) + " inversions.slim")
else:
    os.system(slim_loc + " neutral.slim")

trees = tskit.load("slim.trees")
if invlength != 0 and trees.num_mutations != 2:
    print("Inversion not fixed")
else:
    os.system("mv slim.trees " + trees_loc + "slim_" + str(invlength) + "_" + seed + ".trees")












