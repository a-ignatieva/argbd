import numpy as np
import scipy
import sys
from tqdm import tqdm
from collections import defaultdict
import math

from . import branchdurations


class Clade:
    def __init__(
        self,
        binid,
        treeindex,
        tbl,
        sampleset,
        probability,
        start,
        sequence_length,
        depth,
    ):
        self.id = None
        self.binid = binid
        self.treeindex = treeindex
        self.sampleset = sampleset
        self.cladesize = len(sampleset)
        self.probability = probability
        self.sf = None
        self.q = None
        self.tbl = tbl
        self.sequence_length = sequence_length
        self.start = start
        self.end = None
        self.left_mut = math.inf
        self.right_mut = -1
        self.num_mutations = 0
        self.duration = None
        self.mut_duration = None
        self.rate = None
        self.depth = depth
        self.mutations = set()
        self.merged = 0

    def set_duration(self, end):
        self.end = end
        self.duration = (end - self.start) / self.sequence_length
        if not (
            (self.left_mut == math.inf or self.right_mut == -1)
            or (self.left_mut == self.right_mut)
        ):
            self.mut_duration = (self.right_mut - self.left_mut) / self.sequence_length

    def add_mutations(self, mut):
        if len(mut) > 0:
            self.left_mut = min(self.left_mut, min(mut))
            self.right_mut = max(self.right_mut, max(mut))
            self.num_mutations += len(mut)
            self.mutations.update(mut)


class Clades:
    def __init__(self, ts):
        self.name = ""
        self.num = 0
        self.num_ = 0
        self.clades = [None] * (ts.num_trees * (ts.num_samples - 2))
        self.depth = None
        self.cladesize = None
        self.active_clades = {}
        self.q = None
        self.logq = None
        self.sf = None
        self.logsf = None
        self.qsorted = None
        self.sfsorted = None
        self.offset = 0
        self.ids = None

    def add_clade(self, clade):
        clade.id = self.num
        self.clades[self.num] = clade
        self.active_clades[clade.binid] = self.num  # new clade is active
        self.num = self.num + 1

    def print_info(self):
        print(self.num, self.active_clades)

    def print_clades(self):
        for clade in self.clades[0 : self.num]:
            print(
                "Tree:",
                clade.treeindex,
                "sample set:",
                clade.sampleset,
                "P:",
                clade.probability,
                "start:",
                clade.start,
                "end:",
                clade.end,
                "duration:",
                clade.duration,
            )

    def close(self, end):
        for key, value in self.active_clades.items():
            clade = self.clades[value]
            clade.set_duration(end)
        self.clades = self.clades[0 : self.num]
        self.active_clades = {}

    def merge_clades(self, rec_map, cM_limit=0.1):
        # Make a list of clade samples : clade id
        # (This will be in left-to-right order because of the way we insert clades)
        bin_dict = defaultdict(list)
        bin_counts = {}
        for clade in self.clades:
            if clade is not None:
                if clade.binid in bin_counts:
                    # Find position where clade last disappeared
                    cl = self.clades[
                        max(bin_dict[(clade.binid, bin_counts[clade.binid])])
                    ]
                    # Get genetic distance
                    d = (
                        rec_map.get_cumulative_mass(clade.start)
                        - rec_map.get_cumulative_mass(cl.end)
                    ) * 100
                    if d < cM_limit:
                        bin_dict[(clade.binid, bin_counts[clade.binid])].append(
                            clade.id
                        )
                    else:
                        bin_counts[clade.binid] += 1
                        bin_dict[(clade.binid, bin_counts[clade.binid])].append(
                            clade.id
                        )
                else:
                    bin_dict[(clade.binid, 0)].append(clade.id)
                    bin_counts[clade.binid] = 0

        # Merge clades where possible
        num_removed = 0
        for _, clade_list in bin_dict.items():
            if len(clade_list) > 1:
                clade_to_extend = self.clades[clade_list[0]]
                for i, clade_id in enumerate(clade_list):
                    if i > 0:
                        clade = self.clades[clade_id]
                        clade_to_extend.end = clade.end
                        clade_to_extend.num_mutations += clade.num_mutations
                        clade_to_extend.right_mut = max(
                            clade_to_extend.right_mut, clade.right_mut
                        )
                        self.clades[clade_id] = None
                        num_removed += 1
                        clade_to_extend.merged += 1
                clade_to_extend.duration = (
                    clade_to_extend.end - clade_to_extend.start
                ) / clade_to_extend.sequence_length
                if not (
                    (
                        clade_to_extend.left_mut == math.inf
                        or clade_to_extend.right_mut == -1
                    )
                    or (clade_to_extend.left_mut == clade_to_extend.right_mut)
                ):
                    clade_to_extend.mut_duration = (
                        clade_to_extend.right_mut - clade_to_extend.left_mut
                    )
        print("Merged", num_removed, "clades")

    def calculate_q(
        self,
        rec_map,
        depth=None,
        supported=False,
        argn_trees=False,
        argn_ts=None,
        use_muts=False,
        muts_per_kb=0,
    ):
        if argn_trees and argn_ts is None:
            sys.exit("Need ARG-Needle tree sequence")
        Q = np.zeros(self.num, dtype=float)
        logQ = np.zeros(self.num, dtype=float)
        SF = np.zeros(self.num, dtype=float)
        logSF = np.zeros(self.num, dtype=float)
        ids = []
        cladesizes = np.zeros(self.num, dtype=int)
        depths = np.zeros(self.num, dtype=int)
        offset = 0
        if argn_trees:
            offset = int(argn_ts.metadata["offset"]) - 1
            self.offset = offset
        i = 0
        for j, clade in enumerate(self.clades):
            if clade is not None:
                clade.sf = None
                clade.q = None
                if use_muts:
                    d = clade.mut_duration
                    start = clade.left_mut
                    end = clade.right_mut
                else:
                    d = clade.duration
                    start = clade.start
                    end = clade.end
                if d is not None and start is not None and end is not None:
                    if (
                        1000
                        * clade.num_mutations
                        / (clade.duration * clade.sequence_length)
                        >= muts_per_kb
                    ):
                        if (not supported) or (supported and clade.num_mutations > 0):
                            if (depth is not None and clade.cladesize == depth) or (
                                depth is None
                            ):
                                P = clade.probability
                                Left = max(start + offset, rec_map.left[0])
                                Right = min(end + offset, rec_map.right[-1])
                                R = rec_map.get_cumulative_mass(
                                    Right
                                ) - rec_map.get_cumulative_mass(Left)
                                cladesizes[i] = clade.cladesize
                                depths[i] = clade.depth
                                logQ[i] = scipy.stats.expon.logcdf(
                                    P * clade.tbl * R, loc=0, scale=1
                                )
                                Q[i] = scipy.stats.expon.cdf(
                                    P * clade.tbl * R, loc=0, scale=1
                                )
                                logSF[i] = scipy.stats.expon.logsf(
                                    P * clade.tbl * R, loc=0, scale=1
                                )
                                SF[i] = scipy.stats.expon.sf(
                                    P * clade.tbl * R, loc=0, scale=1
                                )
                                ids.append(j)
                                clade.sf = SF[i]
                                clade.q = Q[i]
                            i += 1
        self.depth = depths
        self.num_ = i
        self.cladesize = cladesizes
        self.q = Q[0:i]
        self.logq = logQ[0:i]
        self.sf = SF[0:i]
        self.logsf = logSF[0:i]
        self.qsorted = np.sort(Q[0:i])
        self.sfsorted = np.sort(SF[0:i])
        self.ids = ids


def clade_binid(sampleset):
    """
    Compute the binary representation of a clade (1 if sample is in clade, 0 otherwise)
    :param sampleset: list of sample ids
    :return:
    """
    binid = 0
    for i in sampleset:
        binid = binid | 1 << i
    return binid


def ltt_g(g, t, tree, verbose=False):
    """
    Lineages through time that do and do not belong to clade under branch G
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param verbose: print info
    :return:
    """
    num_G_lineages = np.zeros(len(tree.num_lineages), dtype=int)
    num_Gg_lineages = np.zeros(len(tree.num_lineages), dtype=int)
    count = 0
    gup = None
    Gup = None
    for i, n in enumerate(t.nodes(order="timedesc")):
        if verbose:
            print("i =", i, "node = ", n)
        if t.time(n) == 0:
            if verbose:
                print("reached samples: recording", count, "and stopping")
            num_G_lineages[i] = count
            if not Gup:
                Gup = i + 1
            break
        elif t.is_descendant(n, g):
            num_G_lineages[i] = count
            if count == 0:
                count = 1
                Gup = i + 1
            if verbose:
                print("descendant, recording", count)
            count += 1
        elif n == t.parent(g):
            gup = i + 1
        else:
            if verbose:
                print("not a descenant, recording", count)
            num_G_lineages[i] = count
    for i in range(len(num_G_lineages)):
        if gup <= i < Gup:
            num_Gg_lineages[i] = 1
        else:
            num_Gg_lineages[i] = num_G_lineages[i]
    if verbose:
        print(
            "num_G_lineages:",
            num_G_lineages,
            "num_Gg_lineages:",
            num_Gg_lineages,
            "gup:",
            gup,
            "Gup:",
            Gup,
        )
    if len(num_G_lineages) != t.num_samples():
        sys.exit("Error: len(num_G_lineages) wrong in tree " + str(t.index))
    return num_G_lineages, num_Gg_lineages, Gup, gup


def Ls_g(g, t, tree):
    """
    Total branch length of clade below g
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    return sum(t.branch_length(u) for u in t.nodes(root=g) if u != g) / (2 * tree.Ne)


def G_term2(k, lineages, up, tree, verbose=False):
    N = 2 * tree.Ne
    Tk = tree.tim_lineages[k - 2] / N
    Tk1 = tree.tim_lineages[k - 1] / N
    p = 0
    for j in range(1 + up, k + 1):
        nj1 = lineages[j - 1]
        p += nj1 * tree.lookup_Qkj(k, j)
        if verbose:
            print(
                "Term 2: summing", j, k, ", number of lineages = ", nj1, "up = ", up, p
            )
    p = p * 1 / k * (np.exp(k * Tk) - np.exp(k * Tk1))
    return p


def prob_CnotinGg_RinG(
    g, t, tree, num_G_lineages, num_Gg_lineages, Gup, gup, verbose=False
):
    """
    P(C not in G u g | R in G)
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param num_G_lineages: lineages through time for branches in clade G
    :param num_Gg_lineages: lineages through time for branches in clade G including branch above g
    :param Gup: number of lineages at time of clade MRCA
    :param gup: number of lineages at upper end of g branch
    :param verbose: print info
    :return:
    """
    if t.is_sample(g):
        return 1.0
    else:
        down = t.num_samples()
        P = 0
        for k in range(Gup + 1, down + 1):
            # print("k = ", k)
            nGTk1 = num_G_lineages[k - 1]
            nGgTk1 = num_Gg_lineages[k - 1]
            # print("nG = ", nGTk1, "nGg = ", nGgTk1)
            P += nGTk1 * (
                nGgTk1 * branchdurations.Q1(k, tree)
                + G_term2(k, num_Gg_lineages, gup, tree, verbose)
            )
        P = 1 - P / Ls_g(g, t, tree)
        return P


def prob_CinG_RnotinGg(g, t, tree, num_G_lineages, num_Gg_lineages, Gup, verbose=False):
    """
    P(C in G | R not in G u g)
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param num_G_lineages: lineages through time for branches in clade G
    :param num_Gg_lineages: lineages through time for branches in clade G including branch above g
    :param Gup: number of lineages at time of clade MRCA
    :param verbose: print info
    :return:
    """
    if tree.tbl <= Ls_g(g, t, tree):
        return 1.0
    else:
        tg = (t.time(t.parent(g)) - t.time(g)) / (2 * tree.Ne)
        down = t.num_samples()
        P = 0
        for k in range(Gup + 1, down + 1):
            # print("k = ", k)
            nGTk1 = num_G_lineages[k - 1]
            nGgTk1 = num_Gg_lineages[k - 1]
            # print("nG = ", nGTk1, "nGg = ", nGgTk1)
            P += (k - nGgTk1) * (
                nGTk1 * branchdurations.Q1(k, tree)
                + G_term2(k, num_G_lineages, Gup, tree, verbose)
            )
        P = P / (tree.tbl - Ls_g(g, t, tree) - tg)
        return P


def prob_RinG(g, t, tree):
    """
    Probability of recombination event in clade G
    :param g: branch id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    return Ls_g(g, t, tree) / tree.tbl


def prob_RnotinGg(g, t, tree):
    """
    Probability of recombination event not in clade G or on branch above g
    :param g: branch id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    tg = (t.time(t.parent(g)) - t.time(g)) / (2 * tree.Ne)
    return (tree.tbl - Ls_g(g, t, tree) - tg) / tree.tbl


def prob_Ring(g, t, tree):
    """
    Probability of recombination event on branch above g
    :param g: branch id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    tg = (t.time(t.parent(g)) - t.time(g)) / (2 * tree.Ne)
    return tg / tree.tbl


def prob_g_disrupted(g, t, tree, test=False, verbose=False):
    """
    Probability clade under g is disrupted by the recombination event
    :param g: branch id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param test: whether to check that all calculated probabilities sum to 1
    :param verbose: print info
    :return:
    """
    num_G_lineages, num_Gg_lineages, Gup, gup = ltt_g(g, t, tree)
    P1 = prob_RinG(g, t, tree)
    P2 = prob_RnotinGg(g, t, tree)
    P3 = prob_CnotinGg_RinG(
        g, t, tree, num_G_lineages, num_Gg_lineages, Gup, gup, verbose
    )
    P4 = prob_CinG_RnotinGg(g, t, tree, num_G_lineages, num_Gg_lineages, Gup, verbose)
    if verbose:
        print(P1, P2, P3, P4)
    if test:
        if abs(prob_g_test(g, t, tree) - 1) > 1e-5:
            sys.exit("Error: test not passed with threshold 1e-5")
        else:
            print("Test passed")
    return P1 * P3 + P2 * P4


def prob_g_test(g, t, tree):
    """
    Testing that all the probabilities we calculate sum to 1
    :param g: branch id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    num_G_lineages, num_Gg_lineages, Gup, gup = ltt_g(g, t, tree)
    P0 = prob_Ring(g, t, tree)
    P1 = prob_RinG(g, t, tree)
    P2 = prob_RnotinGg(g, t, tree)
    P3 = prob_CnotinGg_RinG(g, t, tree, num_G_lineages, num_Gg_lineages, Gup, gup)
    P4 = prob_CinG_RnotinGg(g, t, tree, num_G_lineages, num_Gg_lineages, Gup)
    return P1 * P3 + P2 * P4 + P1 * (1 - P3) + P2 * (1 - P4) + P0


def clade_duration(ts, Ne, polytomies=False):
    """
    Compute the duration of clades in a given tree sequence.
    This is defined by the samples subtending a clade staying the same.
    :param ts: tskit tree sequence
    :param Ne: effective population size (diploids)
    :param polytomies: whether ts contains polytomies
    :return:
    """
    trees = branchdurations.compute_trees(ts, Ne, polytomies=polytomies)
    clades = Clades(ts)
    num_trees = ts.num_trees
    print("Computing clade duration")
    with tqdm(total=num_trees) as pbar:
        for t in ts.trees():
            if (
                t.num_roots == 1
            ):  # Sometimes first tree in ts out of stdpopsim is empty so skip it
                tree = trees.trees[t.index]
                prevclades = list(clades.active_clades.values())
                # tree_muts = {n: 0 for n in t.nodes()}
                tree_muts = defaultdict(set)
                for mut in t.mutations():
                    tree_muts[mut.node].add(ts.site(mut.site).position)
                for g in t.nodes():
                    if g != t.root and not t.is_sample(g):
                        m = tree_muts[g]
                        sampleset = set([s for s in t.samples(g)])
                        g_id = clade_binid(sampleset)
                        # Could allow inexact matches (same as Relate say)
                        # Could check a few trees away to see if clade reappears (gene conv style events in homs)
                        if g_id in clades.active_clades:
                            p = clades.active_clades[g_id]
                            clade = clades.clades[p]
                            clade.add_mutations(m)
                            prevclades.remove(clades.active_clades[g_id])
                            # Add number of mutations above g to clade mutation count
                        else:
                            clade = Clade(
                                g_id,
                                t.index,
                                t.total_branch_length,
                                sampleset,
                                prob_g_disrupted(g, t, tree),
                                t.interval[0],
                                ts.sequence_length,
                                t.depth(g),
                            )
                            clade.add_mutations(m)
                            clades.add_clade(clade)
                for p in prevclades:
                    # These clades have disappeared
                    clade = clades.clades[p]
                    clade.set_duration(t.interval[0])
                clades.active_clades = {
                    key: val
                    for key, val in clades.active_clades.items()
                    if val not in prevclades
                }

                if t.index == num_trees - 1:
                    clades.close(t.interval[1])
                    pbar.update(1)
                    break
            pbar.update(1)
    clades.print_info()
    return clades
