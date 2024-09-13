#!/usr/bin/env python
import glob, re, os, random
import numpy as np, pandas as pd
from afpdb.afpdb import Protein,ATS
from afpdb import util
from Bio.PDB import PDBParser, Selection, Superimposer, NeighborSearch
from Bio.PDB.Polypeptide import three_to_one
from numpy.testing import assert_array_equal
import time,warnings,traceback,json
from maps import map
warnings.filterwarnings("ignore")

# the server has 64 cores, no other job is running
# we use 50 cores
# you should adjust your cores, so that processes do not compete for CPUs
# you can specify n_CPU=0, which will use (#cores - 2) on your system
n_CPU=50
# number of repeats in each use case
N1=5

def cost(exp_pdb):
    # these does not seem to be enough to make the result exact the same each time
    random.seed(42)
    np.random.seed(42)

    m = re.search(r"(?P<pdb>\w+).pdb$", exp_pdb)
    m = re.search(r"(?P<pdb>\w+).pdb$", exp_pdb)
    pdb_code=m.group('pdb')

    p0 = Protein(exp_pdb)
    # p0: experimental structure
    # p1: AF predicted structure

    # let's always clone objects to prevent potential caching
    try:
        ##################
        ### Use case 1: interface
        ##################

        ats=ATS("ALL")
        rs_a=p0.rs("H:L")
        rs_b=p0.rs("G")
        res_a=[ p0._get_xyz(i, ats)[2] for i in rs_a.data ]
        res_b=[ p0._get_xyz(i, ats)[2] for i in rs_b.data ]

        # EvoPro - score_contacts_pae_weighted in
        # https://github.com/Kuhlman-Lab/evopro/blob/main/evopro/score_funcs/score_funcs.py
        start=time.time()
        for i in range(N1):
            out=[]
            for i,r1 in enumerate(res_a):
                for j,r2 in enumerate(res_a):
                    for i_atom,xyz1 in enumerate(r1):
                        for j_atom,xyz2 in enumerate(r2):
                            dist=(xyz1 - xyz2)
                            dist=np.sqrt(np.sum(dist*dist))
                            if dist<=10: out.append((i, i_atom, j, j_atom))
        dur_evopro=time.time()-start

        # dockQ (non-Cython) - atom_distances_to_residue_distances in
        # code taken from
        # https://github.com/bjornwallner/DockQ/blob/master/src/DockQ/operations_nocy.py
        def get_distances_across_chains(model_A_atoms, model_B_atoms):

            distances = ((model_A_atoms[:, None] - model_B_atoms[None, :]) ** 2).sum(-1)

            return distances


        def atom_distances_to_residue_distances(atom_distances, atoms_per_res1, atoms_per_res2):
            res_distances = np.zeros((len(atoms_per_res1), len(atoms_per_res2)))

            cum_i_atoms = 0
            for i, i_atoms in enumerate(atoms_per_res1):
                cum_j_atoms = 0
                for j, j_atoms in enumerate(atoms_per_res2):
                    res_distances[i, j] = atom_distances[
                        cum_i_atoms : cum_i_atoms + i_atoms, cum_j_atoms : cum_j_atoms + j_atoms
                    ].min()
                    cum_j_atoms += j_atoms
                cum_i_atoms += i_atoms
            return res_distances


        def residue_distances(
            atom_coordinates1, atom_coordinates2, atoms_per_res1, atoms_per_res2
        ):
            atom_distances = get_distances_across_chains(atom_coordinates1, atom_coordinates2)
            res_distances = atom_distances_to_residue_distances(
                atom_distances, atoms_per_res1, atoms_per_res2
            )

            return res_distances

        start=time.time()
        for i in range(N1):
            _, _, atom_a=p0._get_xyz(rs_a, ats)
            _, _, atom_b=p0._get_xyz(rs_b, ats)
            n_atom_per_res_a=np.sum(p0.data.atom_mask[rs_a.data], axis=1).astype(int)
            n_atom_per_res_b=np.sum(p0.data.atom_mask[rs_b.data], axis=1).astype(int)
            dist=residue_distances(atom_a, atom_b, n_atom_per_res_a, n_atom_per_res_b)
            dist=np.sqrt(dist)<=10
        dur_dockQ=time.time()-start

        # afpdb
        start=time.time()
        for i in range(N1):
            rs_g, rs_hl, dist = p0.rs_around("H:L", rs_within="G", dist=10)
        dur_afpdb_dist=time.time()-start

        return {
            "PDB": pdb_code,
            "EvoPro": dur_evopro,
            'DockQ': dur_dockQ,
            'Afpdb': dur_afpdb_dist
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "PDB": pdb_code,
            "EvoPro": np.nan,
            'DockQ': np.nan,
            'Afpdb': np.nan,
        }

if __name__ == "__main__":
    import multiprocessing
    if not os.path.exists('AFDB') or not os.path.exists('AbDb'):
        util.unix('tar xvfz pdb.tar.gz')

    if n_CPU > multiprocessing.cpu_count()-2: n_CPU=0

    t=pd.read_csv('benchmark.csv')
    tasks = []
    for i, r in t.iterrows():
        s_pdb=r['pdb']+".pdb"
        tasks.append(f"AbDb/{s_pdb}")
    #tasks=tasks[:20]

    out = map(cost, tasks, n_CPU=n_CPU)

    tmp = pd.DataFrame(out)
    util.df2sdf(tmp, s_format="%.4g").to_csv("afpdb_dist.csv", index=False)
    print(tmp.mean(axis=0))
