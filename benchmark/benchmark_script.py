#!/usr/bin/env python
import sys
sys.path.insert(0, '/da/NBC/ds/zhoyyi1/score/')
#sys.path.insert(0, '/da/NBC/ds/zhoyyi1/afpdb/')
import glob, util, re, parmap, os, shutil, random, re
import numpy as np, pandas as pd
from afpdb.afpdb import Protein
from protein.mypdb import MySeq, MyColabFold as CF
from afdb import AFDB
from protein.abseq import Ab

from Bio.PDB import PDBParser, Selection, Superimposer, NeighborSearch
from Bio.PDB.Polypeptide import three_to_one
import tempfile
from numpy.testing import assert_array_equal
import time
import warnings
import traceback
warnings.filterwarnings("ignore")

ABDB_NR = "/da/NBC/ds/zhoyyi1/abdb/NR_LH_Protein_Chothia"
db = AFDB()
# number of repeats in each use case
N1=20
N2=100
N3=500
N4=500
#N1=N2=N3=N4=5

def cost(X):
    fd_af, exp_pdb, md5 = X
    m = re.search(r"(?P<pdb>\w+).pdb$", exp_pdb)
    pdb_code=m.group('pdb')

    # obtain the AF-predicted structure as p1
    s_pdb = CF.get_pdb(fd_af)
    p1 = Protein(s_pdb)
    chains = p1.chain_id()
    p1.rename_chains({"A": "H", "B": "L", "C": "G"}, inplace=True)
    # obtain experimental structure as p0
    p0 = Protein(exp_pdb)
    ag = [x for x in p0.chain_id() if x not in ("H", "L")][0]
    p0.rename_chains({ag: "G"}, inplace=True)
    # both p0 and p1 will have chain named H, L (Antibody), and G (Antigen)

    p1 = p1.extract("H:L:G", as_rl=True)
    p0 = p0.extract("H:L:G", as_rl=True)
    s1 = p1.seq().replace(":", "")
    s0 = p0.seq().replace(":", "")

    # AlphaFold model had missing residues replaced as G
    rs_G = p1.rs(p0.rsi_missing())
    if len(rs_G):  # we need to exclude X -> G residues from p1
        rs = p1.rs_not(rs_G)
        p1 = p1.extract(rs)
        s1 = p1.seq().replace(":", "")

    if len(p0) != len(p1) or s0 != s1:
        print("ERROR> ", fd_af, p0.seq(), "\n", p1.seq(), len(p0), len(p1))
        exit()

    p0.renumber('NOCODE', inplace=True)
    p1.renumber('NOCODE', inplace=True)

    # p0: experimental structure
    # p1: AF predicted structure

    # let's always clone objects to prevent potential caching
    try:
        ##################
        ### Use case 1: interface
        ##################

        # ----------------
        # afpdb implementation: return both residue selections
        # no K-D tree optimization
        dur_int_afpdb=0
        for i in range(N1):
            p=p1.clone()
            start=time.time()
            rs_binder, rs_seed, t_dist = p.rs_around("G", dist=5, drop_duplicates=False, kdtree_bucket_size=0)
            rs_int = rs_binder | rs_seed
            dur_int_afpdb+=time.time()-start
        dur_int_afpdb=dur_int_afpdb/N1*1000

        # With K-D tree optimization
        dur_int_afpdb2=0
        for i in range(N1):
            p=p1.clone()
            start=time.time()
            rs_binder_v2, rs_seed_v2, t_dist = p.rs_around("G", dist=5, drop_duplicates=False, kdtree_bucket_size=10)
            rs_int = rs_binder | rs_seed
            dur_int_afpdb2+=time.time()-start
        dur_int_afpdb2=dur_int_afpdb2/N1*1000

        # ----------------
        # biopython implementation: return residues within distance on chain_id chain and non-chain id chain
        # version one, all-by-all search
        dur_int_bio=0
        for i in range(N1):
            p=p1.to_biopython()
            start=time.time()
            res_on_non_chain_id_chain, res_on_chain_id_chain, t_dist = get_interface_res(p, "G", 5)
            all_res = res_on_non_chain_id_chain | res_on_chain_id_chain
            dur_int_bio+=time.time()-start
        dur_int_bio=dur_int_bio/N1*1000

        #----- version 2
        # search by each seed atom
        dur_int_bio2=0
        for i in range(N1):
            p=p1.to_biopython()
            start=time.time()
            res_on_non_chain_id_chain_v2, res_on_chain_id_chain_v2, t_dist = get_interface_res_search_only(p, "G", 5)
            all_res_v2 = res_on_non_chain_id_chain_v2 | res_on_chain_id_chain_v2
            dur_int_bio2+=time.time()-start
        dur_int_bio2=dur_int_bio2/N1*1000
        #------version 2 end

        # check they are the same
        assert rs_binder.name() == rs_binder_v2.name(), "kdtree rs_binder ids are not the same"
        assert rs_seed.name() == rs_seed_v2.name(), "kdtree rs_binder ids are not the same"
        assert set([i[1] for i in res_on_chain_id_chain]) == set(
            rs_seed.name()
        ), "v1 rs_seed ids are not the same"

        assert set([i[1] for i in res_on_non_chain_id_chain]) == set(
            rs_binder.name()
        ), "v1 rs_binder ids are not the same"
        assert set([i[1] for i in res_on_chain_id_chain]) == set(
            rs_seed.name()
        ), "v1 rs_seed ids are not the same"
        assert set([i[1] for i in res_on_non_chain_id_chain_v2]) == set(
            rs_binder.name()
        ), "v2 rs_binder ids are not the same"
        assert set([i[1] for i in res_on_chain_id_chain_v2]) == set(
            rs_seed.name()
        ), "v2 rs_seed ids are not the same"

        # ------------------

        ##################
        ### Use case 2: alignment
        ##################

        # ----------------
        # afpdb implementation:

        dur_align_afpdb=0
        for i in range(N2):
            # use a new object each time
            pp0=p0.clone()
            pp1=p1.clone()
            start=time.time()
            R, t = pp1.align(pp0, rs_binder, rs_binder, ats="N,CA,C,O")
            dur_align_afpdb+=time.time()-start
        dur_align_afpdb = dur_align_afpdb/N2*1000

        p_afpdb=pp1 # p_afpdb is the already aligned version of p1 using afpdb

        # ----------------
        # biopython implementation:

        # prepare input
        res_on_non_chain_id_chain_exp, _, _ = get_interface_res(p0.to_biopython(), "G", 5)

        dur_align_bio= 0
        for i in range(N2):
            pp0=p0.to_biopython()
            pp1=p1.to_biopython()
            start=time.time()
            # res_on_non_chain_id_chain was the value returned from the previous use case
            binder_atoms_exp = get_atoms_for_alignment(
                pp0, res_on_non_chain_id_chain_exp
            )
            binder_atoms_af = get_atoms_for_alignment(
                pp1, res_on_non_chain_id_chain_exp
            )
            # Now we initiate the superimposer:
            super_imposer = Superimposer()
            super_imposer.set_atoms(binder_atoms_exp, binder_atoms_af)
            super_imposer.apply(pp1.get_atoms())
            dur_align_bio+=time.time()-start
        dur_align_bio= dur_align_bio/N2*1000

        p_biopython=pp1 # p_biopython is the already aligned version of p1 using afpdb

        # check the euclidean distance b/t the two rotation matrix and translation vector
        print(
            f"the L2 between afpdb rotation matrix and biopython is {np.linalg.norm(R - super_imposer.rotran[0])}, the L2 distance b/t afpdb translation vector and biopython is {np.linalg.norm(t - super_imposer.rotran[1])}"
        )

        rmsd_diff = p_afpdb.rmsd(p0, rs_binder, rs_binder, ats="N,CA,C,O") - Protein(p_biopython).rmsd(p0, rs_binder, rs_binder, ats="N,CA,C,O")
        if rmsd_diff > 0.3:
            raise ValueError(f"rmsd difference is too large: {rmsd_diff}")

        ##################
        ### Use case 3: rmsd calculation for alignment
        ##################

        # -----------------
        # afpdb implementation
        dur_rmsd_afpdb = 0
        for i in range(N3):
            pp0=p0.clone()
            pp1=p1.clone()
            start=time.time()
            pp1.rmsd(pp0, rs_int, rs_int, ats="N,CA,C,O")
            dur_rmsd_afpdb +=time.time()-start
        dur_rmsd_afpdb = dur_rmsd_afpdb/N3*1000

        # -----------------
        # biopython implementation
        # Do not align, calculate RMSD only, no alignment was done here
        dur_rmsd_bio= 0
        for i in range(N3):
            pp0=p0.to_biopython()
            pp1=p1.to_biopython()
            start=time.time()
            # get atoms from binder_target_res
            atoms_for_rsmd_aligned = get_atoms_for_alignment(
                pp1, all_res
            )
            atoms_for_rsmd_target = get_atoms_for_alignment(
                pp0, all_res
            )
            # Get coordinates of atoms
            coords1 = [atom.coord for atom in atoms_for_rsmd_aligned]
            coords2 = [atom.coord for atom in atoms_for_rsmd_target]
            calculate_rmsd(coords1, coords2)
            dur_rmsd_bio+=time.time()-start

            assert atoms_for_rsmd_aligned == atoms_for_rsmd_target

        dur_rmsd_bio= dur_rmsd_bio/N3*1000

        #########################
        ### Use case 4: Assign B factor
        #########################

        # create a b_factor array to flag CDRs
        seq = p0.seq_dict()
        ab = Ab(seq["H"], chain="H")
        cdr_h, _, _ = ab.combine_cdr()
        ab = Ab(seq["L"], chain="L")
        cdr_l, _, _ = ab.combine_cdr()
        c_pos = p0.chain_pos()
        b_factor = np.zeros(len(p0))
        if cdr_h is not None:
            for i, (a, b) in enumerate(cdr_h):
                offset, _ = c_pos["H"]
                b_factor[a + offset : b + offset + 1] = (i + 1) / 10
        if cdr_l is not None:
            for i, (a, b) in enumerate( cdr_l):
                offset, _ = c_pos["L"]
                b_factor[a + offset : b + offset + 1] = (i + 5) / 10

        dur_b_afpdb = 0
        for i in range(N4):
            pp1=p1.clone()
            start=time.time()
            pp1.b_factors(b_factor)
            dur_b_afpdb+=time.time()-start
        dur_b_afpdb = dur_b_afpdb/N4*1000
        p_afpdb=pp1

        # -----------------
        # biopython implementation
        dur_b_bio= 0
        for i in range(N4):
            pp1=p1.to_biopython()
            start=time.time()
            b_factor = set_b_factor_per_residue(pp1, b_factor)
            dur_b_bio+=time.time()-start
        dur_b_bio= dur_b_bio/N4*1000

        biopython_assigned = get_res_level_b_factor(pp1)
        assert_array_equal(biopython_assigned, p_afpdb.b_factors())


        return {
            "PDB": pdb_code,
            "Interface (Afpdb)": dur_int_afpdb,
            "Interface (Afpdb2)": dur_int_afpdb2,
            "Interface (BioPython)": dur_int_bio,
            "Interface (Biopython 2)": dur_int_bio2,
            "Align (Afpdb)": dur_align_afpdb,
            "Align (Biopython)": dur_align_bio,
            "RMSD (Afpdb)": dur_rmsd_afpdb,
            "RMSD (Biopython)": dur_rmsd_bio,
            "B-factor (Afpdb)": dur_b_afpdb,
            "B-factor (Biopython)": dur_b_bio,
            "N_interface": len(rs_int),
            'N_Ab': len(p0.rs("H:L")),
            'N_ag': len(p0.rs('G'))
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "PDB": pdb_code,
            "Interface (Afpdb)": np.nan,
            "Interface (Afpdb2)": np.nan,
            "Interface (BioPython)": np.nan,
            "Interface (Biopython 2)": np.nan,
            "Align (Afpdb)": np.nan,
            "Align (Biopython)": np.nan,
            "RMSD (Afpdb)": np.nan,
            "RMSD (Biopython)": np.nan,
            "B-factor (Afpdb)": np.nan,
            "B-factor (Biopython)": np.nan,
            "N_interface": np.nan,
            'N_Ab': len(p0.rs("H:L")),
            'N_ag': len(p0.rs('G'))
        }


############################################################
### Auxilary functions in Biopython to perform the above task
############################################################


def get_interface_res(structure, seeds, dist: float):
    """Given a list of residues (or a chain id in the special case),
     get nearby residues (in the format of index+insertion code
     and one letter AA) that have atoms within the cutoff distance away from other unspecified
     chains, in addition, get the nearby residues from other unspecified chains on the chain_id.
     Distance are calculated based on all atoms

    Args:
        structure (Bio.PDB.Structure.Structure): input structure object to retrive atoms from
        chain_id (str): a chain id, e.g 'C' for chain C, or a list of residue tuples
        dist (float): distance cutoff to use

    Returns:
        a list of tuple: a tuple of two lists, the first list contains the nearby residues on the other chains
         and the second list contains the nearby residues on the chain_id. Each element in the list is a tuple
         in the format of (chain_name, index+insertion code, one letter AA), e.g. ('C', '1A', 'A')
    """
    if type(seeds) is str:
        ## we handle the special case tht seeds is specified using a chain name
        out=set()
        for chain in structure[0]:
            chain_id=chain.get_id()
            if chain_id!=seeds: continue
            for residue in chain:
                _, res_num, icode=residue.get_id()
                out.add((chain_id, str(res_num)+icode.strip()))
        seeds=out

    # implementation is to support the general case the input seeds is a list of residue id tuples

    all_res_atom = [atom for chain in structure[0] for residue in chain for atom in residue]
    neighbor = NeighborSearch(all_res_atom)
    neighbors_all = neighbor.search_all(radius=dist, level="A") #make one for search
    seed_res_info=set()
    neighbor_res_info=set()

    out=[]
    for (atm0, atm1) in neighbors_all: #item (C_res1, A_chain_res1) # (A_res1, B_res2) ('', 1, 'A')
        res0=atm0.get_parent()
        res1=atm1.get_parent()
        res0_chain = res0.get_parent().get_id()
        res1_chain = res1.get_parent().get_id()
        aa0, num_res0, insertion_code_res0 = res0.get_id()
        aa1, num_res1, insertion_code_res1 = res1.get_id()
        res0_info = (res0_chain, str(num_res0) + insertion_code_res0.strip())
        res1_info = (res1_chain, str(num_res1) + insertion_code_res1.strip())
        if res0_info in seeds and res1_info not in seeds:
            seed_res_info.add(res0_info)
            neighbor_res_info.add(res1_info)
            out.append((res0_info[0], res0_info[1], aa0, atm0.get_name(), res1_info[0], res1_info[1], aa1, atm1.get_name(), np.linalg.norm(atm0.coord-atm1.coord)))
        elif res0_info not in seeds and res1_info in seeds:
            seed_res_info.add(res1_info)
            neighbor_res_info.add(res0_info)
            out.append((res1_info[0], res1_info[1], aa1, atm1.get_name(), res0_info[0], res0_info[1], aa0, atm0.get_name(), np.linalg.norm(atm0.coord-atm1.coord)))
    dist=pd.DataFrame(data=out, columns=['chain_a','resn_a','aa_a','atom_a','chain_b','resn_b','aa_b','atom_b','dist'])
    dist=dist.sort_values('dist')
    #dist=dist.drop_duplicates(['chain_a','resn_a','chain_b','resn_b'])
    return neighbor_res_info, seed_res_info, dist


def get_interface_res_search_only(structure, seeds, dist: float):
    """Given a list of residues (or a chain id in the special case),
     get nearby residues (in the format of index+insertion code
     chains, in addition, get the nearby residues from other unspecified chains on the chain_id.
     Distance are calculated based on all atoms

    Args:
        structure (Bio.PDB.Structure.Structure): input structure object to retrive atoms from
        chain_id (str): a chain id, e.g 'C' for chain C
        dist (float): distance cutoff to use

    Returns:
        A list of tuple: a tuple of two lists, the first list contains the nearby residues on the other chains
         and the second list contains the nearby residues on the chain_id. Each element in the list is a tuple
         in the format of (chain_name, index+insertion code, one letter AA), e.g. ('C', '1A', 'A')
    """
    if type(seeds) is str:
        ## we handle the special case tht seeds is specified using a chain name
        out=set()
        for chain in structure[0]:
            chain_id=chain.get_id()
            if chain_id!=seeds: continue
            for residue in chain:
                _, res_num, icode=residue.get_id()
                out.add((chain_id, str(res_num)+icode.strip()))
        seeds=out

    # implementation is to support the general case the input seeds is a list of residue id tuples

    atom_from_neighbor= list()
    atom_from_seed = list()
    for chain in structure[0]:
        chain_id=chain.get_id()
        for residue in chain:
            _, res_num, icode=residue.get_id()
            res_info=(chain_id, str(res_num)+icode.strip())
            if res_info in seeds:
                atom_from_seed.extend([atom for atom in residue])
            else:
                atom_from_neighbor.extend([atom for atom in residue])
    neighbor = NeighborSearch(atom_from_neighbor)
    neighbor_res_info = set()
    seed_res_info = set()
    out=[]
    for seed in atom_from_seed:
        neighbors = neighbor.search(seed.get_coord(), radius=dist, level="A")
        if len(neighbors)==0: continue
        aa, num_res, insertion_code = seed.get_parent().get_id()
        seed_res=(chain_id, str(num_res)+insertion_code.strip())
        seed_res_info.add(seed_res)
        xyz_seed=seed.coord
        for atm in neighbors:
            res=atm.get_parent()
            aa2, num_res, insertion_code = res.get_id()
            res_info=(res.get_parent().get_id(), str(num_res) + insertion_code.strip())
            neighbor_res_info.add(res_info)
            xyz=atm.coord
            out.append((seed_res[0], seed_res[1], aa, seed.get_name(), res_info[0], res_info[1], aa2, atm.get_name(), np.linalg.norm(xyz_seed-xyz)))
            # compute shortest distance
    dist=pd.DataFrame(data=out, columns=['chain_a','resn_a','aa_a','atom_a','chain_b','resn_b','aa_b','atom_b','dist'])
    dist=dist.sort_values('dist')
    dist=dist.drop_duplicates(['chain_a','resn_a','chain_b','resn_b'])
    return neighbor_res_info, seed_res_info, dist

def get_atoms_for_alignment(structure, res_on_non_chain_id_chain: list) -> list:
    """Get backbone atoms from the input structure for alignment,
    modified from https://gist.github.com/andersx/6354971

    Args:
        structure (Bio.PDB.Structure.Structure): input structure object to retrive atoms from
        res_on_non_chain_id_chain (list): one or joint output from `get_interface_res` func,
                                            for determining which atoms to retrieve

    Returns:
        list: atoms to use to derive the rotation and translation matrix
    """
    atoms_for_alignment = []
    chain_non_chainid_lst = set([i[0] for i in res_on_non_chain_id_chain])
    for query_chain_element in chain_non_chainid_lst:
        query_chain = structure[0][query_chain_element]
        for other_residue in query_chain:  # query chain is usually the antibody
            _, res_seq, icode = other_residue.get_id()
            res_info =(query_chain_element, str(res_seq) + icode.strip())
            if res_info in res_on_non_chain_id_chain:
                atoms_for_alignment.extend([other_residue[x] for x in ["N", "CA", "C", "O"]])
    return atoms_for_alignment

def calculate_rmsd(array1, array2):
    """"Calculate RMSD given two arrays"""
    return np.linalg.norm(np.array(array1) - np.array(array2))

def set_b_factor_per_residue(structure, b_factor: np.ndarray):
    """assign residue level b factor to a structure, the residue
    level b factor must have the same order and length as the amino acid
    in the structure

    Args:
        structure (Bio.PDB.Structure.Structure): a structure whose bfactor will be assigned
        b_factor (np.ndarray): residue level b factor, should be the same length and order
                                as the number of AA in the structure file
    """
    idx=0
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                atom.set_bfactor(b_factor[idx])
            idx+=1
    return b_factor

def get_res_level_b_factor(structure):
    assigned_b_factor = []
    for chain in structure[0]:  # structure[0] is the 1st model
        assigned_b_factor.extend([res['CA'].get_bfactor() for res in chain])
    return np.array(assigned_b_factor)

def has_backbone(exp_pdb):
    p0=Protein(exp_pdb)
    return len(p0.rs_missing_atoms(ats="N,CA,C,O"))==0

def counts(exp_pdb):
    p0=Protein(exp_pdb)
    return len(p0), len(p0.chain_id())

if __name__ == "__main__":
    if os.path.exists('benchmark.csv'):
        t=pd.read_csv('benchmark.csv')
    else:
        t = pd.read_csv("../RUN.NR/tasks.csv")  # [:20]
        # AF failed to predict these entries (coordinates are np.nan)
        failed_af = ["430c5a7be3100c8caff702a7eaa43780", "1e606ec24022f31ae17dc1313af88faa"]
        t=t[~t.md5.isin(failed_af)].copy()
        # remove entries where backbone atoms are missing
        tasks=[ ABDB_NR + "/" + r["pdb"] + ".pdb" for i,r in t.iterrows()]
        mask=parmap.map(has_backbone, tasks)
        t=t[mask]
        t.to_csv('benchmark.csv', index=False)

    # statistics for paper
    if False:
        tasks=[ ABDB_NR + "/" + r["pdb"] + ".pdb" for i,r in t.iterrows()]
        out=parmap.map(counts, tasks)
        print(np.mean([x[0] for x in out]))
        print(np.mean([x[1] for x in out]))
        exit()

    tasks = []
    for i, r in t.iterrows():
        fd_af = "/da/NBC/ds/zhoyyi1/score/"+AFDB.FOLDER + "/" + r["md5"]
        exp_pdb = ABDB_NR + "/" + r["pdb"] + ".pdb"
        tasks.append([fd_af, exp_pdb, r["md5"]])

    # the server has 64 cores, no other job is running
    # we use 50 cores
    out = parmap.map(cost, tasks, n_CPU=50)

    tmp = pd.DataFrame(out)
    print(f"there are {sum(tmp.isna().any(axis =1))} entries that failed")
    #PDB,Interface (Afpdb),Interface (Afpdb2),Interface (BioPython),Interface (Biopython 2),Align (Afpdb),Align (Biopython),RMSD (Afpdb),RMSD (Biopython),B-factor (Afpdb),B-factor (Biopython),N_interface,N_Ab,N_ag
    util.df2sdf(tmp, s_format="%.4g").to_csv("afpdb_cost.csv", index=False)
