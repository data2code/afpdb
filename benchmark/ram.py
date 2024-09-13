#!/usr/bin/env python

def rss():
    import psutil
    return psutil.Process().memory_info().rss/(1024*1024)

def afpdb_lib(X):
    before=rss()
    from afpdb.afpdb import Protein
    after=rss()
    return (after-before)

def biopython_lib(X):
    before=rss()
    from Bio import PDB
    from Bio.PDB import PDBIO
    after=rss()
    return (after-before)

def afpdb_io(X):
    from afpdb.afpdb import Protein
    af_pdb, exp_pdb = X
    before=rss()
    p1 = Protein(af_pdb)
    p0 = Protein(exp_pdb)
    after=rss()
    return (after-before)

def biopython_io(X):
    af_pdb, exp_pdb = X
    from Bio import PDB
    from Bio.PDB import PDBIO
    before=rss()
    parser = PDB.PDBParser()
    p1 = parser.get_structure('protein', af_pdb)
    p0 = parser.get_structure('protein', exp_pdb)
    after=rss()
    return (after-before)

def afpdb_int(X):
    from afpdb.afpdb import Protein
    af_pdb = X[0]
    p = Protein(af_pdb)
    rs_binder, rs_seed, t_dist = p.rs_around("G", dist=5, drop_duplicates=False, kdtree_bucket_size=10)
    rs_int = rs_binder | rs_seed

def biopython_int(X):
    af_pdb = X[0]
    from Bio import PDB
    parser = PDB.PDBParser()
    p = parser.get_structure('protein', af_pdb)
    res_on_non_chain_id_chain_v2, res_on_chain_id_chain_v2, t_dist = get_interface_res_search_only(p, "G", 5)
    all_res_v2 = res_on_non_chain_id_chain_v2 | res_on_chain_id_chain_v2

def afpdb_align(X):
    from afpdb.afpdb import Protein
    af_pdb, exp_pdb = X
    p1 = Protein(af_pdb)
    p0 = Protein(exp_pdb)
    #rs_binder, rs_seed, t_dist = p0.rs_around("G", dist=5, drop_duplicates=False, kdtree_bucket_size=10)
    rs_ab="H:L"
    R, t = p1.align(p0, rs_ab, rs_ab, ats="N,CA,C,O")

def biopython_align(X):
    af_pdb, exp_pdb = X
    from Bio import PDB
    from Bio.PDB import Superimposer
    parser = PDB.PDBParser()
    p1 = parser.get_structure('protein', af_pdb)
    p0 = parser.get_structure('protein', exp_pdb)

    rs_ab=get_res_by_chains(p0, ["H","L"])
    binder_atoms_exp = get_atoms_for_alignment(p0, rs_ab)
    binder_atoms_af = get_atoms_for_alignment(p1, rs_ab)
    # Now we initiate the superimposer:
    super_imposer = Superimposer()
    super_imposer.set_atoms(binder_atoms_exp, binder_atoms_af)
    super_imposer.apply(p1.get_atoms())

def afpdb_rmsd(X):
    from afpdb.afpdb import Protein
    af_pdb, exp_pdb = X
    p1 = Protein(af_pdb)
    p0 = Protein(exp_pdb)
    rs_ab = "H:L"
    p1.rmsd(p0, rs_ab, rs_ab, ats="N,CA,C,O")

def biopython_rmsd(X):
    af_pdb, exp_pdb = X
    from Bio import PDB
    parser = PDB.PDBParser()
    p1 = parser.get_structure('protein', af_pdb)
    p0 = parser.get_structure('protein', exp_pdb)

    rs_ab=get_res_by_chains(p0, ["H","L"])
    binder_atoms_exp = get_atoms_for_alignment(p0, rs_ab)
    binder_atoms_af = get_atoms_for_alignment(p1, rs_ab)
    # Get coordinates of atoms
    coords1 = [atom.coord for atom in atoms_for_rsmd_aligned]
    coords2 = [atom.coord for atom in atoms_for_rsmd_target]
    calculate_rmsd(coords1, coords2)

def afpdb_b(X):
    from afpdb.afpdb import Protein
    import numpy as np
    af_pdb = X[0]
    p = Protein(af_pdb)
    b_factor=np.random.rand(len(p))
    p.b_factors(b_factor)

def biopython_b(X):
    af_pdb = X[0]
    from Bio import PDB
    import numpy as np
    parser = PDB.PDBParser()
    p = parser.get_structure('protein', af_pdb)
    residue_count = 0
    for model in p:
        for chain in model:
            for residue in chain:
                residue_count += 1
    b_factor=np.random.rand(residue_count)
    set_b_factor_per_residue(p, b_factor)

def get_res_by_chains(structure, chains):
    out=set()
    for chain in structure[0]:
        chain_id=chain.get_id()
        if chain_id not in chains: continue
        for residue in chain:
            _, res_num, icode=residue.get_id()
            out.add((chain_id, str(res_num)+icode.strip()))
    return out

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
    import pandas as pd,numpy as np
    from Bio.PDB import NeighborSearch
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
    import numpy as np
    return np.linalg.norm(np.array(array1) - np.array(array2))

def set_b_factor_per_residue(structure, b_factor):
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

if __name__ == "__main__":
    import sys
    cmd=sys.argv[1]
    func=globals()[cmd]
    print(func(sys.argv[2:]))
