#!/usr/bin/env python
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import tempfile
import os, re, traceback, pathlib, requests

from .myalphafold.common import protein as afprt
from .myalphafold.common import residue_constants as afres
from .mycolabdesign import getpdb as cldpdb
from .mycolabdesign import utils as cldutl
from .mycolabdesign import protein as cldprt
from .mycolabfold.utils import CFMMCIFIO

# Type aliases for commonly used selection types
ContigType = Union[str, 'RS', 'RL', List[int], Tuple[int, ...], np.ndarray, int, None]
"""Type for residue/contig selections that can be used to initialize RS or RL objects.
Supports: strings ("H:L", "A1-10"), RS/RL objects, lists of integers, tuples of integers,
numpy arrays, single integers, or None."""

AtomSelectionType = Union[str, 'ATS', List[int], Tuple[int, ...], np.ndarray, int, None]
"""Type for atom selections that can be used to initialize ATS objects.
Supports: strings ("N,CA,C,O"), ATS objects, lists of integers, tuples of integers,
numpy arrays, single integers, or None."""

try:
    from .mol3D import Mol3D
    _Mol3D_=True
except:
    _Mol3D_=False
try:
    import pymol2
    _PYMOL_=True
except Exception as e:
    print(e)
    _PYMOL_=False
try:
    from afpdb import antibody
    _Ab_=True
except Exception as e:
    print(e)
    _Ab_=False

from afpdb import util,pdbinfo
from scipy.stats import rankdata
from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict, PDBIO, Structure
import numpy as np, pandas as pd
import tempfile
import os,re,traceback,pathlib,requests

def check_Mol3D() -> None:
    """Check if Py3DMol is installed and available for 3D visualization.

    Raises:
        SystemExit: If Py3DMol is not installed

    Note:
        Py3DMol is required for interactive 3D visualization in Jupyter notebooks.
        Install with: conda install conda-forge::py3Dmol
    """
    if not _Mol3D_:
        print("Requirement Py3DMol is not installed")
        exit()

def check_PyMOL() -> None:
    """Check if PyMOL is installed and available for advanced visualization.

    Raises:
        SystemExit: If PyMOL is not installed

    Note:
        PyMOL is required for advanced molecular visualization features.
        Install with: conda install conda-forge::pymol-open-source
    """
    if not _PYMOL_:
        print("Requirement PyMOL is not installed, try 'conda install conda-forge::pymol-open-source'")
        exit()

def check_Ab() -> None:
    """Check if anarci is installed and available for antibody analysis.

    Raises:
        SystemExit: If anarci is not installed

    Note:
        anarci is required for antibody sequence analysis and CDR detection.
        Install with: conda install bioconda::anarci
    """
    if not _Ab_:
        print("Requirement anarci is not installed, try 'conda install bioconda::anarci'")
        exit()

def rot_a2b(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate rotation matrix and translation vector to align coordinate sets.

    Returns (R, T) that can transform coordinates in a into coordinates in b by a@R+T ~ b

    Args:
        a (np.ndarray): Source coordinate array of shape (N, 3)
        b (np.ndarray): Target coordinate array of shape (N, 3)

    Returns:
        tuple: (R, t) where R is 3x3 rotation matrix and t is 1x3 translation vector

    Example:
        >>> coords_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> coords_b = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1]])
        >>> R, t = rot_a2b(coords_a, coords_b)
        >>> transformed = coords_a @ R + t
    """
    b_c=np.reshape(np.mean(b, axis=0), (1,3))
    a_c=np.reshape(np.mean(a, axis=0), (1,3))
    R=cldprt._np_kabsch(a-a_c, b-b_c, use_jax=False)
    #print("OUT>\n", a_c, "\n", R, "\n", t)
    # b=(a-a_c)@R+b_c
    # so, t=b_c-a_c@R
    t=b_c-a_c@R
    return (R, t)

def rot_a(a: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation matrix R and translation t to coordinates a.

    Args:
        a: Coordinate array of shape (N, 3)
        R: 3x3 rotation matrix
        t: Translation vector

    Returns:
        Transformed coordinates
    """
    return a@R+t

def _test_rot_a2b():
    """Test function for rotation alignment between two coordinate sets.

    Returns:
        None: Prints test results to console.
    """
    # N, CA, C, O, CB
    a=np.array([
        [-0.529, 1.360, -0.000],
        [0.000, 0.000, 0.000],
        [1.525, -0.000, -0.000],
        [-0.518, -0.777, -1.211],
        [0.626, 1.062, -0.000]
    ])
    theta=60/180*3.14159
    R=np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    t=np.array([[1,2,3]])
    b=a@R+t

    # now we solve (R, t), given a, b
    R2,t2 = rot_a2b(a, b)
    b2=a@R2+t2

    print("COMPARE\n\n", b2, "\n\n", b, "\n\n", b2-b)
    print("R>\n", R, R2, R-R2)
    print("t>\n", t, t2, t-t2)
#_test_rot_a2b()


# This is a wrapper for the AlphaFold alphafold.common.protein.Protein class
# the member self.data is a Protein instance
#
#  """Protein structure representation."""
#
#  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
#  # residue_constants.atom_types, i.e. the first three are N, CA, C.
#  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]
#
#  # Amino-acid type for each residue represented as an integer between 0 and
#  # 20, where 20 is 'X'.
#  aatype: np.ndarray  # [num_res]
#
#  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
#  # is present and 0.0 if not. This should be used for loss masking.
#  atom_mask: np.ndarray  # [num_res, num_atom_type]
#
#  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
#  residue_index: np.ndarray  # [num_res]
#
#  # 0-indexed number corresponding to the chain in the protein that this residue
#  # belongs to.
#  chain_index: np.ndarray  # [num_res]
#
#  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
#  # representing the displacement of the residue from its ground truth mean
#  # value.
#  b_factors: np.ndarray  # [num_res, num_atom_type]

class Protein:
    """
    A class representing a protein structure with comprehensive PDB manipulation capabilities.

    The Protein class is the core of afpdb, storing protein structures as NumPy arrays for
    efficient manipulation and analysis. It supports multiple input formats including PDB files,
    PDB IDs, AlphaFold model codes, BioPython structures, and CIF files.

    Data Structure:
        All protein data is stored in NumPy arrays within the .data attribute:
        - atom_positions (N_res, 37, 3): 3D coordinates for all 37 possible atom types
        - atom_mask (N_res, 37): binary mask indicating which atoms are present
        - residue_index: residue numbering from PDB
        - chain_index: numerical chain indices
        - b_factors (N_res, 37): temperature factors
        - chain_id: array of chain identifiers

    The sparse array design accommodates all 20 amino acid types with 37 unique atoms.
    Missing atoms have mask=0 and should be ignored based on atom_mask, not coordinates.

    Key Features:
        - Contig-based residue selection (e.g., "H:L", "H26-33", "P1-10")
        - Atom selection using standard PDB atom names
        - Structural alignments, RMSD calculations, and transformations
        - Integration with PyMOL and 3Dmol for visualization
        - Contact analysis and interface detection
        - Support for antibody-specific analysis methods

    Attributes:
        data: AlphaFold Protein instance containing atomic coordinates and metadata
        res_map: Dictionary mapping residue contigs to internal array indices

    Constructor Examples:
        >>> p = Protein("1crn")                    # PDB ID from RCSB
        >>> p = Protein("Q2M403")                  # AlphaFold model from EMBL
        >>> p = Protein("structure.pdb")           # Local PDB file
        >>> p = Protein("structure.cif")           # Local CIF file
        >>> p = Protein(biopython_structure)       # BioPython Structure object
        >>> p = Protein(pdb_string)                # PDB content as string
        >>> p = Protein("5cil.pdb", contig="H:L")  # Load specific chains only

    Basic Usage:
        >>> p = Protein("1crn")
        >>> print(p.chain_id())                   # ['A']
        >>> print(p.seq())                        # 'TTCCPSIVAR...'
        >>> p.show()                              # 3D visualization
        >>> ab_chains = p.extract("H:L")          # Extract antibody heavy/light chains
        >>> rmsd_val = p.rmsd(other_protein)      # Calculate RMSD

    See the afpdb tutorial for comprehensive examples and advanced usage patterns.
    """

    # Residue renumbering mode
    #    None: keep original numbering
    #    RESTART: 1, 2, 3, ... for each chain
    #    CONTINUE: 1 ... 100, 101 ... 140, 141 ... 245
    #    GAP200: 1 ... 100, 301 ... 340, 541 ... 645 # mimic AlphaFold gaps
    #    You can define your own gap by replacing GAP200 with GAP{number}, e.g., GAP10
    #    NOCODE: remove insertion code, 6A, 6B will become 6, 7

    RENUMBER={'NONE':None, 'RESTART':'RESTART', 'CONTINUE':'CONTINUE', 'GAP200':'GAP200', 'NOCODE':'NOCODE'}
    MIN_GAP_DIST=4.5 # used by seq(), when there are numbering gap in neighboring residues
                    # we consider no gap if Ca-Ca distance is smaller than this threshold (typically 3.82A, if no deletion)


    _DEBUG=False

    @staticmethod
    def debug(value: Optional[bool] = None) -> bool:
        """Get or set debug mode for protein operations.

        Args:
            value: If provided, sets debug mode. If None, returns current debug state.

        Returns:
            bool: Current debug mode state.
        """
        if value is None:
            return Protein._DEBUG
        else:
            Protein._DEBUG=afprt._DEBUG=value
            return value

    def __init__(self, pdb_str: Union[str, 'Protein', afprt.Protein, Structure.Structure, pathlib.Path, None] = None,
                 contig: ContigType = None, assembly1: bool = False) -> None:
        """Initialize a Protein object from various sources.

        Args:
            pdb_str (str, Protein, afprt.Protein, Structure, or None):
                - String: PDB/mmCIF content, file path, or PDB code
                - Protein: Another afpdb.Protein instance to clone
                - afprt.Protein: AlphaFold Protein instance to wrap
                - Structure: BioPython Structure object
                - None: Creates empty protein structure
            contig (str, optional): Contig selection to extract after loading.
                Format examples: "A10-20", "A,B", "A10-20,B30-40"
            assembly1 (bool, optional): If True, load biological assembly 1
                instead of asymmetric unit. Only applies when fetching from PDB code.

        Examples:
            # From PDB code
            >>> p = Protein("1crn")

            # From file
            >>> p = Protein("structure.pdb")
            >>> p = Protein("/path/to/structure.cif")

            # From PDB string content
            >>> pdb_content = "ATOM   1  N   ALA A   1 ..."
            >>> p = Protein(pdb_content)

            # Extract specific region during loading
            >>> p = Protein("1crn", contig="A10-50")

            # Load biological assembly
            >>> p = Protein("7epp", assembly1=True)

            # Clone existing protein
            >>> p2 = Protein(p)

        Note:
            - File format is automatically detected from extension (.pdb, .cif)
            - PDB codes are automatically fetched from RCSB PDB
            - Chain IDs are preserved from input when possible
        """
        # contains two data members
        # self.data # AlphaFold Protein instance
        self._set_data(pdb_str, contig, assembly1=assembly1)
        self.chain_id() # make sure we have self.data.chain_id

    def _set_data(self, data: Any, contig: ContigType = None, assembly1: bool = False) -> None:
        cls=type(data)
        if data is None or ((type(data) is str) and data ==''):
            data="MODEL     1\nENDMDL\nEND"
        if type(data) is pathlib.PosixPath:
            data=str(data)
        if type(data) is str:
            if "\n" not in data:
                if os.path.exists(data):
                    fmt=Protein.guess_format(data)
                    if fmt in ('pdb',''):
                        self.from_pdb(data)
                    else:
                        self.from_cif(data)
                else: # we assume it's a pdb code, we fetch
                    self.fetch_pdb(data, assembly1=assembly1)
            else:
                self.data=afprt.from_pdb_string(data)
        # in jupyter, isinstance is not always work, probably due to auto reload
        elif isinstance(data, Protein) or (cls.__name__=="Protein" and cls.__module__.endswith("afpdb")):
            self.data=data.data.clone()
        elif isinstance(data, afprt.Protein) or (cls.__name__=="Protein" and 'myalphafold.common.protein' in cls.__module__): #AF protein object
            self.data=data.clone()
        elif isinstance(data, Structure.Structure): #AF protein object
            self.data=afprt.Protein.from_biopython(data)
        self._make_res_map()
        if contig is not None:
            self.extract(contig, inplace=True)
        # lookup (chain, resi) => position

    def __getattr__(self, key: str) -> Any:
        if key=='data_prt': return self.data
        raise AttributeError(key)

    def _make_res_map(self) -> None:
        # map residue ID (format B34A means B chain, 34A residue) into its position in the data arrays
        self.res_map={}
        c_pos=self.chain_pos()
        resi=self.data.residue_index
        #print(c_pos)
        for k,(b,e) in c_pos.items():
            for i in range(b, e+1):
                self.res_map[f'{k}{resi[i]}']=i

    def split_residue_index(self, residue_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split residue indices into integer and insertion code parts.

        Args:
            residue_index: Array of residue indices that may contain insertion codes.

        Returns:
            tuple: (integer_indices, insertion_codes) where integer_indices are the
                   numeric parts and insertion_codes are the alphabetic suffixes.
        """
        # residue index may contain insertion code, we split it into two lists, one contain the integer numbers, another contain only insertion code
        idx=[]
        code=[]
        for x in residue_index:
            m=re.search(r'(?P<idx>-?\d+)(?P<code>\D*)$', x)
            idx.append(int(m.group('idx')))
            code.append(m.group('code'))
        return np.array(idx), np.array(code)

    def merge_residue_index(self, idx: np.ndarray, code: np.ndarray) -> np.ndarray:
        """Merge integer indices and insertion codes back into residue indices.

        Args:
            idx: Array of integer residue numbers.
            code: Array of insertion code strings.

        Returns:
            numpy.ndarray: Combined residue indices with insertion codes.
        """
        out=[ f"{a}{b}" for a,b in zip(idx, code) ]
        return np.array(out, dtype=np.dtype('<U6'))

    def b_factors(self, data: Optional[Union[np.ndarray, int, float]] = None, rs: ContigType = None) -> np.ndarray:
        """Get or set B-factors for residues.

        Args:
            data: B-factor values to set, or None to get current values
            rs: Residue selection to apply to

        Returns:
            Array of B-factor values
        """
        rs=RS(self, rs)
        if data is not None:
            if isinstance(data, (int, float)):
                data=np.array([data]*len(rs))
            n=data.shape[0]
            assert(n==len(rs))
            _, m=self.data.b_factors.shape
            if data.ndim==1:
                data=np.tile(data.reshape(n,1), (1,m))
            data[self.data.atom_mask[rs.data]==0]=0
            self.data.b_factors[rs.data,:]=data
        return self.data.b_factors[rs.data,1].copy() # use CA

    def b_factors_by_chain(self, data: Optional[Dict[str, Union[np.ndarray, int, float]]] = None) -> Dict[str, np.ndarray]:
        """Get or set B-factors by chain.

        Args:
            data: Dictionary mapping chain IDs to B-factor values

        Returns:
            Dictionary mapping chain IDs to B-factor arrays
        """
        c_pos=self.chain_pos()
        if data is not None: #get b-factor
            c_seq=self.seq_dict()
            c_n=self.len_dict()
            for k,v in data.items():
                if k not in c_seq:
                    raise Exception(f"Chain {k} not in c_pos")
                # c_seq may contain missing residues, so we should use c_pos instead
                n=c_n[k]
                if isinstance(v, (int, float)):
                    v=np.array([v]*n)
                if(len(v)!=n):
                    raise Exception(f"Sequence length, without X, does not match: chain {k}, b-factors {len(v)} {v[:3]}, seq {n} {c_seq[k]}")

                _, m=self.data.b_factors.shape
                if v.ndim==1:
                    v=np.tile(v.reshape(n,1), (1,m))
                v2=np.copy(v)
                #assert(k in c_seq and len(v)==len(c_seq[k]))
                b,e=c_pos[k][0],c_pos[k][1]
                v2[self.data.atom_mask[b:e+1]==0]=0
                self.data.b_factors[b:e+1]=v2
        c_out={}
        for k,v in c_pos.items():
            c_out[k]=self.data.b_factors[c_pos[k][0]:c_pos[k][1]+1, 1].copy()
        return c_out

    def search_seq(self, seq: str, in_chains: Optional[ContigType] = None) -> Any:
        """Search a sequence fragment, return all matches"""
        util.warn_msg("search_seq() is replaced by rs_seq, it returns a list of selections, not a sequence any more!")
        return self.rs_seq(seq, in_chains=in_chains)

    def chain_id(self) -> np.ndarray:
        """Get the chain identifiers for all chains in the structure.

        Returns:
            numpy.ndarray: NumPy array of chain identifiers (strings) in the order
                          they appear in the data structure. If chain_id attribute
                          doesn't exist, generates standard PDB chain IDs (A, B, C, ...).

        Examples:
            >>> p = Protein("1crn")
            >>> chains = p.chain_id()
            >>> print(chains)
            array(['A'], dtype='<U1')

            >>> p = Protein("2hhb")  # Hemoglobin with multiple chains
            >>> chains = p.chain_id()
            >>> print(chains)
            array(['A', 'B', 'C', 'D'], dtype='<U1')

        Note:
            - Returns NumPy array containing string chain identifiers
            - This method ensures that the protein structure has valid chain
              identifiers and will create them if they don't exist
            - Array elements can be accessed like chains[0] to get first chain ID
        """
        if self.data is not None and hasattr(self.data, 'chain_id'):
            return self.data.chain_id
        else:
            n=len(np.unique(self.data.chain_index))
            self.data.chain_id=np.array(afprt.PDB_CHAIN_IDS)[:n]
            return self.data.chain_id

    def chain_list(self) -> List[str]:
        """Get chain identifiers in their order of appearance in the PDB structure.

        **OBSOLETE**: This method is deprecated and can be replaced by chain_id().
        Use chain_id() instead for better performance and consistency.

        Returns:
            list: Chain identifiers in PDB order (order of first occurrence of each chain)

        Examples:
            >>> p = Protein("1crn")
            >>> chains = p.chain_list()  # DEPRECATED
            >>> print(chains)
            ['A']

            # Preferred alternative:
            >>> chains = p.chain_id().tolist()  # Use this instead
            >>> print(chains)
            ['A']

        Note:
            - This method is basically obsolete and maintained for backward compatibility
            - Use chain_id() instead, which provides the same functionality more efficiently
            - Unlike chain_id(), this returns a Python list instead of NumPy array
        """
        chains=self.chain_id()
        # PDB order
        return util.unique2([chains[x] for x in self.data.chain_index ])

    def rs_missing_atoms(self, ats: AtomSelectionType = None, debug: bool = False) -> 'RS':
        """a residue selection object pointing to residues with missing atoms"""
        p=self.data
        n,m=p.atom_positions.shape[:2]
        out=[]
        chains=self.chain_id()
        ats=ATS(ats)
        c_map={afres.restype_order[k]:ATS(np.array([afres.atom_order[x] for x in afres.residue_atoms[v]])) for k,v in afres.restype_1to3.items() }
        c_map={k:(v & ats).data for k,v in c_map.items()}
        for i,res in enumerate(p.aatype):
            if res >= 20: continue # residu X
            mask=p.atom_mask[i, c_map.get(res)] > 0
            if not np.all(mask):
                if debug:
                    out2=[ a for a,x in zip(atoms, mask) if x<1 ]
                    print(f"Missing atoms: {out2}")
                out.append(i)
        return RS(self, out)

    def chain_pos(self) -> Dict[str, Tuple[int, int]]:
        """Return a dict of the residue position of each chain."""
        x=self.data.chain_index
        chains=self.chain_id()
        c_pos={}
        for i,c in enumerate(self.data.chain_index):
            #print(">>>", i, c)
            if c not in c_pos:
                #c_pos[c]=[np.inf,-np.inf]
                c_pos[c]=[i, i]
            else:
                # residues for the same chain must not be broken for the result to be meaningful
                if i!=c_pos[c][1]+1:
                    raise Exception("The residues for chain {c} are not continous!")
            #c_pos[c][0]=min(c_pos[c][0], i)
            c_pos[c][1]=i #max(c_pos[c][1], i)
        c_pos={chains[k]:v for k,v in c_pos.items()}
        return c_pos

    def merge_chains(self, gap: int = 200, inplace: bool = False) -> Tuple['Protein', Dict[str, Tuple[int, int]]]:
        """Merge chains for AlphaFold monomer input.

        Args:
            gap: Gap size between chains in residue numbering
            inplace: Whether to modify current object or return new one

        Returns:
            Tuple of (merged_protein, original_chain_positions)
        """
        obj = self if inplace else self.clone()
        c_pos=self.chain_pos()
        chains=sorted([ (k,v[0],v[1]) for k,v in c_pos.items()], key=lambda x: x[1])
        # use the chain of the first residue as the target chain name
        obj.data.chain_id=np.array([obj.chain_list()[0]])
        obj.data.chain_index[:]=0
        base=0
        out=[]
        for i,(k,b,e) in enumerate(chains):
            n=e-b+1

            # take care of residue id with insertion code
            idx,code=self.split_residue_index(self.data.residue_index[b:e+1])
            idx=idx-idx[0]+1 # change to start from 1
            n_cnt=idx[-1]-idx[0]+1 # including missing residues
            obj.data.residue_index[b:e+1]=self.merge_residue_index(idx+base, code)
            base+=n_cnt+gap
        obj._make_res_map()
        return (obj, c_pos)

    def peptide_bond_length(self, rs: ContigType = None) -> pd.DataFrame:
        """Calculate peptide bond lengths in the protein backbone.

        Args:
            rs: Residue selection to analyze. If None, analyzes all residues.

        Returns:
            pd.DataFrame: DataFrame with bond length information for backbone connections.

        Note:
            This method is obsolete. Use backbone_bond_length() instead.
        """
        # mark this method as obsolete, ask users to use backbone_bond_length instead
        util.warn_msg("peptide_bond_length is obsolete, use backbone_bond_length instead.")
        return self.backbone_bond_length(rs)

    def backbone_bond_length(self, rs: ContigType = None) -> pd.DataFrame:
        """Calculate backbone bond distances. See peptide_bond_length for residue_range format.
        Return:
        DataFrame with columns: chain, resi_a, resn_a, resi_b, resn_b, bond_n_ca, bond_ca_c, bond_pept'
        """
        from numpy.linalg import norm
        l_rs = self.rs(rs).split()

        N,C,CA=afres.atom_order['N'],afres.atom_order['C'],afres.atom_order['CA']
        data=[]
        for my_rs in l_rs:
            chain = my_rs.chain(unique=True)[0]
            resn = my_rs.name()
            for i, pos in enumerate(my_rs.data):
                bond_n_ca=norm(self.data.atom_positions[pos, CA]-self.data.atom_positions[pos, N])
                bond_ca_c=norm(self.data.atom_positions[pos, CA]-self.data.atom_positions[pos, C])

                this_n=self.data.atom_positions[pos, N]
                bond_pept=norm(this_n - prev_c) if i>0 else np.nan
                prev_c=self.data.atom_positions[pos, C]
                prev_pos = pos - 1 if i > 0 else -1
                prev_resn = resn[i -1] if i > 0 else ""
                data.append([chain, prev_pos, prev_resn, pos, resn[i], bond_n_ca, bond_ca_c, bond_pept])
        df = pd.DataFrame(data, columns=['chain', 'resi_a', 'resn_a', 'resi_b', 'resn_b', 'bond_n_ca', 'bond_ca_c', 'bond_pept'])
        return df

    def resn(self, new_resn: Optional[Union[List[str], np.ndarray]] = None, rl: ContigType = None) -> Optional[np.ndarray]:
        old=self.data.residue_index.copy()
        rl=self.rl(rl)
        if new_resn is not None:
            assert(len(new_resn)==len(rl))
            if len(rl)==0: return None
            s_old_seq=self.seq(gap="-")
            self.data.residue_index[rl.data]=np.array(new_resn)
            self._make_res_map()
            s_new_seq=self.seq(gap="-")
            # check if the renaming introduce artificial gaps that did not exist previously
            # E.g., after Chothia renumbering, two residue 95, 96 becomes 95, 100, creating a fake gap
            # This has been improved as we introduce spatial distance checking into the gap logic.
            if s_old_seq!=s_new_seq and Protein.debug():
                util.warn_msg(f"Gapping in sequence was changed after renumbering!!!\nOld: {s_old_seq}\nNew: {s_new_seq}")
            return old[rl.data]
        else:
            if len(rl)==0: return old[:0]
            return old[rl.data]

    def renumber(self, renumber: Optional[str] = None, inplace: bool = False) -> Tuple['Protein', np.ndarray]:
        """Renumber residues in the protein structure using different numbering schemes.

        Provides systematic renumbering of residues to standardize or modify residue
        numbering schemes. This is particularly useful for structure prediction inputs,
        PyMOL visualization, or when working with structures that have complex or
        inconsistent numbering.

        Args:
            renumber (str, optional): Renumbering scheme to apply:
                - None: No renumbering, returns original structure
                - "RESTART": Start from 1 for each chain independently
                - "CONTINUE": Continuous numbering across all chains (1,2,3...)
                - "GAP200": 200-residue gaps between chains (AlphaFold style)
                - "GAP{N}": Custom gap size N between chains (e.g., "GAP50")
                - "NOCODE": Remove insertion codes, preserve gaps for missing residues
            inplace (bool, optional): Whether to modify current object or create new:
                - False: Return new Protein object, preserve original (default)
                - True: Modify current object and return it

        Returns:
            tuple: (renumbered_protein, original_numbering) where:
                - renumbered_protein: Protein object with new numbering
                - original_numbering: NumPy array of original residue indices

        Renumbering Schemes Explained:
            RESTART (most common):
                Original: A15-A20, B100-B105
                Result:   A1-A6,   B1-B6

            CONTINUE:
                Original: A15-A20, B100-B105
                Result:   A1-A6,   B7-B12

            GAP200 (AlphaFold style):
                Original: A15-A20, B100-B105
                Result:   A1-A6,   B201-B206

            NOCODE:
                Original: A10, A10A, A10B, A11
                Result:   A10, A11,  A12,  A13 (removes insertion codes)

        Examples:
            >>> p = Protein("5cil")  # Complex numbering with gaps
            >>> print("Original chain P residues:", p.rs("P").name()[:5])
            ['671', '672', '673', '674', '675']

            # Standard restart numbering (most common)
            >>> p_restart, old_nums = p.renumber("RESTART")
            >>> print("Renumbered chain P residues:", p_restart.rs("P").name()[:5])
            ['1', '2', '3', '4', '5']

            # Continuous numbering across chains
            >>> p_cont, _ = p.renumber("CONTINUE")
            >>> print("Last H chain residue:", p_cont.rs("H").name()[-1])
            >>> print("First L chain residue:", p_cont.rs("L").name()[0])

            # AlphaFold-style numbering with 200-residue gaps
            >>> p_gap, _ = p.renumber("GAP200")
            >>> chains = p_gap.chain_id()
            >>> for i, chain in enumerate(chains):
            ...     first_res = p_gap.rs(chain).name()[0]
            ...     print(f"Chain {chain} starts at residue {first_res}")

            # Remove insertion codes but preserve gaps
            >>> p_clean, _ = p.renumber("NOCODE")
            >>> # Converts 95A, 95B, 95C to 95, 96, 97

        Gap Detection and Preservation:
            The method intelligently handles missing residues:
            - Detects real gaps using CA-CA distances
            - Preserves biologically meaningful gaps
            - Removes artificial gaps from renumbering artifacts
            - Uses MIN_GAP_DIST threshold (default 4.5 Å)

            >>> # Real gap (>4.5 Å): 95→100 becomes 1→6 (preserves 5-residue gap)
            >>> # Fake gap (<4.5 Å): 95→100 becomes 1→2 (removes artificial gap)

        Structure Prediction Applications:
            AlphaFold Input:
                >>> p_af, _ = p.renumber("GAP200")  # Standard AlphaFold format
                >>> sequence = p_af.seq(gap="G")    # Replace gaps with glycine

            ColabFold Input:
                >>> p_cf, _ = p.renumber("CONTINUE") # Continuous numbering
                >>> contig = p_cf.rs("ALL").contig_mpnn()  # For ProteinMPNN

        Visualization Applications:
            PyMOL Spectrum Coloring:
                >>> p_vis, _ = p.renumber("RESTART")
                >>> p_vis.save("renumbered.pdb")
                >>> # In PyMOL: spectrum count, rainbow

            Chain Alignment:
                >>> # Align chains with consistent numbering
                >>> p1, _ = structure1.renumber("RESTART")
                >>> p2, _ = structure2.renumber("RESTART")
                >>> rmsd = p1.rmsd(p2, "A", "A", align=True)

        Advanced Usage:
            Custom Gap Sizes:
                >>> p_gap50, _ = p.renumber("GAP50")   # 50-residue gaps
                >>> p_gap10, _ = p.renumber("GAP10")   # 10-residue gaps

            Preserving Original for Comparison:
                >>> p_new, original_nums = p.renumber("RESTART")
                >>> # Map back to original numbering if needed
                >>> for i, orig_num in enumerate(original_nums):
                ...     current_num = p_new.data.residue_index[i]
                ...     print(f"{current_num} was originally {orig_num}")

        Error Handling and Validation:
            The method includes validation for:
            - Continuous chain integrity
            - Distance-based gap validation
            - Insertion code handling
            - Negative residue numbers (some PDB files)

        Note:
            Renumbering is essential for structure prediction pipelines and
            standardized analysis. The "RESTART" scheme is most commonly used
            for general purposes, while "GAP200" matches AlphaFold conventions.
            Always verify that gaps represent real missing residues rather than
            crystallographic artifacts before structure prediction.
        """
        obj = self if inplace else self.clone()
        old_num=obj._renumber(renumber)
        return (obj, old_num)

    def _renumber(self, renumber=None):
        """Renumber residues in the protein structure.

        Args:
            renumber: Renumbering scheme. Can be None, "GAP" followed by gap size, or other schemes.

        Returns:
            None: Modifies the protein structure in place.
        """
        if renumber is None: return
        c_pos=self.chain_pos()
        chains=sorted([ (k,v[0],v[1]) for k,v in c_pos.items()], key=lambda x: x[1])
        out=[]
        base=gap=0
        old_num=np.copy(self.data.residue_index)
        if renumber.startswith("GAP"):
            gap=int(renumber.replace("GAP",''))
        for i,(k,b,e) in enumerate(chains):
            # we try to preserve the insertion code and missing residues
            idx,code=self.split_residue_index(self.data.residue_index[b:e+1])
            if renumber==Protein.RENUMBER['NOCODE']:
                # we still want to preserve the missing residues
                new_idx=[1]
                for j,x in enumerate(idx[1:]):
                    # if no missing, gap is 0 (has code) or 1, if missing, >1
                    step=max(idx[j+1]-idx[j],1)
                    if step>1: # check if there is a real gap
                        d=self.ca_dist(b+j+1, b+j)
                        if d<Protein.MIN_GAP_DIST:
                            if Protein.debug():
                                print(f"WARNING> remove numbering gap, dist={d:.2f} between {self.rs(np.array([b+j, b+j+1]))}")
                            step=1
                        else:
                            if Protein.debug():
                                print(f"INFO> Keep numbering gap, dist={d:.2f} between {self.rs(np.array([b+j, b+j+1]))}")
                    new_idx.append(new_idx[-1]+step)
                idx=np.array(new_idx)
                code=['']*len(idx)
            else:
                idx=idx-idx[0]+1 # change to start from 1
                # remove fake gaps
                for j,x in enumerate(idx[1:]):
                    step=idx[j+1]-idx[j]
                    if step>1:
                        d=self.ca_dist(b+j+1, b+j)
                        if d<Protein.MIN_GAP_DIST: # fake gap
                            idx[j+1:]-=step-1 # rename to remove gap
                            if Protein.debug():
                                print(f"WARNING> remove numbering gap, dist={d:.2f} between {self.rs(np.array([b+j, b+j+1]))}")
                        else:
                            if Protein.debug():
                                print(f"INFO> Keep numbering gap, dist={d:.2f} between {self.rs(np.array([b+j, b+j+1]))}")
            n_cnt=idx[-1]-idx[0]+1 # including missing residues
            if renumber == Protein.RENUMBER['RESTART']:
                self.data.residue_index[b:e+1]=self.merge_residue_index(idx, code)
            else: # continue or gap200
                self.data.residue_index[b:e+1]=self.merge_residue_index(idx+base, code)
            base+=n_cnt+gap # gap=0 should be equivalent to CONTINUE
        self._make_res_map()
        return old_num

    def split_chains(self, c_pos: Dict[str, Tuple[int, int]], renumber: Optional[str] = None, inplace: bool = False) -> 'Protein':
        """Split chains using position mapping. Takes chain positions and undoes merge_chains()."""
        obj=self if inplace else self.clone()
        chains=sorted([ (k,v[0],v[1]) for k,v in c_pos.items()], key=lambda x: x[1])
        #obj=self if inplace else self.clone()
        obj.data.chain_id=[x[0] for x in chains]
        for i,(k,b,e) in enumerate(chains):
            obj.data.chain_index[b:e+1]=i
        rs=np.array([ x for k,v in c_pos.items() for x in range(v[0], v[1]+1) ])
        obj=obj.extract(rs, inplace=True)
        obj._renumber(renumber)
        return obj

    def reorder_chains(self, chains: Union[List[str], np.ndarray], renumber: str = 'RESTART', inplace: bool = False) -> 'Protein':
        """Reorder chains and optionally renumber residues."""
        # Convert numpy array to list if needed
        if isinstance(chains, np.ndarray):
            chains = chains.tolist()

        assert set(chains)==set(self.chain_id())
        c_pos=self.chain_pos()
        obj=self if inplace else self.clone()

        idx=[]
        for k in chains:
            b,e=c_pos[k]
            idx.extend(list(range(b,e+1)))
        obj.data.aatype[:]=obj.data.aatype[idx]
        obj.data.atom_positions[:]=obj.data.atom_positions[idx]
        obj.data.atom_mask[:]=obj.data.atom_mask[idx]
        obj.data.residue_index[:]=obj.data.residue_index[idx]
        obj.data.b_factors[:]=obj.data.b_factors[idx]
        obj.data.chain_index[:]=obj.data.chain_index[idx]
        obj.data.chain_id[:]=np.array(chains)
        base=0
        for i,k in enumerate(chains):
            b,e=c_pos[k]
            n=e-b+1
            obj.data.chain_index[base:base+n]=i
            base+=n
        # _renumber will recreate res_map
        #obj._make_res_map()
        obj._renumber(renumber)
        return obj

    def extract_by_contig(self, rs: ContigType, ats: AtomSelectionType = None, as_rl: bool = False, inplace: bool = False) -> 'Protein':
        util.warn_msg('extract_by_contig will be depreciated, please use extract() instead.')
        return self.extract(rs, ats=ats, as_rl=as_rl, inplace=inplace)

    def extract(self, rs: ContigType, ats: AtomSelectionType = None, as_rl: bool = True, inplace: bool = False) -> 'Protein':
        """
        Extract a subset of residues and/or atoms from the protein structure.

        This method creates a new protein containing only the specified residues
        and atoms. It's commonly used to isolate chains, domains, or binding sites
        for focused analysis.

        Args:
            rs (str, RS, list): Residue selection using contig syntax or any data type that
                can be used to initialize a RS object:
                - String: "H:L" (chains), "H26-33" (fragment), "H26-33,51-57" (multiple)
                - RS object: Existing residue selection object
                - RL object: Existing residue list object
                rs is cast to RS or RL controlled by as_rl = True or False, respectively
            ats (str, ATS, list, optional): Atom selection. Default is all atoms:
                - String: "N,CA,C,O" (backbone), "CA" (alpha carbons)
                - ATS object: Existing atom selection
                - None: All atoms (default)
            as_rl (bool): Treatment of rs parameter:
                - True: cast rs to RL, order of residues is preserved (default)
                - False: cast rs to RS, order of residues is sorted
            inplace (bool): Modify current object (True) vs return new object (False)

        Returns:
            Protein:
                - If inplace=False: New Protein object with extracted subset
                - If inplace=True: modifies and returns the current object

        Common Use Cases:
            Antibody analysis: extract("H:L") for heavy and light chains
            CDR extraction: extract("H26-33,51-57,93-102") for all heavy CDRs
            Interface analysis: extract antigen and antibody separately
            Backbone analysis: extract("H:L", ats="N,CA,C,O")
            Domain isolation: extract specific regions for focused study


        Syntax of contig:
            "A" for the whole A chain, ":" separate chains, range "A2-5,10-13" (residue by residue index)
            "A:B1-10", "A-20,30-:B" (A from beginning to number 20, 30 to end, then whole B chain)

        Examples:
            >>> p = Protein("5cil.pdb")  # Antibody-antigen complex

            # Extract antibody chains only
            >>> ab = p.extract("H:L")
            >>> print(ab.chain_id())                    # ['H', 'L']

            # Extract antigen chain
            >>> ag = p.extract("P")
            >>> print(ag.chain_id())                    # ['P']

            # Extract heavy chain CDRs
            >>> cdrs = p.extract("H26-33,51-57,93-102")

            # Extract alpha carbons only for RMSD calculation
            >>> ca_only = p.extract("H:L", ats="CA")

            # In-place modification
            >>> p.extract("H:L", inplace=True)          # p now contains only H:L

            # Backbone atoms for secondary structure analysis
            >>> backbone = p.extract("ALL", ats="N,CA,C,O")

        Note:
            - Residues must be contiguous within each chain when using contig syntax
            - Chain ordering in the contig determines the output chain arrangement (when as_rl=True)
            - Use inplace=True carefully as it permanently modifies the protein object

        """
        def validate_rl(rl):
            """Validate residue list to ensure no duplicates.

            Args:
                rl: Residue list to validate.

            Returns:
                None: Raises error if duplicates found.
            """
            # make sure no duplicates
            t=pd.DataFrame({"resi":rl.data})
            t['chain']=rl.chain()
            if len(t.resi.unique())!=len(t):
                v,cnt=np.unique(t.resi.values, return_counts=True)
                v=v[cnt>1]
                S=self.rl(v).unique_name()
                raise Exception(f"There are duplicate residues: {S} in {rs}")
            # make sure chains are not fragmented
            chain=t.chain.values
            chain=np.array([ x for i,x in enumerate(t.chain) if i==0 or x!=chain[i-1] ])
            if len(t.chain.unique())!=len(chain):
                v,cnt=np.unique(chain, return_counts=True)
                v=v[cnt>1]
                raise Exception(f"Residues for chain {v} are not grouped together.")
            # make sure residues within a chain are sorted
            for k,t_v in t.groupby('chain'):
                if not np.array_equal(t_v.resi.values, np.unique(t_v.resi.values)):
                    raise Exception(f"Residues within chain {k} are not sorted in order.")

        if as_rl:
            rs=RL(self, rs)
            validate_rl(rs)
        else:
            rs=RS(self, rs)
        ats=ATS(ats)
        obj=self if inplace else self.clone()
        chains=obj.chain_id()
        if len(rs)==0:
            obj.data=Protein().data
        else:
            obj.data.aatype=obj.data.aatype[rs.data]
            obj.data.atom_positions=obj.data.atom_positions[rs.data]
            obj.data.atom_mask=obj.data.atom_mask[rs.data]
            obj.data.residue_index=obj.data.residue_index[rs.data]
            obj.data.b_factors=obj.data.b_factors[rs.data]
            obj.data.chain_index=obj.data.chain_index[rs.data]
        if ats.not_full():
            notats=(~ats).data
            if len(notats):
                obj.data.atom_mask[:, notats]=0
        # renumber chain index and chain name
        chain_index=util.unique2(obj.data.chain_index)
        c_map={x:i for i,x in enumerate(chain_index) }
        obj.data.chain_id=np.array([chains[x] for x in chain_index])
        obj.data.chain_index=np.array([c_map[x] for x in obj.data.chain_index])
        obj._make_res_map()
        return obj

    def fetch_pdb(self, code: str, remove_file: bool = True, assembly1: bool = False) -> Optional[str]:
        """Fetch and load a protein structure from the Protein Data Bank.

        Downloads PDB or mmCIF files from RCSB PDB servers and loads the structure
        into the current Protein object. Supports both asymmetric units and
        biological assemblies.

        Args:
            code (str): 4-character PDB identification code (case-insensitive).
                Examples: "1crn", "5cil", "7epp"
            remove_file (bool, optional): Whether to delete the downloaded file after
                loading. Set to False to keep local copy. Defaults to True.
            assembly1 (bool, optional): Whether to download biological assembly 1
                instead of asymmetric unit. Biological assemblies represent the
                functional form of the protein complex. Defaults to False.

        Returns:
            str or None: If remove_file=False, returns the path to downloaded file.
            Otherwise returns None. Returns None if download fails.

        Examples:
            # Basic PDB fetch (most common usage)
            >>> p = Protein()
            >>> p.fetch_pdb("1crn")
            >>> print(p.chain_id())  # ['A']

            # Fetch antibody-antigen complex
            >>> p = Protein()
            >>> p.fetch_pdb("5cil")
            >>> print(p.summary())  # Shows H, L, P chains

            # Keep downloaded file for inspection
            >>> p = Protein()
            >>> filename = p.fetch_pdb("1crn", remove_file=False)
            >>> print(f"Downloaded to: {filename}")  # 1crn.cif or 1crn.pdb

            # Load biological assembly (functional oligomer)
            >>> p = Protein()
            >>> p.fetch_pdb("7epp", assembly1=True)  # Gets biological unit

            # AlphaFold model from UniProt accession
            >>> p = Protein()
            >>> p.fetch_pdb("AF_Q2M403F1")  # AlphaFold model

        Note:
            - mmCIF format (.cif) is used in the download
            - Requires internet connection to RCSB PDB servers
        """
        fn=cldpdb.get_pdb(code, assembly1=assembly1)
        if fn is None: return fn
        if fn.lower().endswith(".cif"):
            self.from_cif(fn)
        else:
            self.from_pdb(fn)
        if remove_file:
            os.remove(fn)
        else:
            return fn

    @staticmethod
    def get_pdb_info(code: str) -> Dict[str, Any]:
        """Get PDB metadata including release date, resolution, method, and sequences."""
        return pdbinfo.get_pdb_info(code)

    def __len__(self) -> int:
        return len(self.data.aatype)

    def len_dict(self) -> Dict[str, int]:
        """Get dictionary mapping chain IDs to their lengths."""
        c,n=np.unique(self.data.chain_index, return_counts=True)
        chains=self.chain_id()
        return {chains[k]:v for k,v in zip(c,n)}

    @staticmethod
    def create_from_file(fn: Union[str, pathlib.Path]) -> 'Protein':
        """Create Protein object from PDB or CIF file."""
        if str(fn).lower().endswith('.cif'):
            return Protein().from_cif(fn)
        else:
            return Protein().from_pdb(fn)

    @staticmethod
    def guess_format(fn: Union[str, pathlib.Path]) -> str:
        """Guess if a file is PDB or CIF format."""
        ext=os.path.splitext(fn)[1]
        if ext in ('.gz'):
            ext=os.path.splitext(fn[:-3])[1]
        if ext in ('.pdb','.pdb1'): return "pdb"
        if ext in ('.cif'): return "cif"
        S=util.read_list(fn)
        pat_loop=re.compile(r'^(loop_|data_|_atom)')
        pat_key=re.compile(r'^(HEADER|TITLE|KEYWDS|REMARK|MODEL|END) ')
        for s in S:
            if pat_key.search(s): return 'pdb'
            if pat_loop.search(s): return "cif"
        return ""

    def from_pdb(self, fn: Union[str, pathlib.Path], chains: Optional[List[str]] = None, model: Optional[int] = None) -> 'Protein':
        #self._set_data(cldprt.pdb_to_string(str(fn), chains, models))
        structure=afprt.pdb2structure(str(fn))
        return self.from_biopython(structure, model=model, chains=chains)

    def to_pdb_str(self) -> str:
        """Convert protein structure to PDB format string."""
        data_str=afprt.to_pdb(self.data)
        return data_str

    def from_cif(self, fn: Union[str, pathlib.Path], model: Optional[int] = None, chains: Optional[List[str]] = None) -> 'Protein':
        """Load protein structure from mmCIF file."""
        parser = MMCIFParser(QUIET=True)
        from Bio.PDB.PDBExceptions import PDBConstructionWarning
        import warnings
        # Suppress the PDBConstructionWarning
        warnings.simplefilter('ignore', PDBConstructionWarning)
        fn=str(fn)
        ext=os.path.splitext(fn)[1]
        if ext in ('.gz'):
            import gzip
            with gzip.open(fn, 'rt') as f:
                structure=parser.get_structure("none", f)
        else:
            structure=parser.get_structure("", fn)
        return self.from_biopython(structure, model=model, chains=chains)

    def from_biopython(self, structure: Structure.Structure, model: Optional[int] = None, chains: Optional[List[str]] = None) -> 'Protein':
        """Convert BioPython structure object to Protein object."""
        # code taken from Bio/PDB/PDBIO.py
        # avoid converting to PDB to support huge strutures
        self._set_data(afprt.Protein.from_biopython(structure, model=model, chains=chains))
        return self

    @staticmethod
    def chain_label2id(fn_cif: Union[str, pathlib.Path]) -> Dict[str, str]:
        """Get chain labels from a CIF file, return a dict of {chain_label: chain_id}
        chain_id is assigned by the author, label is generated by PDB.
        When there are multiple symmetric units, the system can have multiple chain labels
        but the same chain_id, e.g., 7st5 has 3 chains B/H/O all mapped to chain_id L."""
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("none", fn_cif)
        chain_labels = {}
        mmcif_dict = parser._mmcif_dict
        first_model = next(structure.get_models())
        model_chain_ids = {chain.id for chain in first_model.get_chains()}
        # Extract mapping between label_asym_id and auth_asym_id
        label2id = {}
        if "_atom_site.label_asym_id" in mmcif_dict and "_atom_site.auth_asym_id" in mmcif_dict:
            label_ids = mmcif_dict["_atom_site.label_asym_id"]
            auth_ids = mmcif_dict["_atom_site.auth_asym_id"]
            for label, auth in zip(label_ids, auth_ids):
                if auth in model_chain_ids:  # Only include chains present in the model
                    label2id[label] = auth  # This may overwrite duplicates
        return label2id

    @staticmethod
    def assembly_units(fn_cif: Union[str, pathlib.Path]) -> Dict[str, List[str]]:
        """Get the mapping between individual assembly unit and their chains"""
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        mmcif_dict = MMCIF2Dict(fn_cif)
        # Get assembly info
        assemblies = mmcif_dict["_pdbx_struct_assembly.id"]
        assembly_gen = mmcif_dict["_pdbx_struct_assembly_gen.asym_id_list"]

        label2id = Protein.chain_label2id(fn_cif)
        # This gives you chain groupings per assembly
        assembly_units={}
        for asm_id, chains in zip(assemblies, assembly_gen):
            #print(asm_id, chains)
            assembly_units[asm_id] = util.unique2([label2id[x] for x in chains.split(",")])  # Split by comma to get individual chains
        return assembly_units

    @staticmethod
    def fix_pymol_pdb(fn: Union[str, pathlib.Path]) -> None:
        """Save PDB in PyMOL ends with 'END' instead of 'END   ', which then triggers a warning in BioPython get_structure()"""
        S=util.read_list(fn)
        for i in range(len(S)-1, 0, -1):
            if S[i]=='END':
                S[i]='END   '
                break
        util.save_list(fn, S, s_end="\n")

    def to_biopython(self, add_hydrogen: bool = False) -> Structure.Structure:
        """Convert protein structure to BioPython Structure object."""
        tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=True, suffix=".pdb").name
        self.save(tmp)
        # add hydrogen with PyMOL
        if add_hydrogen:
            check_PyMOL()
            pm=self.PyMOL()
            pm.run(f"load {tmp}, myobj; h_add myobj; save {tmp}")
            pm.close()
            Protein.fix_pymol_pdb(tmp)
            util.unix(f"cp {tmp} tx.pdb")
        # Load the PDB structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("mypdb", tmp)
        os.remove(tmp)
        return structure

    def from_sequence(self, seq: Union[str, Dict[str, str]]) -> 'Protein':
        """Build protein structure from sequence string or dictionary."""
        if type(seq) is str:
            S=seq.upper().split(":")
            seq={afprt.PDB_CHAIN_IDS[i] :x for i,x in enumerate(S)}
        check_PyMOL()
        pm=self.PyMOL()
        objs=[]

        def noXpos(s):
            """Get positions for non-X residues in sequence.

            Args:
                s: Sequence string.

            Returns:
                numpy.ndarray: Array of positions where residues are not 'X'.
            """
            # positions for non-X residues
            return np.array([ i for i,x in enumerate(s) if x!='X' ], dtype=int)

        # add a gap for "X+"
        seq_G={}
        rl_G={}
        seq_resn={}
        for k,v in seq.items():
            # resn numbering
            seq_resn[k]=noXpos(v)+1
            rl_G[k]=noXpos(re.sub(r'X+', 'X', v))
            seq_G[k]=re.sub(r'X+', 'G', v)

        for k,v in seq_G.items():
            pm.run(f"fab {k}/1/ {v}, obj_{k}")
            dist = np.random.rand(3)*20
            pm.run(f"translate [{dist[0]}, {dist[1]}, {dist[2]}], obj_{k}")
            objs.append(f"obj_{k}")
        pm.run("create obj_comb, "+ " or ".join(objs))
        tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".pdb").name
        pm.run(f"save {tmp}, obj_comb")
        pm.close()
        p=Protein(tmp)
        # remove inserted G, renumber residues to observe the gaps
        c_pos=p.chain_pos()
        rl=p.rl('NONE')
        for k,v in rl_G.items():
            rl+=rl_G[k]+c_pos[k][0]
        p=p.extract(rl)
        for k,v in seq_resn.items():
            p.resn(seq_resn[k], rl=k)
        os.remove(tmp)
        return p

    def html(self, show_sidechains: bool = False, show_mainchains: bool = False, color: str = "chain",
             style: str = "cartoon", width: int = 320, height: int = 320) -> Any:
        """Generate HTML for 3D visualization in Jupyter notebooks."""
        check_Mol3D()
        obj=Mol3D()
        obj = obj.show(self.to_pdb_str(), show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color, style=style, width=width, height=height)
        return obj.show(html=True)

    def show(self, show_sidechains: bool = False, show_mainchains: bool = False, color: str = "chain",
             style: str = "cartoon", width: int = 320, height: int = 320) -> Any:
        """Display interactive 3D visualization of the protein structure in Jupyter.

        Args:
            show_sidechains (bool, optional): If True, displays side chains. Defaults to False.
            show_mainchains (bool, optional): If True, displays main chains. Defaults to False.
            color (str, optional): Coloring scheme - "chain", "b" (b-factors), etc.
                Defaults to "chain".
            style (str, optional): Display style - "cartoon", "stick", "sphere", etc.
                Defaults to "cartoon".
            width (int, optional): Viewer width in pixels. Defaults to 320.
            height (int, optional): Viewer height in pixels. Defaults to 320.

        Examples:
            # Basic visualization
            >>> p = Protein("1crn")
            >>> p.show()

            # Show with side chains
            >>> p.show(show_sidechains=True)

            # Larger viewer with different style
            >>> p.show(style="stick", width=640, height=480)

            # Color by B-factors
            >>> p.show(color="b")

        Note:
            Requires Py3DMol and IPython. Only works in Jupyter notebook environments.
            Uses html() method internally for rendering.

        Note: To show multiple protein objects, use MOL3D class for chaining:

        from afpdb.mol3D import MOL3D
        PyMOL3D().show(experimental_structure, color="chain", style="cartoon")
            .show(predicted_structure, color="b", style="surface")
            .show(output="comparison.pse", width=600, height=600))

        The chaining ends when we no longer provide a protein object.
        """
        try:
            import IPython.display
            _has_IPython = True
        except ImportError:
            _has_IPython = False
        check_Mol3D()
        obj=Mol3D()
        if color=="b":
            # make sure b_factors are within [0, 1]
            p=self.clone()
            p.data.b_factors=p.data.b_factors.clip(0, 1)
        else:
            p=self
        obj.show(p.to_pdb_str(), show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color, style=style, width=width, height=height)
        return obj.show()

    def show_pymol(self, show_sidechains: bool = False, show_mainchains: bool = False, color: str = "chain",
                   style: str = "cartoon", output: Optional[str] = None, save_png: bool = True,
                   width: int = 480, height: int = 480) -> None:
        """Display protein structure in PyMOL using fluent interface (shortcut method).

        This is a convenient shortcut that uses the new fluent interface internally.
        Equivalent to: PyMOL3D().show(self, ...).show(output=...)

        Args:
            show_sidechains (bool): Show side chains as sticks.
            show_mainchains (bool): Show backbone as sticks.
            color (str): Coloring scheme ("chain", "b", "spectrum", "ss", or color name).
            style (str): Display representation ("cartoon", "stick", "sphere", "line", "surface").
            output (str): Output file path for PyMOL session (.pse).
            save_png (bool): Whether to also save PNG image.
            width (int): PNG image width in pixels.
            height (int): PNG image height in pixels.

        Returns:
            str: Path to the generated PyMOL session file

        Examples:
            >>> p = Protein("5cil.pdb")
            >>> session_file = p.show_pymol()  # Creates "5cil.pse" and "5cil.png"
            >>> p.show_pymol(color="b", output="confidence.pse", width=1600)

        Note: To show multiple protein objects, use PyMOL3D class for chaining:

        from afpdb.pymol3D import PyMOL3D
        PyMOL3D().show(experimental_structure, color="chain", style="cartoon")
            .show(predicted_structure, color="b", style="surface")
            .show(output="comparison.pse", width=600, height=600))

        The chaining ends when we no longer provide a protein object.
        """
        check_PyMOL()
        try:
            from .pymol3D import PyMOL3D
        except ImportError:
            raise ImportError("PyMOL3D not available. Please check pymol3D.py file.")

        # Generate default output name if not provided
        if output is None:
            output = f"protein_visualization.pse"

        # Use fluent interface: create PyMOL3D, show protein, then finalize
        PyMOL3D().show(self, show_sidechains=show_sidechains,
                        show_mainchains=show_mainchains, color=color,
                        style=style, output=output, save_png=save_png,
                        width=width, height=height).show()

    def sasa(self, rs_chain: ContigType = None, add_hydrogen: bool = True) -> pd.DataFrame:
        """Calculate solvent-accessible surface area for residues."""
        obj=self.extract(rs_chain)
        structure=obj.to_biopython(add_hydrogen=add_hydrogen)
        from Bio.PDB.SASA import ShrakeRupley
        sasa = ShrakeRupley()
        sasa.compute(structure, level="R")
        out=[]
        for chain in structure.get_chains():
            for residue in chain.get_residues():
                resid=residue.get_id()
                out.append([chain.id, f"{resid[1]}{resid[2].strip()}", int(resid[1]), residue.sasa])
        t=pd.DataFrame(data=out, columns=["chain","resn","resn_i","SASA"])
        t['resi']=t.apply(lambda r: self.res_map[r['chain']+r['resn']], axis=1)
        t['aa']=RL(self, t.resi.values).aa()

        # add relative sasa
        # where the max sasa is defined as Miller et al. 1987 in
        # https://arxiv.org/pdf/1211.4251
        max_sasa={
            'A':113.0, 'R':241.0, 'N':158.0, 'D':151.0, 'C':140.0,
            'E':183.0, 'Q':189.0, 'G':85.0, 'H':194.0, 'I':182.0,
            'L':180.0, 'K':211.0, 'M':204.0, 'F':218.0, 'P':143.0,
            'S':122.0, 'T':146.0, 'W':259.0, 'Y':229.0, 'V':160.0
        }
        t['rSASA']=t.SASA.values/t.aa.map(max_sasa)
        t=t[['chain','resn','resn_i','resi','aa','SASA','rSASA']]
        #t.display()
        return t

    def dsasa(self, rs_chain_a: ContigType, rs_chain_b: ContigType, add_hydrogen: bool = True) -> pd.DataFrame:
        """Classify residues based on SASA, according to:
            Levy ED, A Simple Definition of Structural Regions in Proteins and
            Its Use in Analyzing Interface Evolution
            https://doi.org/10.1016/j.jmb.2010.09.028

        rs_chain_a and rs_chain_b are chains designating the two components that
        defines the interface. E.g. for an Ab-Ag complex with chains H:L:G
        (G for antigen)

        p.dsasa("H", "L") study the interface between H and L within Ab
        p.dsasa("H:L", "G") study the interface between Ab and Ag

        Return dataframe
            rSASAm: relative accessible area for monomers
            rSASAc: relative accessible area for the complex
            dSASA: rSASAm-rSASAc
            label: classification labels
                core: key interacting residues
                support: residue in the interface that are not solvent accessible
                rim: around the edge of the binding interface
                surface: surface residues not involved in the interaction
                interior: interior residues not exposed to solvent
        """
        rs_a=self.rs(rs_chain_a)
        rs_b=self.rs(rs_chain_b)
        if len(rs_a & rs_b)>0:
            raise Exception("Two selections cannot overlap.")
        t_a=self.sasa(rs_a, add_hydrogen=add_hydrogen)
        t_b=self.sasa(rs_b, add_hydrogen=add_hydrogen)
        # monomer
        t_m=pd.concat([t_a, t_b], ignore_index=True)
        t_m.rename2({'rSASA':'rSASAm', 'SASA':'SASAm'})
        # complex
        t_c=self.sasa(rs_a | rs_b)
        t_c.rename2({'rSASA':'rSASAc', 'SASA':'SASAc'})
        t=t_m.merge(t_c, on=['chain','resi','resn','resn_i','aa'])
        t['drSASA']=t['SASAm']-t['SASAc']

        def label(r):
            """Classify residue based on SASA values according to Levy criteria.

            Args:
                r: Row from DataFrame with rSASAm and rSASAc values.

            Returns:
                str: Classification label ('core', 'support', 'rim', 'surface', 'interior').
            """
            rASAc=r['rSASAc']
            rASAm=r['rSASAm']
            drASA=r['drSASA']
            if drASA>0:
                if rASAm<0.25: return 'support'
                if rASAc<0.25: return 'core'
                return 'rim'
            else:
                if rASAc<0.25: return 'interior'
                return 'surface'

        t['label']=t.apply(label, axis=1)
        return t

    def save(self, filename: str, format: Optional[str] = None) -> None:
        """
        Save the protein structure to a file in PDB or mmCIF format.

        This method writes the complete protein structure including all atomic
        coordinates, B-factors, chain identifiers, and residue numbering to disk
        in standard structural biology formats.

        Args:
            filename (str): Output file path. Can be absolute or relative path.
                The file extension is used for format auto-detection if format
                parameter is not explicitly specified.
            format (str, optional): Explicit format specification:
                - None: Auto-detect from filename extension (default)
                - "pdb": Legacy PDB format (Protein Data Bank)
                - "cif" or "mmcif": Modern mmCIF format (recommended)

        Supported File Formats:
            PDB Format (.pdb):
                - Traditional column-based text format
                - Limited to 99,999 atoms and 9,999 residues
                - Widely compatible with visualization software
                - Good for smaller structures and quick inspection

            mmCIF Format (.cif):
                - Modern replacement for PDB format
                - No size limitations, supports large structures
                - More precise and complete metadata storage
                - Required for structures >99,999 atoms
                - Preferred for data archival and exchange

        Examples:
            >>> p = Protein("1crn")

            # Auto-detect format from file extension
            >>> p.save("output.pdb")        # Saves as PDB format
            >>> p.save("output.cif")        # Saves as mmCIF format

            # Explicit format specification overrides extension
            >>> p.save("structure.txt", format="pdb")    # PDB in .txt file
            >>> p.save("data", format="cif")             # mmCIF without extension

            # Save extracted substructure
            >>> antibody = p.extract("H:L")
            >>> antibody.save("antibody_only.pdb")

            # Save with renumbered residues
            >>> renumbered, old_nums = p.renumber("RESTART")
            >>> renumbered.save("renumbered_structure.cif")

        Note:
            For structures with >99,999 atoms or >9,999 residues, PDB format
            will cause data loss. Always use mmCIF format for large structures.
            The mmCIF format is also more robust for preserving metadata and
            supporting future extensions.
        """
        data_str=self.to_pdb_str()
        if format is None: # get it from the extension
            _, ext=os.path.splitext(filename)
            format=ext.replace('.', '').lower()
        if format=="pdb":
            util.save_string(filename, data_str)
        elif format=="cif" or filename.lower().endswith(".cif"):
            tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".pdb").name
            util.save_string(tmp, data_str)
            self.pdb2cif(tmp, filename)
            os.remove(tmp)

    def clone(self) -> 'Protein':
        """Create a deep copy of the protein structure.

        Creates an independent copy of the protein object with all data duplicated
        in memory. Changes to the cloned object will not affect the original and
        vice versa. This is essential for operations that modify structure data
        while preserving the original.

        Returns:
            Protein: A new Protein object with identical structure data

        Examples:
            >>> p = Protein("5cil.pdb")
            >>> p_copy = p.clone()
            >>> # Modifications to copy don't affect original
            >>> p_copy.mutate("A:10:ALA")
            >>> print("Original still has:", p.rs("A:10").aa()[0])
            >>> print("Copy now has:", p_copy.rs("A:10").aa()[0])

            # Useful for comparisons after modifications
            >>> original = p.clone()
            >>> p.renumber("RESTART", inplace=True)
            >>> rmsd = p.rmsd(original, align=True)
            >>> print(f"RMSD after renumbering: {rmsd:.3f} Å")

            # Chain extraction preserving original
            >>> full_protein = p.clone()
            >>> heavy_chain = p.rs("H").extract()  # Extract heavy chain only
            >>> print(f"Original chains: {full_protein.chain_list()}")
            >>> print(f"Extracted chain: {heavy_chain.chain_list()}")

        Memory Usage:
            The clone operation duplicates all structure data including:
            - Atomic coordinates (atom_positions array)
            - B-factors and occupancies
            - Chain and residue metadata
            - Sequence information
            - Atom masks and connectivity

            For large structures, this roughly doubles memory usage temporarily.

        Use Cases:
            Structural Modifications:
                >>> original = protein.clone()
                >>> protein.rs("100-200").extract()  # Remove residues 100-200
                >>> # Compare before/after

            Multiple Numbering Schemes:
                >>> p_restart = protein.clone().renumber("RESTART")[0]
                >>> p_gap200 = protein.clone().renumber("GAP200")[0]

            Comparative Analysis:
                >>> wild_type = protein.clone()
                >>> protein.mutate("A:25:PRO")  # Introduce proline kink
                >>> flexibility_change = compare_flexibility(wild_type, protein)

        Performance:
            - Fast for typical protein sizes (<1000 residues)
            - Scales linearly with number of atoms
            - Memory allocation is the main bottleneck

        Note:
            Use clone() whenever you need to preserve the original structure
            while performing modifications. This is safer than trying to undo
            operations and ensures data integrity in analysis pipelines.
        """
        return Protein(self.data.clone())

    @staticmethod
    def pdb2cif(pdb_file: Union[str, pathlib.Path], cif_file: Union[str, pathlib.Path]) -> None:
        """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
        # code copied from colabfold/batch.py
        parser = PDBParser(QUIET=True)
        code=os.path.splitext(os.path.basename(pdb_file))[0]
        structure = parser.get_structure(code, pdb_file)
        cif_io = CFMMCIFIO()
        cif_io.set_structure(structure)
        cif_io.save(str(cif_file))

    @staticmethod
    def cif2pdb(cif_file: Union[str, pathlib.Path], pdb_file: Union[str, pathlib.Path]) -> None:
        parser = MMCIFParser()
        structure = parser.get_structure("", cif_file)
        io=PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_file))

    @staticmethod
    def remove_hetero(pdb_file: Union[str, pathlib.Path], pdb_file2: Union[str, pathlib.Path]) -> None:
        # Remove hetatms
        p=Protein()
        p.from_pdb(pdb_file)
        p.save(pdb_file2)

    def fill_pos_with_ca(self, backbone_only: bool = False, cacl_cb: bool = False, inplace: bool = False) -> Optional['Protein']:
        """file missing atom positions by Ca
            if backbone_only, only assign positions to N, CA, C
            calc_cb, if True, also compute missing Cb position (for non-Gly, if N, CA, C were not missing)
        """
        obj=self if inplace else self.clone()
        CA=afres.atom_order['CA']
        CB=afres.atom_order['CB']
        GLY=afres.restype_order['G']
        for i,res in enumerate(obj.data.aatype):
            std_mask=afres.STANDARD_ATOM_MASK[res].copy()
            if backbone_only: std_mask[3:]=0
            if cacl_cb: std_mask[CB]=1
            obj.data.atom_positions[i, std_mask]=obj.data.atom_positions[i,CA] # 1 is the CA positon
            obj.data.b_factors[i, std_mask]=obj.data.b_factors[i,CA] # 1 is the CA positon
            # check CB
            if cacl_cb and (obj.data.atom_mask[i][CB]==0) and (np.all(obj.data.atom_mask[i][:3])):
                # N, CA, C were all present
                if obj.data.aatype[i]!=GLY:
                    pos=obj.data.atom_positions[i, :3]
                    obj.data.atom_positions[i, CB]=cldprt._np_get_cb(pos[0],pos[1],pos[2], use_jax=False)
            obj.data.atom_mask[i, std_mask]=1
        return obj

    def thread_sequence(self, seq: Union[str, Dict[str, str]], output: str, relax: int = 1, replace_X_with: str = '',
                       seq2bfactor: bool = True, amber_gpu: bool = False, cores: int = 1,
                       side_chain_pdb: Optional[str] = None, rl_from: Optional[ContigType] = None,
                       rl_to: Optional[ContigType] = None) -> 'Protein':
        tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".pdb").name
        self.save(tmp, format="pdb")
        data=Protein.thread_sequence2(tmp, seq, output, relax=relax, replace_X_with=replace_X_with, seq2bfactor=seq2bfactor, amber_gpu=amber_gpu, cores=cores, side_chain_pdb=side_chain_pdb, rl_from=rl_from, rl_to=rl_to)
        os.remove(tmp)
        return data

    @staticmethod
    def thread_sequence2(pdb: str, seq: Union[str, Dict[str, str]], output: str, relax: int = 1,
                        replace_X_with: str = '', seq2bfactor: bool = True, amber_gpu: bool = False,
                        cores: int = 1, side_chain_pdb: Optional[str] = None,
                        rl_from: Optional[ContigType] = None, rl_to: Optional[ContigType] = None) -> 'Protein':
        """seq: can be a string, multiple chains are ":"-separated
                ELTQSPATLSLSPGERATLSCRASQSVGRNLGWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQARLLLPQTFGQGTKVEIKRTV:EVQLLESGPGLLKPSETLSLTCTVSGGSMINYYWSWIRQPPGERPQWLGHIIYGGTTKYNPSLESRITISRDISKNQFSLRLNSVTAADTAIYYCARVAIGVSGFLNYYYYMDVWGSGTAVTVSS:WNWFDITNK
            We strongly recommend to use dictionary or JSON string
                {"A": "ELTQSPATLSLSPGERATLSCRASQSVGRNLGWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQARLLLPQTFGQGTKVEIKRTV", "B": "EVQLLESGPGLLKPSETLSLTCTVSGGSMINYYWSWIRQPPGERPQWLGHIIYGGTTKYNPSLESRITISRDISKNQFSLRLNSVTAADTAIYYCARVAIGVSGFLNYYYYMDVWGSGTAVTVSS", "C": "WNWFDITNK"}
            rl_from and rl_to specifies the RL of fixed residues in side_chain_pdb and input pdb, respectively.

        """
        from afpdb.thread_seq import ThreadSeq
        ts=ThreadSeq(pdb)
        data=ts.run(output, seq, relax=relax, replace_X_with=replace_X_with, seq2bfactor=seq2bfactor, amber_gpu=amber_gpu, cores=cores, side_chain_pdb=side_chain_pdb, rl_from=rl_from, rl_to=rl_to)
        return data

    def center(self) -> np.ndarray:
        """Get center of CA mass."""
        return np.mean(self.data.atom_positions[:,1], axis=0)

    def translate(self, v: np.ndarray, inplace: bool = False) -> 'Protein':
        """Move structure by vector v."""
        obj=self if inplace else self.clone()
        obj.data.atom_positions[obj.data.atom_mask>0]= \
            obj.data.atom_positions[obj.data.atom_mask>0]+v
        return obj

    def center_at(self, v: Optional[np.ndarray] = None, inplace: bool = False) -> 'Protein':
        """Position structure so that center is at vector v."""
        obj=self if inplace else self.clone()
        c=obj.center()
        if v is None: v=np.zeros(3)
        obj.translate(v-c, inplace=True)
        return obj

    def rotate(self, ax: np.ndarray, theta: float, inplace: bool = False) -> 'Protein':
        """Rotate protein around axis by angle in degrees."""
        #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
        theta=theta/180*np.pi
        obj=self if inplace else self.clone()
        ax=self._unit_vec(ax)
        K=np.array([
            [0.0, -ax[2], ax[1]],
            [ax[2], 0.0, -ax[0]],
            [-ax[1], ax[0], 0.0]
          ])
        R=np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
        obj.data.atom_positions[obj.data.atom_mask>0]= \
            obj.data.atom_positions[obj.data.atom_mask>0]@R.T
        return obj

    @staticmethod
    def rotate_a2b(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
        '''Rotate so that vector a will point at b direction'''
        a=Protein._unit_vec(a)
        b=Protein._unit_vec(b)
        ang=Protein._np_ang_acos(a, b)
        ax=np.cross(a, b)
        return (ax, ang)

    def reset_pos(self, inplace: bool = False) -> Optional['Protein']:
        '''Set center at origin, align the PCA axis along, Z, X and Y'''
        obj=self if inplace else self.clone()
        obj.center_at(inplace=True)
        # center of mass at origin

        #https://github.com/ljmartin/align/blob/main/0.2%20aligning%20principal%20moments%20of%20inertia.ipynb
        def calc_m_i(pcl):
            """
            Computes the moment of inertia tensor.
            a more convoluted but easier to understand alternative is in here:
            https://github.com/jwallen/ChemPy/blob/master/chempy/geometry.py
            """
            A = np.sum((pcl**2) * np.ones(pcl.shape[0])[:,None],0).sum()
            B = (np.ones(pcl.shape[0]) * pcl.T).dot(pcl)
            eye = np.eye(3)
            return A * eye - B

        def get_pmi(coords):
            """
            Calculate principal moment of inertia
            This code is a re-write of the function from MDAnalysis:
            https://github.com/MDAnalysis/mdanalysis/blob/34b327633bb2aa7ce07bbd507a336d1708871b6f/package/MDAnalysis/core/topologyattrs.py#L1014
            """
            momint = calc_m_i(coords)
            eigvals, eigvecs = np.linalg.eig(momint)

             # Sort
            indices = np.argsort(-eigvals) #sorting it returns the 'long' axis in index 0.
            # Return transposed which is more intuitive format
            return eigvecs[:, indices].T

        for i in range(2): # Rotate twice
            X=obj.data.atom_positions[:, 1] # CA
            pmi=get_pmi(X)
            # rotate major axis to Z, minor to X (in PyMOL, X goes out of the screen, Z at right, Y at Up)
            if i==0:
                b=np.array([1.,0.,0.])
                idx=2
            else:
                b=np.array([0.,0.,1.])
                idx=0
            ax, ang = Protein.rotate_a2b(pmi[idx], b)
            obj.rotate(ax, ang*180/np.pi, inplace=True)
        return obj
        #X=self.data.atom_positions[:, 1] # CA
        #mi=calc_m_i(X)
        #X=p.data.atom_positions[:,1]
        #print(np.min(X, axis=0), np.max(X, axis=0))
        #print(mi)

    def spin_to(self, phi: float, theta: float, inplace: bool = False) -> Optional['Protein']:
        """Assume the molecule is pointing at Z, rotate it to point at phi, theta,
            theta is the angle wrt Z, phi is the angle wrt to X,
            see https://math.stackexchange.com/questions/2247039/angles-of-a-known-3d-vector
        You should first center the molecule!
        """
        obj=self if inplace else self.clone()
        obj.rotate(np.array([0.,0.,1.]), phi, inplace=True)
        ax=np.array([np.sin(phi), -np.cos(phi), 0])
        obj.rotate(ax, theta, inplace=True)
        return obj

    @staticmethod
    def _unit_vec(v: np.ndarray) -> np.ndarray:
        return (v/cldprt._np_norm(v)+1e-8)

    @staticmethod
    def _np_ang_acos(a: np.ndarray, b: np.ndarray) -> float:
        '''compute the angle between two vectors'''
        cos_ang=np.clip((Protein._unit_vec(a)*Protein._unit_vec(b)).sum(), -1, 1)
        return np.arccos(cos_ang)

    def range(self, ax: np.ndarray, CA_only: bool = True) -> Tuple[float, float]:
        """Return (min, max) coordinates along the ax direction"""
        if CA_only:
            X=self.data.atom_positions[:, 1] # CA
        else:
            X=self.data.atom_positions[ self.data.atom_mask>0 ]
        ax=self._unit_vec(ax)
        Y=np.sum(X*ax, axis=1)
        return (np.min(Y), np.max(Y))

    @staticmethod
    def from_atom_positions(atom_positions: np.ndarray, atom_mask: Optional[np.ndarray] = None,
                          aatype: Optional[np.ndarray] = None, residue_index: Optional[np.ndarray] = None,
                          chain_index: Optional[np.ndarray] = None, b_factors: Optional[np.ndarray] = None) -> 'Protein':
        """Purely for visualization purpose, crease a strucutre from atom positions, guess atom_mask"""
        n,m=atom_positions.shape[:2]
        if atom_mask is None:
            atom_mask=np.any(atom_positions, axis=-1)
            atom_mask[:,:3]=1 # assume N, CA, C has positions
            atom_mask[:,36]=0 # remove possible OXT

        def match_res(mask, pos):
            """Match residue type based on atom mask pattern.

            Args:
                mask: Binary mask indicating which atoms are present.
                pos: Position index.

            Returns:
                str or None: Best matching residue type based on atom pattern.
            """
            fp=set(np.where(mask>0)[0])
            score=0.0
            aa=None
            for k,v in c_atoms.items():
                if fp==v:
                    return k
                else:
                    ab=len(fp & v)
                    s=ab/(len(fp)+len(v)-ab)
                    if s>score:
                        score=s
                        aa=[k, fp&v, fp-v, v-fp]
            for i in range(1,4):
                aa[i]=sorted([afres.atom_types[x] for x in aa[i]])
            print("Closest match: (shared, input, expect)", aa, score)
            raise ValueError(f'Invalid aatypes at position: {pos}')

        if aatype is None:
            # guess residue identity via mask
            c_atoms={}
            for res,atoms in afres.residue_atoms.items():
                c_atoms[afres.resname_to_idx[res]]=set([util.index(x, afres.atom_types) for x in atoms])

            aatype=[]
            for i in range(n):
                aatype.append(match_res(atom_mask[i], i))

        if residue_index is None:
            residue_index=np.array(range(1,n+1))
        if b_factors is None:
            b_factors=np.zeros((n,m))
        if chain_index is None:
            chain_index=np.zeros(n)

        prt=afprt.Protein(
            atom_positions=np.array(atom_positions),
            atom_mask=np.array(atom_mask),
            aatype=np.array(aatype),
            residue_index=np.array(residue_index),
            chain_index=np.array(chain_index),
            b_factors=b_factors)
        return Protein(prt)

    def seq_dict(self, keep_gap: bool = True, gap: str = "X") -> Dict[str, str]:
        """Get dictionary mapping chain IDs to their sequences."""
        chains=self.chain_list()
        seq=self.seq(keep_gap=keep_gap, gap=gap).split(":")
        return {k:v for k,v in zip(chains, seq)}

    def seq_case(self, rs_full: ContigType, rs_upper: ContigType, reverse=False) -> str:
        """We generate a formatted sequence string for all residues in rs_full,
            Those residues in rs_upper will be shown as upper case, the rest in lower case.
            This can be handy for highlighting CDRs in antibody.
            If reverse, we print rs_full in upper case and rs_upper in lower case
        """
        rs_upper=self.rs(rs_upper)
        rs_full=self.rs(rs_full)
        s_seq=self.extract(rs_full).seq()
        if reverse:
            idx_upper=[ i for i,x in enumerate(rs_full.data) if x not in rs_upper.data ]
        else:
            idx_upper=[ i for i,x in enumerate(rs_full.data) if x in rs_upper.data ]
        S=[]
        i=0
        for x in s_seq:
            if x in ('X',':'):
                S.append(x.upper() if reverse else x.lower())
            else:
                S.append(x.upper() if i in idx_upper else x.lower())
                i+=1
        return "".join(S)

    @staticmethod
    def seq_gap_by_seq(seq_x: str, seq_nogap: str, is_global: bool = False, match: int = 2,
                      xmatch: int = -1, gap_open: float = -1.5, gap_ext: float = -0.2,
                      gap_lower_case: bool = False) -> str:
        """We align the sequence with X, seq_x, to a sequence without gap (from get_pdb_info)
        Extract residues from seq_nogap to replace X in seq_x.
        This is useful to generate the no-gap sequence for structure prediction models

        The gap sequence are shown in lower case
        """
        _, pos1, pos2 = Protein.find_aligned_positions(seq_x, seq_nogap, is_global=is_global, match=match, xmatch=xmatch, gap_open=gap_open, gap_ext=gap_ext)
        print(_[0])
        out=[]
        inX=False
        seq_a=_[0][0]
        seq_a=re.sub(r'-(?=-*X)', 'X', seq_a) # make sure a gap is always started with an X
        seq_b=_[0][1]

        for a,b in zip(seq_a, seq_b):
            if not inX:
                if a=="-": continue # ignore unaligned piece
                if a=="X": inX=True # enter gap
            if a in ("X","-"): # must be in gap
                if b not in ("-","X"):
                    if gap_lower_case:
                        out.append(b.lower())
                    else:
                        out.append(b)
            else:
                if inX: inX=False # exit gap
                out.append(a)
        return ("".join(out))

    def seq_dict_gap_by_pdb(self, pdb: str, is_global: bool = False, match: int = 2,
                           xmatch: int = -1, gap_open: float = -1.5, gap_ext: float = -0.2,
                           gap_lower_case: bool = False) -> Dict[str, str]:
        """Return seq_dict, where gaps in seq were replaced residues found in the full sequence as provided by the PDB entry"""
        c_info=self.get_pdb_info(pdb)
        c_seq=self.seq_dict()
        for k,v in c_seq.items():
            if "X" not in v: continue
            if k in c_info.get('seq_dict', {}):
                try:
                    print(f"Fill gap for chain {k}")
                    new_seq=Protein.seq_gap_by_seq(v, c_info.get('seq_dict')[k], \
                        is_global=is_global, match=match, xmatch=xmatch, gap_open=gap_open, gap_ext=gap_ext, gap_lower_case=gap_lower_case)
                    c_seq[k]=new_seq
                except Exception as e:
                    print(e)
        return c_seq

    def seq_add_chain_break(self, seq: str) -> str:
        """Given a multichain sequence without ':', we add ':'"""
        chains=self.data.chain_index
        idx=np.where(chains[1:]!=chains[:-1])[0]+1
        S=[]
        for i,aa in enumerate(seq):
            if i in idx:
                S.append(":")
            S.append(aa)
        return "".join(S)

    def rename_chains(self, c_chains: Dict[str, str], inplace: bool = False) -> 'Protein':
        """c_chains is dict, key is old chain name, value is new chain name"""
        obj=self if inplace else self.clone()
        chains=obj.chain_id()
        # check if chain names remain unique after renaming
        new_chains=[c_chains.get(x, x) for x in chains]
        bad=[k for k,v in util.unique_count(new_chains).items() if v>1]
        if len(bad)>0:
            util.error_msg(f"Chain names will duplicate after renaming: {bad}!")
        obj.data.chain_id=new_chains
        obj._make_res_map()
        return obj

    def rename_reorder_chains(self, p: 'Protein', c_chains: Dict[str, str], inplace: bool = False) -> 'Protein':
        """Rename chain names, then reorder all chains in the same order as chains are arranged in object p.
            Object must have all the chains p contains after renaming, extra chains will be removed.

            At the end, object should share the same chain names as p and chains are in the exact same order.
            This way, the RS objects of associated with the two projects maybe Boolean combined.

            This is used to prepare input structures for dockQ scoring.
        """
        obj=self if inplace else self.clone()
        obj.rename_chains(c_chains, inplace=True)
        obj.extract(":".join(p.chain_id()), inplace=True)
        return obj

    def ca_dist(self, resi_a: int, resi_b: int) -> float:
        ca=ATS.i('CA')
        a=self.data.atom_positions[resi_a, ca]
        b=self.data.atom_positions[resi_b, ca]
        dist=np.linalg.norm(a - b)
        return dist

    def seq(self, keep_gap: bool = True, gap: str = "X") -> str:
        """Get the amino acid sequence of the protein structure.

        Extracts the primary sequence from the structural data, handling missing
        residues and multiple chains. The sequence represents the actual residues
        present in the coordinate file, not the theoretical full sequence.

        Args:
            keep_gap (bool, optional): Whether to insert gap characters for missing
                residues based on residue numbering discontinuities. When True,
                gaps indicate structurally missing regions. Defaults to True.
            gap (str, optional): Character to use for representing gaps. Defaults
                to "X" for AlphaFold compatibility. Common alternatives: "-", "G"

        Returns:
            str: Multi-chain amino acid sequence with chains separated by ":"
                 and gaps represented by the specified gap character.

        Sequence Features:
            - Uses single-letter amino acid codes (A, R, N, D, C, etc.)
            - Unknown residues represented as "X"
            - Chain boundaries marked with ":"
            - Missing residues filled with gap character when keep_gap=True
            - Preserves the order of chains as they appear in the structure

        Gap Detection Logic:
            When keep_gap=True, gaps are inserted based on:
            1. Residue numbering discontinuities (e.g., 95→100 inserts 4 gaps)
            2. CA-CA distance validation using MIN_GAP_DIST threshold
            3. Only "real" gaps (large distances) are kept as gaps
            4. Artifacts from renumbering are automatically removed

        Examples:
            >>> p = Protein("1crn")
            >>> seq = p.seq()
            >>> print(seq)
            'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN'

            # Multi-chain structure with gaps
            >>> p = Protein("5cil")
            >>> seq_with_gaps = p.seq()
            >>> print(seq_with_gaps[:50])  # First 50 characters
            'VQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEW...'

            # Remove gaps to get continuous sequence
            >>> seq_continuous = p.seq(keep_gap=False)
            >>> print(len(seq_with_gaps), len(seq_continuous))
            445 437  # Shows 8 missing residues

            # Use different gap character (for AlphaFold)
            >>> seq_for_af = p.seq(gap="G")  # Replace missing with Glycine
            >>> print("Missing residues replaced with G:", seq_for_af.count("G"))

            # Count chains and their lengths
            >>> chains = seq_with_gaps.split(":")
            >>> print(f"Structure has {len(chains)} chains")
            >>> for i, chain_seq in enumerate(chains):
            ...     print(f"Chain {i+1}: {len(chain_seq)} residues")

        Common Applications:
            Structure Prediction Input:
                >>> sequence = p.seq(gap="G")  # AlphaFold-compatible

            Sequence Analysis:
                >>> nogap_seq = p.seq(keep_gap=False)
                >>> identity = calculate_identity(nogap_seq, reference)

            Chain Extraction:
                >>> chain_sequences = p.seq().split(":")
                >>> heavy_chain = chain_sequences[0]  # First chain

        Technical Details:
            - Reads aatype array containing residue type indices
            - Converts indices to single-letter codes using afres.restypes_with_x
            - Detects chain breaks from chain_index discontinuities
            - Gap insertion uses spatial distance validation (MIN_GAP_DIST)
            - Handles insertion codes and negative residue numbers

        Note:
            The sequence reflects the actual structural content, not the theoretical
            protein sequence. Missing residues (common in crystal structures) are
            indicated by gaps when keep_gap=True. For structure prediction inputs,
            use gap="G" to replace missing regions with modelable residues.

            Sequence order follows the chain arrangement in the PDB file, which
            may not be the biological or alphabetical order.
        """
        aatype=self.data.aatype
        chains=self.data.chain_index
        # there are bad entries where resi is negative integer, 1txgg
        resi=[int(re.sub(r'\D+$', '', x)) for x in self.data.residue_index]
        # where chain breaks
        idx=np.where(chains[1:]!=chains[:-1])[0]+1
        S=[]
        for i,aa in enumerate(aatype):
            if i in idx:
                S.append(":")
            else:
                if keep_gap and i>0:
                    # check if gap based on CA-CA distance
                    if resi[i]-resi[i-1]>1:
                        d=self.ca_dist(i-1, i)
                        if d<Protein.MIN_GAP_DIST:
                            if Protein.debug():
                                print(f"WARNING> no gap, dist={d:.2f} between {self.rs(np.array([i-1, i]))}")
                        else:
                            for j in range(resi[i-1], resi[i]-1):
                                S.append(gap)
            S.append(afres.restypes_with_x[aa])
        return "".join(S)

    def summary(self) -> pd.DataFrame:
        """Summarize key structure information"""
        c_seq=self.seq_dict()
        chains=list(c_seq.keys())
        t=pd.DataFrame({"Chain":chains})
        t['Sequence']=t.Chain.map(c_seq)
        c_len=self.len_dict()
        t['Length']=t.Chain.map(c_len)
        t['#Missing Residues']=t.Sequence.apply(lambda s: len(s)-len(s.replace("X", "")))
        t['#Insertion Code']=t.Chain.apply(lambda k: sum([re.search(r'[A-Za-z]$', s) is not None for s in self.rs(k).name()]))
        c_pos=self.chain_pos()
        t['First Residue Name']=t.Chain.apply(lambda k: self.data.residue_index[c_pos[k][0]])
        t['Last Residue Name']=t.Chain.apply(lambda k: self.data.residue_index[c_pos[k][1]])
        miss_bb=util.unique_count(self.rs_missing_atoms(ats="N,CA,C,O").chain())
        t['#Missing Backbone']=t.Chain.apply(lambda k: miss_bb.get(k, 0))
        return t

    @staticmethod
    def copy_side_chain(to_pdb: str, src_pdb: str, rl_from: Optional[ContigType] = None,
                       rl_to: Optional[ContigType] = None) -> None:
        """
            RFDiffusion output PDB: to_pdb does not contain side chain atoms
            We would like to preserve side chains for the fixed positions
            This method is to copy side chain for fixed residues based on rs_from and rs_to
            This method is used in thread_seq.py --side_chain

            rl_from and rl_to are RL specifying where fixed residues are in src_pdb and to_pdb, respectively
        """
        b=Protein(src_pdb)
        seq_b=b.seq_dict(gap='')
        chains_b=b.chain_list()
        e=Protein(to_pdb)
        seq_e=e.seq_dict(gap='')
        chains_e=e.chain_list()
        rl_from=b.rl(rl_from)
        rl_to=e.rl(rl_to)
        if len(rl_from)==0:
            print("No conserved residues, no side chain to copy...")
            return e

        seq_from=rl_from.seq()
        seq_to=rl_to.seq()
        if seq_from != seq_to:
            raise Exception("Fixed sequences do not match:\nin source> {seq_from}\nin target> {seq_to}\n")

        b.align(e, rl_from, rl_to, ats="N,CA,C,O")
        print("RMSD after backbone alignment:", b.rmsd(e, rl_from, rl_to, ats="N,CA,C,O"))
        sel_idx_e=rl_to.data
        sel_idx_b=rl_from.data
        e.data.atom_positions[sel_idx_e,...]=b.data.atom_positions[sel_idx_b,...]
        e.data.atom_mask[sel_idx_e,...]=b.data.atom_mask[sel_idx_b,...]
        return e

    def rs(self, rs: Optional[ContigType] = None) -> 'RS':
        """
        Create a residue selection (RS) object for this protein.

        This method returns an RS (Residue Selection) object that can be used to
        select and manipulate specific residues based on contig syntax.

        Args:
            rs (str, RS, or None): Residue selection specification using contig syntax.
                - None or "ALL": Select all residues (default)
                - "NONE", "NULL", "": Select no residues
                - Contig syntax: e.g., "H:L", "H26-33", "P1-10:H:L"
                - RS object: Create a copy of an existing selection

        Returns:
            RS: Residue selection object containing the specified residues

        Contig Syntax Examples:
            - Single residue: "H99", "L6A" (chain + residue number + insertion code)
            - Fragment: "H1-98", "H-98" (start to 98), "H98-" (98 to end)
            - Whole chain: "H", "L", "P"
            - Multiple fragments: "H26-33,51-57,93-102" (same chain)
            - Multiple chains: "H:L:P" (different chains)
            - Complex: "H26-33:L:P1-50" (CDRs + light chain + antigen fragment)

        Examples:
            >>> p = Protein("5cil.pdb")
            >>> all_res = p.rs()                    # All residues
            >>> ab_chains = p.rs("H:L")             # Heavy and light chains
            >>> cdr1 = p.rs("H26-33")              # CDR-H1 region
            >>> antigen = p.rs("P")                 # Antigen chain P
            >>> interface = p.rs("H26-33,51-57")    # Multiple CDR regions

        Note:
            RS objects support boolean operations (& | ~) and can be used with
            other afpdb methods for structural analysis, extraction, and manipulation.
        """
        return RS(self, rs)

    def rl(self, rl: ContigType = None) -> 'RL':
        return RL(self, rl)

    def _get_xyz(self, rs: 'RS', ats: 'ATS') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract atom XYZ coordinate, the corresponding residue id and atom id will also be returned
            this is an internal method, so we assume rs is an RS object and ats is an ATS object,
            so we don't waste time converting them again
        """
        if ats.not_full():
            mask=self.data.atom_mask[rs.data][:, ats.data] > 0
            res=self.data.atom_positions[rs.data][:, ats.data][mask].reshape(-1,3)
        else:
            mask=self.data.atom_mask[rs.data] > 0
            res=self.data.atom_positions[rs.data][mask].reshape(-1,3)
        mask=mask.ravel()
        n=len(rs.data)
        n_ats=len(ats)
        # residue idx
        rsi= np.repeat(rs.data, n_ats)[mask]
        # atom idx
        atsi=np.tile(ats.data, n)[mask]
        return (rsi, atsi, res)

    def _get_xyz_pair(self, target_p: 'Protein', rs_a: 'RS', rs_b: 'RS', ats: 'ATS') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """internal method, rs and ats must be RS and ATS object, return pairs of atoms, where both masks are 1,
            used for align and rmsd"""
        if ats.not_full():
            mask_a=self.data.atom_mask[np.ix_(rs_a.data, ats.data)] > 0
            mask_b=target_p.data.atom_mask[np.ix_(rs_b.data, ats.data)] > 0
            mask=mask_a & mask_b
            res_a=self.data.atom_positions[np.ix_(rs_a.data, ats.data)][mask].reshape(-1,3)
            res_b=target_p.data.atom_positions[np.ix_(rs_b.data, ats.data)][mask].reshape(-1,3)
        else:
            mask_a=self.data.atom_mask[rs_a.data] > 0
            mask_b=target_p.data.atom_mask[rs_b.data] > 0
            mask=mask_a & mask_b
            res_a=self.data.atom_positions[rs_a.data][mask].reshape(-1,3)
            res_b=target_p.data.atom_positions[rs_b.data][mask].reshape(-1,3)
        mask=mask.ravel()
        n_a=len(rs_a.data)
        n_b=len(rs_b.data)
        n_ats=len(ats)
        # residue idx
        rsi_a= np.repeat(rs_a.data, n_ats)[mask]
        rsi_b= np.repeat(rs_b.data, n_ats)[mask]
        # atom idx
        atsi_a=np.tile(ats.data, n_a)[mask]
        atsi_b=np.tile(ats.data, n_b)[mask]

        return (rsi_a, atsi_a, rsi_b, atsi_b, res_a, res_b)


    def rs_around(self, rs: ContigType, dist: float = 5, ats: Optional[AtomSelectionType] = None,
                  rs_within: Optional[ContigType] = None, drop_duplicates: bool = False,
                  kdtree_bucket_size: int = 10, keep_atoms: bool = False) -> Tuple['RS', 'RS', pd.DataFrame]:
        """Find all residues within a specified distance of target residues.

        This method performs spatial proximity searches to identify residues that
        have at least one atom within the specified distance of the target residues.
        It's commonly used for interface analysis, contact prediction, and identifying
        binding sites in protein complexes.

        Args:
            rs (str, RS, list): Target residue selection using contig syntax:
                - "H:L": Heavy and light antibody chains
                - "P1-50": First 50 residues of chain P
                - RS object: Existing residue selection
            dist (float, optional): Maximum distance threshold in Angstroms for
                considering residues as neighbors. Defaults to 5.0 Å.
            ats (str, ATS, list, optional): Atom types to consider in distance
                calculations. Defaults to all atoms:
                - "CA": Alpha carbons only (fast, approximate)
                - "N,CA,C,O": Backbone atoms
                - None: All available atoms (comprehensive)
            rs_within (str, RS, optional): Restrict search to residues within this
                selection. If None, searches entire protein. Useful for limiting
                search to specific chains or regions.
            drop_duplicates (bool, optional): If True, keeps only one target
                residue per neighbor found. If False, keeps all target-neighbor
                pairs. Defaults to False.
            kdtree_bucket_size (int, optional): KDTree optimization parameter.
                Use 0 to disable KDTree (slower but more reliable). Defaults to 10.
            keep_atoms (bool, optional): If True, returns all atom pairs in the
                output DataFrame. If False, returns only closest atom pair per
                residue pair. Defaults to False.

        Returns:
            tuple: (rs_neighbor, rs_seed, df_distances) where:
                - rs_neighbor (RS): Residues found within distance of targets
                - rs_seed (RS): Corresponding target residues for each neighbor
                - df_distances (DataFrame): Detailed distance information with columns:
                    ['chain_a', 'resi_a', 'resn_a', 'resn_i_a', 'atom_a',
                     'chain_b', 'resi_b', 'resn_b', 'resn_i_b', 'atom_b', 'dist']

        Common Use Cases:
            Interface Analysis:
                >>> ab_chains, ag_chains, contacts = p.rs_around("P", dist=4, rs_within="H:L")
                >>> print(f"Found {len(ab_chains)} antibody residues contacting antigen")

            Binding Site Detection:
                >>> neighbors, seeds, distances = p.rs_around("H26-33", dist=6, ats="CA")
                >>> binding_site = neighbors | seeds  # Combine for full binding site

            Contact Maps:
                >>> _, _, contacts = p.rs_around("A", rs_within="B", keep_atoms=True)
                >>> print(contacts)

            Epitope-Paratope Analysis:
                >>> paratope, epitope, details = p.rs_around("P", dist=4.5, rs_within="H:L")
                >>> print(f"Epitope size: {len(epitope)} residues")
                >>> print(f"Paratope size: {len(paratope)} residues")

        Performance Optimization:
            Fast Approximation (CA atoms):
                >>> neighbors, _, _ = p.rs_around("target", ats="CA", dist=8)

            Comprehensive Analysis (all atoms):
                >>> neighbors, _, contacts = p.rs_around("target", dist=4, keep_atoms=True)

            Chain-Specific Search:
                >>> neighbors, _, _ = p.rs_around("A", rs_within="B:C", dist=5)

        Distance Calculation Details:
            - Uses Euclidean distance in 3D space
            - Considers all atom pairs between target and candidate residues
            - Returns minimum distance per residue pair (unless keep_atoms=True)
            - KDTree acceleration for large structures (can be disabled)

        DataFrame Output Columns:
            - chain_a/b: Chain identifiers for interacting residues
            - resi_a/b: Internal residue indices
            - resn_a/b: Residue names (with insertion codes)
            - resn_i_a/b: Integer residue numbers
            - atom_a/b: Atom names for closest contact pair
            - dist: Distance in Angstroms between atoms

        Examples:
            >>> p = Protein("5cil.pdb")  # Antibody-antigen complex

            # Find antibody residues contacting antigen
            >>> binders, antigen, contacts = p.rs_around("P", dist=4, rs_within="H:L")
            >>> print(f"Binding residues: {len(binders)}")
            >>> print(f"Contact details: {len(contacts)} atom pairs")

            # Analyze CDR contacts with high precision
            >>> cdr_neighbors, _, details = p.rs_around("H26-33", dist=3.5,
            ...                                        rs_within="P", keep_atoms=True)
            >>> close_contacts = details[details.dist < 3.0]
            >>> print(f"Very close contacts: {len(close_contacts)}")

            # Fast screening with alpha carbons
            >>> ca_neighbors, _, _ = p.rs_around("active_site", ats="CA", dist=10)

        """
        rs=RS(self, rs)
        if rs.is_empty(): util.error_msg("rs is empty!")
        rs_b=rs._not(rs_full=rs_within)
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in rs_around()!")

        def box(xyz, dist):
            """Return the vertices for min/max box around coordinates.

            Args:
                xyz: Coordinate array.
                dist: Distance to extend box boundaries.

            Returns:
                tuple: (min_coords, max_coords) defining bounding box.
            """
            return (np.min(xyz, axis=0).reshape(-1,3)-dist, np.max(xyz, axis=0).reshape(-1,3)+dist)

        rsi_a, atsi_a, xyz_a=self._get_xyz(rs, ats)
        rsi_b, atsi_b, xyz_b=self._get_xyz(rs_b, ats)

        if kdtree_bucket_size>0:
            from Bio.PDB.kdtrees import KDTree
            kdt=KDTree(xyz_b, bucket_size=kdtree_bucket_size)
            points=[kdt.search(center, dist) for center in xyz_a]
            repeat=np.array([len(x) for x in points])
            neighbor=np.array([point.index for x in points for point in x])
            if len(neighbor)==0:
                rsi_a,rsi_b,atsi_a,atsi_b,d=np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0)
            else:
                rsi_a=np.repeat(rsi_a, repeat, axis=0)
                atsi_a=np.repeat(atsi_a, repeat, axis=0)
                xyz_a=np.repeat(xyz_a, repeat, axis=0)
                rsi_b, atsi_b, xyz_b=rsi_b[neighbor], atsi_b[neighbor], xyz_b[neighbor]
                d=np.linalg.norm(xyz_a-xyz_b, axis=-1)
        else:
            min_a, max_a=box(xyz_a, dist)
            min_b, max_b=box(xyz_b, dist)
            mask_b=np.min((xyz_b>=min_a) & (xyz_b<=max_a), axis=1)
            rsi_b, atsi_b, xyz_b=rsi_b[mask_b], atsi_b[mask_b], xyz_b[mask_b]
            mask_a=np.min((xyz_a>=min_b) & (xyz_a<=max_b), axis=1)
            rsi_a, atsi_a, xyz_a=rsi_a[mask_a], atsi_a[mask_a], xyz_a[mask_a]

            if len(rsi_a)==0 or len(rsi_b)==0:
               rsi_a,rsi_b,atsi_a,atsi_b,d=np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0)
            else:
                # all atom-to-atom distance
                d=np.linalg.norm(xyz_a[:, None]-xyz_b[None, :], axis=-1).ravel()
                mask= d<=dist

                n_a=len(rsi_a) # rows in res_a
                n_b=len(rsi_b) # rows in res_b
                # compute resiude index and atom index for rows in d
                # residue idx
                rsi_a= np.repeat(rsi_a, n_b)[mask]
                rsi_b= np.tile(rsi_b, n_a)[mask]
                # atom idx
                atsi_a=np.repeat(atsi_a, n_b)[mask]
                atsi_b=np.tile(atsi_b, n_a)[mask]
                d=d[mask]

        at=np.array(afres.atom_types)
        atom_a=at[atsi_a]
        atom_b=at[atsi_b]

        rs_a=RL(self, rsi_a)
        rs_b=RL(self, rsi_b)
        df=pd.DataFrame(data={
            'chain_a':rs_a.chain(), 'resi_a':rsi_a, 'resn_a':rs_a.name(), 'resn_i_a':rs_a.namei(), 'atom_a':atom_a,
            'chain_b':rs_b.chain(), 'resi_b':rsi_b, 'resn_b':rs_b.name(), 'resn_i_b':rs_b.namei(), 'atom_b':atom_b,
            'dist':d})
        df=df.sort_values('dist')
        if not keep_atoms:
            df.drop_duplicates(['resi_a','resi_b'], inplace=True)
            if drop_duplicates:
                df.drop_duplicates('resi_b', inplace=True)
        return (RS(self, rsi_b), RS(self, rsi_a), df)

    def rsi_missing(self) -> np.ndarray:
        """Return a selection array (not a residue selection object) for missing residues.
            self (p_old) contains missing residues.
            In AlphaFold, we replace X in the sequence by G, which leads to a new object p_new.

            The returned selection array points to the Gs in p_new
            rsi_m=p_old.rsi_missing()
            # rsi_m cannot be used on p_old, as the missing residues do not exist in the backend ndarrays
            # rsi_m should be used on p_new, where the missing residues are no longer missing (replaced by G).
            # You should make sure p_old and p_new has the same chain order.
            # delete all G residues in p_new
            p_no_G=p_new.extract(p_new.rs_not(rsi_m))
        """
        s=self.seq().replace(":", "") # need to remove ":", otherwise index is off
        if 'X' not in s:
            return [] # empty selection
        return np.array(util.index_all("X", list(s)))

    def rs_missing(self) -> 'RS':
        """
        **OBSOLETE**: This method is deprecated. Use rsi_missing() instead.

        This method was originally intended to return an RS (Residue Selection) object
        for missing residues, but it actually returns an index array. For clarity and
        consistency, use rsi_missing() instead.

        Returns:
            numpy.ndarray: Index array of missing residue positions

        Examples:
            # DEPRECATED - do not use
            >>> missing = p.rs_missing()

            # PREFERRED - use this instead
            >>> missing_indices = p.rsi_missing()

        Note:
            - This method is obsolete and maintained only for backward compatibility
            - Use rsi_missing() for new code
            - Despite the name suggesting an RS object, this returns indices
        """

        print("Please use rsi_missing() instead. The method returns an index array, instead of an RS object.")
        return self.rsi_missing()

    def rs_next2missing(self) -> 'RS':
        """Return a selection for residues next to a gap"""
        chains=self.data.chain_index
        resi,code=self.split_residue_index(self.data.residue_index)
        # where chain breaks
        idx=np.where(chains[1:]!=chains[:-1])[0]+1
        rs_i=[]
        for i,k in enumerate(chains):
            if i in idx: continue
            if i>0 and resi[i]-resi[i-1]>1:
                rs_i.extend([i-1,i])
        return self.rs(rs_i)

    def rs_insertion(self, p_missing: 'Protein') -> 'RS':
        """object p_missing contains missing residues, which were replaced by glycine in the self object
        This methid is to return a selection for those inserted glycines, so that they can be removed:

            p=Protein("alphafold_pred.pdb")
            rs_G=p.rs_insertion(Protein("experiment.pdb"))
            q=p.extract(~ rs_G)
            # q no longer has any inserted residues
        """
        return RS(self, p_missing.rsi_missing())

    def rs_mutate(self, obj: 'Protein') -> 'RS':
        """Return a selection for mutated residues.
        Here, we assume the two proteins have the same same number of residues and the chains are
        in the same order. In the future, we can consider performing alignment"""
        c_a=self.chain_pos()
        c_b=obj.chain_pos()
        out=[]
        for k, (b,e) in c_a.items():
            if k not in c_b:
                continue
            b2,e2=c_b[k]
            if e-b!=e2-b2:
                util.error_msg(f"The length of chain {k} do not match! self: {e-b+1}, obj {e2-b2+1}.")
            for i in range(b, e+1):
                if self.data.aatype[i]!=obj.data.aatype[b2+i-b]: out.append(i)
        return RS(self, out)

    def rs_seq(self, seq: str, in_chains: Optional[List[str]] = None) -> List['RS']:
        """Search a sequence fragment, return all matches
            seq can be a regular expression
            if in_chains list is provided, only search within the given chains
            Return, a list of selections, not a selection object, as there can be multiple matches
        """
        c_pos=self.chain_pos()
        out=[]
        for k,s in self.seq_dict().items():
            #idx=[m.start() for m in re.finditer(rf'(?={seq})', s)]
            if in_chains is not None and k not in in_chains: continue
            for m in re.finditer(seq, s):
                # s may contain gap X, we should not count X
                i1=len(s[:m.start()].replace('X', ''))
                i2=len(s[:m.start()+len(m.group())].replace('X', ''))
                out.append(RS(self, np.array(range(c_pos[k][0]+i1, c_pos[k][0]+i2))))
        return out

    def rs_not(self, rs: ContigType, rs_full: ContigType = None) -> 'RS':
        return RS(self, rs)._not(rs_full=rs_full)

    def rs_notin(self, rs_a: ContigType, rs_b: ContigType) -> 'RS':
        return RS(self, rs_a)-RS(self, rs_b)

    def rs_and(self, *L_rs: ContigType) -> 'RS':
        L_rs=[RS(self, x) for x in L_rs]
        return RS._and(*L_rs)

    def rs_or(self, *L_rs: ContigType) -> 'RS':
        L_rs=[RS(self, x) for x in L_rs]
        return RS._or(*L_rs)

    def rs2str(self, rs: ContigType, format: str = "CONTIG") -> str:
        return RS(self, rs).__str__(format=format)

    def ats(self, ats: AtomSelectionType) -> 'ATS':
        return ATS(ats)

    def ats_not(self, ats: AtomSelectionType) -> 'ATS':
        return ~ATS(ats)

    def ats2str(self, ats: AtomSelectionType) -> str:
        return str(ATS(ats))

    def _atom_dist(self, rs_a: ContigType, rs_b: ContigType, ats: Optional[AtomSelectionType] = None) -> pd.DataFrame:
        ats=ATS(ats)
        if ats.is_empty(): util.error_msg("ats is emtpy!")
        rs_a=RS(self, rs_a)
        rs_b=RS(self, rs_b)
        if rs_a.is_empty(): util.error_msg("rs_a is emtpy!")
        if rs_b.is_empty(): util.error_msg("rs_b is emtpy!")
        rsi_a, atsi_a, xyz_a=self._get_xyz(rs_a, ats)
        rsi_b, atsi_b, xyz_b=self._get_xyz(rs_b, ats)

        # all atom-to-atom distance
        d=np.linalg.norm(xyz_a[:, None]-xyz_b[None, :], axis=-1).ravel()
        n_a=len(rsi_a) # rows in xyz_a
        n_b=len(rsi_b) # rows in xyz_b
        # compute resiude index and atom index for rows in d
        # residue idx
        rsi_a= np.repeat(rsi_a, n_b)
        rsi_b= np.tile(rsi_b, n_a)
        # atom idx
        atsi_a=np.repeat(atsi_a, n_b)
        atsi_b=np.tile(atsi_b, n_a)
        at=np.array(afres.atom_types)
        atom_a=at[atsi_a]
        atom_b=at[atsi_b]

        df=pd.DataFrame(data={'resi_a':rsi_a, 'atom_a':atom_a, 'resi_b':rsi_b, 'atom_b':atom_b, 'dist':d})
        return df

    def _point_dist(self, center: np.ndarray, rs: ContigType, ats: Optional[AtomSelectionType] = None) -> pd.DataFrame:
        rs=RS(self, rs)
        if rs.is_empty(): util.error_msg("rs is empty!")
        ats=ATS(ats)
        if ats.is_empty(): util.error_msg("ats cannot be empty in _point_dist()!")
        rsi, atsi, xyz=self._get_xyz(rs, ats)

        d = np.linalg.norm(xyz - center.reshape(1,3), axis=-1)
        at=np.array(afres.atom_types)
        atom=at[atsi]
        df=pd.DataFrame(data={'resi':rsi, 'atom':atom, 'dist':d})
        return df

    def atom_dist(self, rs_a: ContigType, rs_b: ContigType, ats: Optional[AtomSelectionType] = None) -> float:
        df=self._atom_dist(rs_a, rs_b, ats)
        rs_a=RL(self, df.resi_a.values)
        rs_b=RL(self, df.resi_b.values)
        df['chain_a']=rs_a.chain()
        df['resn_a']=rs_a.name()
        df['resn_i_a']=rs_a.namei()
        df['chain_b']=rs_b.chain()
        df['resn_b']=rs_b.name()
        df['resn_i_b']=rs_b.namei()
        df['res_a']=rs_a.aa()
        df['res_b']=rs_b.aa()
        df=df[['chain_a','resn_a','resn_i_a','resi_a','res_a','chain_b','resn_b','resn_i_b','resi_b','res_b','dist','atom_a','atom_b']]
        df=df.sort_values('dist')
        return df

    def disulfied_pair(self, min_dist: float = 1.0, max_dist: float = 2.5) -> pd.DataFrame:
        L_rs=self.rs_seq('C') # Cys
        empty=pd.DataFrame([], columns=['chain_a','resn_a','resn_i_a','resi_a','res_a','chain_b','resn_b','resn_i_b','resi_b','res_b','dist','atom_a','atom_b'])
        if len(L_rs)==0: return empty
        rs_a=self.rs_or(*L_rs)
        df=self._atom_dist(rs_a, rs_a, ats='SG')
        df=df[(df.dist >= min_dist) & (df.dist <= max_dist)].copy()
        return df

    def rs_dist(self, rs_a: ContigType, rs_b: ContigType, ats: Optional[AtomSelectionType] = None) -> pd.DataFrame:
        df=self.atom_dist(rs_a, rs_b, ats)
        return df.drop_duplicates(['resi_a','resi_b'])

    def cb_dist(self, rs_a: ContigType, rs_b: ContigType) -> pd.DataFrame:
        """Return Cb distances between the two residue groups, uses Ca for Glycine"""
        df=self.atom_dist(rs_a, rs_b, ats='CA,CB')
        mask=((df.res_a!='G')&(df.atom_a=='CA'))|((df.res_b!='G')&(df.atom_b=='CA'))
        return df[~mask].copy()

    def rs_dist_to_point(self, center: np.ndarray, rs: Optional[ContigType] = None,
                        ats: Optional[AtomSelectionType] = None) -> pd.DataFrame:
        """Rank residues (within rs) according to their shortest distance to a point
            center: np.array(3) for XYZ
        """
        df=self._point_dist(center, rs, ats)
        rs=RL(self, df.resi.values)
        df['chain_a']=rs.chain()
        df['resn']=rs.name()
        df['resn_i']=rs.namei()
        df['res']=rs.aa()
        df=df[['chain','resn','resn_i','resi','res','atom','dist']]
        df=df.sort_values('dist')
        return df

    def rmsd(self, obj_b: 'Protein', rl_a: Optional[ContigType] = None, rl_b: Optional[ContigType] = None,
             ats: Optional[AtomSelectionType] = None, align: bool = False) -> float:
        """Calculate Root Mean Square Deviation (RMSD) between two protein structures.

        RMSD quantifies structural similarity by measuring the average distance between
        corresponding atoms after optimal superposition. This is the gold standard
        metric for comparing protein structures and conformations.

        Args:
            obj_b (Protein): Target protein structure to compare against
            rl_a (str, RL, optional): Residue selection for this protein (structure A):
                - None: All residues (default)
                - String: Contig syntax like "H:L", "A1-50", "10-20"
                - RL object: Existing residue list
            rl_b (str, RL, optional): Residue selection for target protein (structure B):
                - None: All residues (default)
                - String: Contig syntax like "H:L", "A1-50", "10-20"
                - RL object: Existing residue list
            ats (str, ATS, optional): Atom selection for RMSD calculation:
                - None: All atoms (default) - includes side chains
                - "CA": Alpha carbons only (most common for global comparison)
                - "N,CA,C,O": Backbone atoms only
                - "N,CA,C": Backbone without carbonyl oxygen
                - ATS object: Custom atom selection
            align (bool): Whether to perform structural alignment before RMSD:
                - False: Calculate RMSD as-is (default)
                - True: Optimally align structure A to B, then calculate RMSD
                         ⚠️ WARNING: This permanently modifies the current protein!

        Returns:
            float: RMSD value in Angstroms (Å)
                - Values < 1.0 Å: Nearly identical structures
                - Values 1-2 Å: Very similar structures (homologs)
                - Values 2-5 Å: Moderately similar (same fold)
                - Values > 5 Å: Structurally different

        RMSD Interpretation Guide:
            Structural Biology Context:
                - 0.1-0.5 Å: Crystal structure precision
                - 0.5-1.0 Å: NMR ensemble range
                - 1.0-2.0 Å: Close homologs/conformers
                - 2.0-3.0 Å: Distant homologs
                - 3.0-5.0 Å: Same fold family
                - > 5.0 Å: Different folds

            Prediction Assessment:
                - < 1.5 Å: Excellent prediction
                - 1.5-3.0 Å: Good prediction
                - 3.0-5.0 Å: Acceptable prediction
                - > 5.0 Å: Poor prediction

        Examples:
            Basic RMSD Calculations:
                >>> exp = Protein("5cil.pdb")         # Experimental structure
                >>> pred = Protein("5cil_af2.pdb")    # AlphaFold prediction

                # Global alpha-carbon RMSD (most common)
                >>> ca_rmsd = exp.rmsd(pred, ats="CA")
                >>> print(f"Global CA-RMSD: {ca_rmsd:.2f} Å")

                # All-atom RMSD (includes side chains)
                >>> all_rmsd = exp.rmsd(pred)
                >>> print(f"All-atom RMSD: {all_rmsd:.2f} Å")

                # Backbone-only RMSD
                >>> bb_rmsd = exp.rmsd(pred, ats="N,CA,C,O")
                >>> print(f"Backbone RMSD: {bb_rmsd:.2f} Å")

            Chain-Specific Comparisons:
                >>> # Compare specific chains
                >>> h_rmsd = exp.rmsd(pred, rl_a="H", rl_b="H", ats="CA")
                >>> l_rmsd = exp.rmsd(pred, rl_a="L", rl_b="L", ats="CA")
                >>> print(f"Heavy chain CA-RMSD: {h_rmsd:.2f} Å")
                >>> print(f"Light chain CA-RMSD: {l_rmsd:.2f} Å")

                # CDR region comparison
                >>> cdr_rmsd = exp.rmsd(pred, rl_a="H:27-38", rl_b="H:27-38")
                >>> print(f"CDR-H1 RMSD: {cdr_rmsd:.2f} Å")

            Domain-Specific Analysis:
                >>> # Compare binding sites
                >>> site_exp = exp.rs_around("H:Y32", 8.0)  # 8Å around tyrosine
                >>> site_pred = pred.rs_around("H:Y32", 8.0)
                >>> site_rmsd = exp.rmsd(pred, rl_a=site_exp.name(),
                ...                      rl_b=site_pred.name(), ats="CA")
                >>> print(f"Binding site RMSD: {site_rmsd:.2f} Å")

            Structural Alignment with RMSD:
                >>> # Clone to preserve original
                >>> exp_aligned = exp.clone()
                >>> aligned_rmsd = exp_aligned.rmsd(pred, align=True, ats="CA")
                >>> print(f"Aligned CA-RMSD: {aligned_rmsd:.2f} Å")
                >>> # exp_aligned is now superposed onto pred

            Conformational Analysis:
                >>> # Compare different conformations
                >>> conf1 = Protein("protein_open.pdb")
                >>> conf2 = Protein("protein_closed.pdb")
                >>> conf_rmsd = conf1.rmsd(conf2, rl_a="100-150", rl_b="100-150")
                >>> print(f"Domain motion RMSD: {conf_rmsd:.2f} Å")

        Selection Requirements:
            The method requires corresponding atoms between structures:
            - rl_a and rl_b must select the same number of residues
            - Selected residues must have the same atoms (atom_mask overlap)
            - Missing atoms in either structure will cause errors

            >>> # Verify atom correspondence
            >>> sel_a = exp.rs("100-200")
            >>> sel_b = pred.rs("100-200")
            >>> if len(sel_a) != len(sel_b):
            ...     print("Selection length mismatch!")

        Performance Considerations:
            - CA-RMSD: Fastest, good for global comparison
            - Backbone RMSD: Moderate speed, good for fold comparison
            - All-atom RMSD: Slowest, most detailed comparison
            - Alignment adds computational cost but improves interpretability

        Error Handling:
            Common issues and solutions:
            - "Different number of atoms": Check that selections match
            - "Empty atom selection": Verify ats parameter is valid
            - "Different residue count": Ensure rl_a and rl_b have same length

        Mathematical Background:
            RMSD = √(1/N × Σᵢ(rᵢₐ - rᵢᵦ)²) where:
            - N = number of atom pairs
            - rᵢₐ, rᵢᵦ = coordinates of corresponding atoms
            - Minimized over all rotations/translations when align=True

        Note:
            When align=True, this protein object is permanently modified through
            superposition. Use clone() to preserve the original structure if needed.
            The alignment uses the Kabsch algorithm for optimal superposition.
        """
        rl_a=RL(self, rl_a)
        rl_b=RL(obj_b, rl_b)
        assert(len(rl_a)==1 or len(rl_b)==1 or len(rl_a)==len(rl_b))
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in rmsd()!")

        if align:
            R,t=self.align(target_p=obj_b, rl_a=rl_a, rl_b=rl_b, ats=ats)

        (rsi_a, atsi_a, rsi_b, atsi_b, res_a, res_b) = self._get_xyz_pair(obj_b, rl_a, rl_b, ats)
        if res_a.shape!=res_b.shape:
            print("WARNING> selections do not share the same shape!", res_a.shape, "vs", res_b.shape)
        d=np.linalg.norm(res_a - res_b, axis=-1)
        n=d.shape[0]
        return np.sqrt(np.sum(d*d)/n)

    @staticmethod
    def dockQ(p, q, rs_a, rs_b, capri_peptide=False):
        """Calculate DockQ score to evaluate the quality of protein-protein docking.

        Args:
            p: First protein structure.
            q: Second protein structure.
            rs_a: Residue selection for interface in first protein.
            rs_b: Residue selection for interface in second protein.
            capri_peptide: Whether to use CAPRI peptide evaluation criteria.

        Returns:
            float: DockQ score ranging from 0 to 1, where 1 indicates perfect match.
        """
        """p is native, q is model
           rs_a and rs_b specify the two interacting groups, e.g., rs_a='H:L' and rs_b='P'
           capri_peptide is the same as https://github.com/bjornwallner/DockQ

        Note: two proteins p and q must share the same chain names and in the same order!
        """

        q=q.clone() # we make a clone, so that when q is aligned, we don't change the original q object

        if ":".join(p.chain_id())!=":".join(q.chain_id()):
            return Exception("Please rename and order all chains, e.g., use q.rename_reorder_chains(p, {'A':'B', 'B':'A'})")

        FNAT_THRESHOLD: float = 5.0
        FNAT_THRESHOLD_PEPTIDE: float = 4.0
        INTERFACE_THRESHOLD: float = 10.0
        INTERFACE_THRESHOLD_PEPTIDE: float = 8.0
        CLASH_THRESHOLD: float = 2.0

        def f1(tp, fp, p): return 2* tp/(tp+fp+p)

        rs_a=p.rs(rs_a)
        rs_b=p.rs(rs_b)
        fnat_threshold=FNAT_THRESHOLD_PEPTIDE if capri_peptide else FNAT_THRESHOLD
        rs_nbr_a, rs_seed_a, t_a=p.rs_around(rs_a, rs_within=rs_b, dist=fnat_threshold)
        rs_nbr_b, rs_seed_b, t_b=q.rs_around(rs_a, rs_within=rs_b, dist=fnat_threshold)
        pair_a={ (r['resi_a'], r['resi_b']) for i,r in t_a.iterrows() }
        pair_b={ (r['resi_a'], r['resi_b']) for i,r in t_b.iterrows() }
        pair_ab=pair_a & pair_b
        out={}
        out['fnat']=len(pair_ab)/len(pair_a) if len(pair_a) else 0
        out['fnonnat']=1-len(pair_ab)/len(pair_b) if len(pair_b) else 0
        out['nat_correct']=len(pair_ab)
        out['nat_total']=len(pair_a)
        out['nonnat_count']=len(pair_b)-len(pair_ab)
        out['model_total']=len(pair_b)
        out['clashes']=sum(t_a.dist<CLASH_THRESHOLD)
        out['len1']=len(rs_a)
        out['len2']=len(rs_b)
        out['F1']=f1(len(pair_ab), len(pair_b)-len(pair_ab), len(pair_a))

        if len(rs_a)>len(rs_b):
            out['class1']='receptor'
            out['class2']='ligand'
            q.align(p, rs_a, rs_a, ats="N,CA,C,O")
            LRMSD=q.rmsd(p, rs_b, rs_b, ats="N,CA,C,O")
        else:
            out['class1']='ligand'
            out['class2']='receptor'
            q.align(p, rs_b, rs_b, ats="N,CA,C,O")
            LRMSD=q.rmsd(p, rs_a, rs_a, ats="N,CA,C,O")
        out['LRMSD']=LRMSD

        int_threshold=INTERFACE_THRESHOLD_PEPTIDE if capri_peptide else INTERFACE_THRESHOLD

        if capri_peptide:
            rs_nbr, rs_seed, t=p.rs_around(rs_a, rs_within=rs_b, dist=int_threshold, ats="CA,CB", keep_atoms=True)
            t['aa_a']=p.rl(t.resi_a).aa()
            t['aa_b']=p.rl(t.resi_b).aa()
            # remove CA entries for non-G residues
            mask=((t.aa_a!='G')&(t.atom_a=='CA'))|((t.aa_b!='G')&(t.atom_b=='CA'))
            t=t[~mask]
            rs_int=p.rs(t.resi_a)|p.rs(t.resi_b)
        else:
            rs_nbr, rs_seed, t=p.rs_around(rs_a, rs_within=rs_b, dist=int_threshold)
            rs_int=rs_nbr | rs_seed
        if len(rs_int)==0:
            util.warn_msg("No interface residues are identified, DockQ score is incomplete!")
            iRMSD=np.nan
            out['DockQ']=np.nan
        else:
            iRMSD=q.rmsd(p, rs_int, rs_int, ats="N,CA,C,O", align=True)
        out['iRMSD']=iRMSD
        out['DockQ']=(out['fnat']+1/(1+(iRMSD/1.5)**2)+1/(1+(LRMSD/8.5)**2))/3
        #print(out)
        return out

    @staticmethod
    def merge(objs: Optional[List['Protein']]) -> Optional['Protein']:
        """merge multiple objects into one object, chains are renamed if the chain name has been used by earlier objects.
            The first available chain letter will be used for the renaming.
           note: there is no split() method, as you can do that by using extract
            (b/c sometimes you may want each object to contain more than one chain, e.g., in the case of antibody dimers)
        """
        if objs is None or len(objs)==1:
            print("WARNING: not merged, as at least two objects are required.")
            if objs is None: return None
            return objs[0].clone()

        # first rename chains
        # preserve the original objects untouched
        objs=[x.clone() for x in objs]
        chains=util.unique([y for x in objs for y in x.chain_id()])
        chain_id=list(objs[0].chain_id())
        n_chain=len(chain_id)
        for i,o in enumerate(objs[1:]):
            cid=o.data.chain_id
            for c in cid:
                if c in chain_id:
                    for x in afprt.PDB_CHAIN_IDS:
                        if x not in chains: break
                    print(f"Rename chain: object {i+1}, {c} to {x}")
                    c=x # a new chain name
                chain_id.append(c)
                chains.append(c)
            o.data.chain_index+=n_chain
            n_chain=len(chain_id)
        o=objs[0]
        q=o.data
        q.chain_id=np.array(chain_id)
        q.chain_index=np.concatenate([o.data.chain_index for o in objs], axis=0)
        q.residue_index=np.concatenate([o.data.residue_index for o in objs], axis=0)
        q.aatype=np.concatenate([o.data.aatype for o in objs], axis=0)
        q.atom_positions=np.concatenate([o.data.atom_positions for o in objs], axis=0)
        q.atom_mask=np.concatenate([o.data.atom_mask for o in objs], axis=0)
        q.b_factors=np.concatenate([o.data.b_factors for o in objs], axis=0)
        o._make_res_map()
        return o

    def __add__(self, other: 'Protein') -> 'Protein':
        """Support '+' operator for merging two protein objects.

        Enables intuitive merging of protein structures using the '+' operator.
        This method is a convenient wrapper around the static merge() method,
        allowing for more natural syntax when combining protein objects.

        Args:
            other (Protein): Another protein object to merge with this one

        Returns:
            Protein: A new merged protein object containing chains from both proteins

        Chain Handling:
            - Chains are automatically renamed if conflicts exist
            - Original objects remain unchanged (creates clones)
            - Uses the first available chain letter for renaming
            - Preserves all structural information from both proteins

        Examples:
            Basic Merging:
                >>> heavy_chain = Protein("heavy.pdb")     # Chain H
                >>> light_chain = Protein("light.pdb")     # Chain L
                >>> antibody = heavy_chain + light_chain
                >>> print("Merged chains:", antibody.chain_list())
                ['H', 'L']

            Multiple Chain Merging:
                >>> fab = heavy_chain + light_chain
                >>> antigen = Protein("antigen.pdb")       # Chain A
                >>> complex_structure = fab + antigen
                >>> print("Complex chains:", complex_structure.chain_list())
                ['H', 'L', 'A']

            Chain Conflict Resolution:
                >>> protein1 = Protein("domain1.pdb")     # Chain A
                >>> protein2 = Protein("domain2.pdb")     # Chain A (conflict!)
                >>> merged = protein1 + protein2          # Automatically renames to A, B
                >>> print("Resolved chains:", merged.chain_list())
                ['A', 'B']  # Second chain A becomes B

        Comparison with merge():
            Using + operator (recommended):
                >>> result = protein1 + protein2

            Using merge() method (explicit):
                >>> result = Protein.merge([protein1, protein2])

            Both produce identical results, but '+' is more intuitive.

        Note:
            This method creates a new protein object and does not modify either
            of the input proteins. If you need to merge multiple proteins at once,
            consider using the static merge() method directly for better performance.
        """
        if not isinstance(other, Protein):
            raise TypeError("Can only merge with another Protein object")
        return Protein.merge([self, other])


    def align(self, target_p: 'Protein', rl_a: Optional[ContigType] = None, rl_b: Optional[ContigType] = None,
              ats: Optional[AtomSelectionType] = None) -> Tuple['Protein', float]:
        """Structurally align this protein to a target protein using selected residues.

        Performs optimal structural superposition by computing the rotation matrix
        and translation vector that minimizes the RMSD between corresponding atoms
        in the two structures. The alignment modifies the coordinates of the current
        protein object.

        Args:
            target_p (Protein): Target protein structure to align to. This structure
                remains unchanged and serves as the reference frame.
            rl_a (str, RL, optional): Residue selection from current protein for
                alignment calculation:
                - None: Use all residues (default)
                - String: Contig syntax like "H:L", "A1-50"
                - RL object: Existing residue list for precise control
            rl_b (str, RL, optional): Corresponding residue selection from target
                protein. Must have same length as rl_a:
                - None: Use all residues (default)
                - String: Contig syntax matching rl_a
                - RL object: One-to-one correspondence with rl_a
            ats (str, ATS, optional): Atom types to use for alignment:
                - None: All available atoms (default)
                - "CA": Alpha carbons only (most common)
                - "N,CA,C,O": Backbone atoms (robust)
                - ATS object: Custom atom selection

        Returns:
            tuple: (R, t) where:
                - R (numpy.ndarray): 3×3 rotation matrix
                - t (numpy.ndarray): 1×3 translation vector
                The transformation applied was: new_coords = old_coords @ R + t

        Common Alignment Strategies:
            Global Alignment (whole structures):
                >>> R, t = protein1.align(protein2, ats="CA")

            Domain-Specific Alignment:
                >>> R, t = protein1.align(protein2, rl_a="A1-100", rl_b="B1-100")

            Backbone-Only Alignment (robust):
                >>> R, t = protein1.align(protein2, ats="N,CA,C,O")

            Antibody Variable Region Alignment:
                >>> R, t = ab1.align(ab2, rl_a="H:L", rl_b="H:L", ats="CA")

        Technical Details:
            Algorithm: Kabsch algorithm for optimal rotation
            - Computes centroids of both coordinate sets
            - Centers coordinates at origin
            - Calculates optimal rotation matrix via SVD
            - Applies rotation and translation to entire structure
            - Minimizes sum of squared distances

        Coordinate Transformation:
            1. All atom coordinates are transformed, not just selected atoms
            2. Transformation: new_pos = (old_pos @ R) + t
            3. Original target protein remains unchanged
            4. Current protein coordinates are permanently modified

        Examples:
            >>> p1 = Protein("1crn.pdb")
            >>> p2 = Protein("2crn.pdb")

            # Global structural alignment
            >>> R, t = p1.align(p2, ats="CA")
            >>> rmsd = p1.rmsd(p2, ats="CA")  # Calculate post-alignment RMSD
            >>> print(f"CA RMSD after alignment: {rmsd:.2f} Å")

            # Align specific chains
            >>> ab1 = Protein("5cil_model.pdb")
            >>> ab2 = Protein("5cil_native.pdb")
            >>> R, t = ab1.align(ab2, rl_a="H:L", rl_b="H:L", ats="N,CA,C,O")
            >>> print(f"Backbone alignment completed")

            # Domain-specific alignment with validation
            >>> try:
            ...     R, t = protein.align(reference, rl_a="A10-50", rl_b="A10-50")
            ...     print("Alignment successful")
            ... except Exception as e:
            ...     print(f"Alignment failed: {e}")

        Quality Assessment:
            Pre-alignment RMSD:
                >>> rmsd_before = p1.rmsd(p2, rl_a, rl_b, ats, align=False)

            Post-alignment RMSD:
                >>> R, t = p1.align(p2, rl_a, rl_b, ats)
                >>> rmsd_after = p1.rmsd(p2, rl_a, rl_b, ats, align=False)
                >>> print(f"RMSD improvement: {rmsd_before:.2f} → {rmsd_after:.2f} Å")

        Error Conditions:
            - Mismatched residue count between rl_a and rl_b
            - Empty residue selections
            - Missing atoms in selected atom types
            - Insufficient atoms for meaningful alignment (<3 points)

        Performance Notes:
            - Alignment speed depends on number of atoms, not residues
            - CA-only alignment is fastest and usually sufficient
            - All-atom alignment is most precise but slower
            - Memory usage scales with total protein size (all coordinates transformed)

        Caution:
            This method permanently modifies the current protein's coordinates.
            Use protein.clone().align() if you need to preserve the original structure.
            The transformation affects the entire protein, not just the aligned region.

        Mathematical Foundation:
            The method implements the Kabsch algorithm for optimal structural alignment:
            1. Center both coordinate sets at their centroids
            2. Compute cross-covariance matrix H = A^T * B
            3. Perform SVD: H = U * S * V^T
            4. Optimal rotation: R = V * U^T (with det(R) = 1)
            5. Optimal translation: t = centroid_B - centroid_A * R
        """
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in align()!")
        rl_a=RL(self, rl_a)
        rl_b=RL(target_p, rl_b)
        if len(rl_a)!=len(rl_b):
            raise Exception(f"Two selections have different residues: {len(rl_a)}, {len(rl_b)}.")
        if len(rl_a)==0:
            raise Exception(f"Empty residue selection: {len(rl_a)}, {len(rl_b)}.")
        (rsi_a, atsi_a, rsi_b, atsi_b, res_a, res_b)=self._get_xyz_pair(target_p, rl_a, rl_b, ats)
        if (res_a.shape != res_b.shape):
            raise Exception(f"Two selections have different atoms after masking: {res_a.shape}, {res_b.shape}.")
        R,t=rot_a2b(res_a,res_b)
        M=self.data.atom_positions
        n=M.shape
        mask=self.data.atom_mask.ravel() > 0
        Ma=M.reshape(-1,3)
        Mb=rot_a(Ma[mask], R, t)
        Ma[mask]=Mb
        self.data.atom_positions[...]=Ma.reshape(*n)
        return (R, t)

    @staticmethod
    def locate_dssp() -> Dict[str, str]:
        # default values used by Biopython
        bin="dssp"
        version="3.9.9"

        def search_cmd():
            """Get the appropriate search command for the current OS.

            Returns:
                str: 'where' for Windows, 'which' for Unix-like systems.
            """
            import platform
            return "where" if platform.system() == "Windows" else "which"
        search=search_cmd()
        for DSSP in ["mkdssp", "dssp"]:
            bin=util.unix(f"{search} {DSSP}", l_print=False).split("\n")[0]
            if bin!="": break
        if bin!="":
            S=[x for x in util.unix(f"{bin} --version", l_print=False).split("\n") if ' version ' in x]
            m=None
            if len(S):
                m=re.search(r'version\s+(?P<ver>[\d.]+)$', S[0])
                if m is not None: version=m.group('ver')

        else:
            print("WARNING: mkdssp is not found, please consider install it with: conda install sbl::dssp")
        return {"DSSP":bin, "dssp_version":version}

    def dssp(self, simplify: bool = False, DSSP: str = "mkdssp", dssp_version: str = "3.9.9") -> Dict[str, str]:
        """Calculate secondary structure using DSSP (Define Secondary Structure of Proteins).

        Computes secondary structure assignments for all residues using the DSSP
        algorithm, which analyzes hydrogen bonding patterns and geometric criteria
        to classify each residue into standard secondary structure categories.

        Args:
            simplify (bool, optional): Whether to use simplified 3-state classification:
                - False: 8-state DSSP classification (default)
                - True: 3-state simplified (helix, strand, loop)
            DSSP (str, optional): Path to DSSP executable. Common values:
                - "mkdssp": Modern DSSP implementation (recommended)
                - "dssp": Legacy DSSP executable
                - "/path/to/dssp": Custom installation path
            dssp_version (str, optional): DSSP version for BioPython compatibility.
                Defaults to "3.9.9". Check with `dssp --version` command.

        Returns:
            dict: Secondary structure assignments by chain:
                {chain_id: secondary_structure_string}

        Secondary Structure Codes:
            8-State Classification (default):
                - 'H': α-helix (4-turn helix)
                - 'B': β-bridge (isolated β-strand)
                - 'E': β-strand (extended strand in β-sheet)
                - 'G': 3₁₀-helix (3-turn helix)
                - 'I': π-helix (5-turn helix)
                - 'T': Turn (hydrogen bonded turn)
                - 'S': Bend (other turn, not hydrogen bonded)
                - 'C': Coil (irregular, no specific structure)

            3-State Simplified (simplify=True):
                - 'a': Alpha structures (H, G, I → helix)
                - 'b': Beta structures (E, B → strand)
                - 'c': Coil structures (S, T, C → loop)

        Examples:
            >>> p = Protein("1crn.pdb")

            # Standard 8-state DSSP classification
            >>> ss_dict = p.dssp()
            >>> print(ss_dict)
            {'A': 'CCCHHHHHHHHHHCCCEEEEECCCHHHHHHHHHHHHHHCCCC'}

            # Simplified 3-state classification
            >>> ss_simple = p.dssp(simplify=True)
            >>> print(ss_simple)
            {'A': 'cccaaaaaaaaaacccbbbbbcccaaaaaaaaaaaaaaacccc'}

            # Multi-chain structure
            >>> p_complex = Protein("5cil.pdb")
            >>> ss_all = p_complex.dssp()
            >>> for chain, ss in ss_all.items():
            ...     print(f"Chain {chain}: {len(ss)} residues")
            ...     print(f"  Helix content: {ss.count('H')/len(ss)*100:.1f}%")
            ...     print(f"  Sheet content: {ss.count('E')/len(ss)*100:.1f}%")

            # Custom DSSP installation
            >>> ss = p.dssp(DSSP="/usr/local/bin/mkdssp", dssp_version="4.0.0")

        Secondary Structure Analysis:
            Content Analysis:
                >>> ss = p.dssp()['A']
                >>> helix_pct = (ss.count('H') + ss.count('G') + ss.count('I')) / len(ss)
                >>> sheet_pct = (ss.count('E') + ss.count('B')) / len(ss)
                >>> print(f"Helix: {helix_pct*100:.1f}%, Sheet: {sheet_pct*100:.1f}%")

            Structural Regions:
                >>> ss = p.dssp(simplify=True)['A']
                >>> regions = []
                >>> current_type = ss[0]
                >>> start = 0
                >>> for i, ss_type in enumerate(ss[1:], 1):
                ...     if ss_type != current_type:
                ...         regions.append((current_type, start, i-1))
                ...         current_type = ss_type
                ...         start = i
                >>> regions.append((current_type, start, len(ss)-1))

        Installation and Setup:
            Install DSSP:
                >>> # Via conda (recommended)
                >>> # conda install sbl::dssp
                >>> # or conda install conda-forge::dssp

            Find Installation:
                >>> dssp_info = Protein.locate_dssp()
                >>> print(dssp_info)
                {'DSSP': '/usr/local/bin/mkdssp', 'dssp_version': '3.9.9'}

        Error Handling:
            Common Issues and Solutions:
                Empty Output: Check DSSP installation and version compatibility
                Version Errors: Update dssp_version parameter to match installed version
                Path Issues: Provide full path to DSSP executable

            Troubleshooting:
                >>> # Check DSSP availability
                >>> dssp_info = Protein.locate_dssp()
                >>> if not dssp_info['DSSP']:
                ...     print("DSSP not found. Install with: conda install sbl::dssp")

        """
        from Bio.PDB.DSSP import dssp_dict_from_pdb_file
        tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".pdb").name
        self.save(tmp)
        dssp_tuple = dssp_dict_from_pdb_file(tmp, DSSP=DSSP, dssp_version=dssp_version)
        os.remove(tmp)
        dict_ss={} # secondary structure by chain
        for k,v in dssp_tuple[0].items():
            if k[0] not in dict_ss:
                dict_ss[k[0]]=[]
                # a new chain
                prev_chain, prev_resi='', 0
            if prev_chain==k[0]:
                # insert GAP
                for i in range(prev_resi, k[1][1]-1):
                    dict_ss[k[0]].append("X")
            dict_ss[k[0]].append(v[1])
            prev_chain, prev_resi=k[0], k[1][1]
        dict_ss={k:"".join(v) for k,v in dict_ss.items()}
        if simplify:
            MAP={"G":"a", "I":"a", "H":"a", "E":"b", "B":"b", "S":"c", "C":"c", "T":"c"}
        # https://en.wikipedia.org/wiki/DSSP_(algorithm)
        # These eight types are usually grouped into three larger classes:
        # helix (G, H and I), strand (E and B) and loop (S, T, and C, where C sometimes is represented also as blank space)
            dict_ss={k:"".join([MAP.get(x,x) for x in v]) for k,v in dict_ss.items()}
        return dict_ss

    def atom_coordinate(self, rs = None, atom='CA'):
        """
        Get the coordinates of a specific atom in the protein structure.

        Args:
            rs (ContigType): The residue selection (default: None).
            atom (str): The atom name (default: 'CA').

        Returns:
            np.ndarray: The coordinates of the specified atom.
            Note: CB for Gly will be replaced by CA
        """
        def coordinate(resi, atom):
            coord = self.data.atom_positions[resi][:, atom]
            mask = self.data.atom_mask[resi][:, atom] == 0
            coord[mask] = np.nan
            return coord
        
        atom = ATS.i(atom)
        resi = self.rs(rs).data
        coord = coordinate(resi, atom)
        if atom == ATS.i("CB"):
            gly = self.data.aatype[resi] == afres.restype_order['G']
            if sum(gly) > 0:
                coord_ca = coordinate(resi, ATS.i("CA"))
                coord[gly] = coord_ca[gly]
        return coord

    def internal_coord(self, rs: ContigType = None, MaxPeptideBond: float = 1.4) -> pd.DataFrame:
        obj=self.extract(rs, inplace=False)
        chains=obj.chain_id()
        q=obj.to_biopython()
        from Bio.PDB.ic_rebuild import structure_rebuild_test
        from Bio.PDB.internal_coords import IC_Chain
        IC_Chain.MaxPeptideBond=MaxPeptideBond
        out=[]
        bonds=['-1C:N','N:CA','CA:C']
        angles=['phi','psi','omg','tau','chi1','chi2','chi3','chi4','chi5']
        n_chi={k:len(v) for k,v in afres.chi_angles_atoms.items()}
        n_chi['ARG']+=1 # DeepMind does not use chi5, as it's always 0+/-5
        for k in chains:
            myChain = q[0][k]
            # compute bond lengths, angles, dihedral angles
            myChain.atom_to_internal_coordinates(verbose=True)
            # check myChain makes sense (can get angles and rebuild same structure)
            resultDict = structure_rebuild_test(myChain)
            assert resultDict['pass'] == True
            for r in myChain.get_residues():
                (_, id, code), aa=r.get_id(), r.get_resname()
                resn=f"{id}{code.strip()}"
                resn_i=int(id)
                resi=obj.res_map[k+resn]
                one=[k, resn, resn_i, resi, afres.restype_3to1.get(aa,aa)]
                one.extend([r.internal_coord.get_length(b) for b in bonds])
                one.extend([r.internal_coord.get_angle(a) for a in angles[:4]])
                one.extend([r.internal_coord.get_angle(a) for a in angles[4:4+n_chi[aa]]])
                one.extend([np.nan for a in angles[4+n_chi[aa]:]])
                out.append(one)
        t=pd.DataFrame(out, columns=['chain','resn','resn_i','resi','aa']+bonds+angles)
        return t

    @staticmethod
    def PyMOL() -> Any:
        """Create and return a PyMOL wrapper object for advanced molecular visualization.

        Creates an instance of the custom PyMOL wrapper class that provides
        programmatic access to PyMOL for script execution, structure manipulation,
        and automated visualization tasks.

        Returns:
            PyMOL: A PyMOL wrapper object that allows programmatic control
                  of PyMOL sessions, including loading structures, running
                  commands, and generating visualizations.

        Raises:
            SystemExit: If PyMOL is not installed or accessible

        Examples:
            >>> pm = Protein.PyMOL()
            >>> pm.run("load structure.pdb")
            >>> pm.run("color blue, chain A")
            >>> pm.run("save session.pse")
            >>> pm.close()

        Note:
            Requires PyMOL to be installed. Install with:
            conda install conda-forge::pymol-open-source
        """
        check_PyMOL()
        from .mypymol import PyMOL
        return PyMOL()

    @staticmethod
    def Mol3D() -> Any:
        check_Mol3D()
        from .mol3D import Mol3D
        return Mol3D()

    @staticmethod
    def PyMOL3D() -> Any:
        check_PyMOL()
        from .pymol3D import PyMOL3D
        return PyMOL3D()

    @staticmethod
    def fold(seq: str, gap: int = 50) -> Optional['Protein']:
        """Input sequence, missing residues can be represented as Xs, chains are concatenated by ':'.
        E.g., for 5cli, use sequence:

        Note: ESMFold can only predict a monomer, multimers are predicted by concatenating chains with poly-glycine.
        gap defines the number of glycines used to link chains.
        The final sequence length cannot exceed 400 for ESMFold service

        WARNING: ESMFold only allows infrequent submissions, so this is not suitable for large-scale structure prediction

        Return: predicted Protein object
        """
        from .myalphafold.common.protein import PDB_CHAIN_IDS
        c_pos={}
        b=0 #begin counter
        i_len=0 # accumulated sequence length without gap
        X_pos=[]
        seq=seq.upper()
        for i,s in enumerate(seq.split(":")):
            chain=PDB_CHAIN_IDS[i]
            # record position of X, without gap
            if len(s.replace("X", ""))==0: continue
            X_pos.extend([j+i_len for j,aa in enumerate(s) if aa=='X'])
            i_len+=len(s)
            # record chain position (b,e) with gap
            e=b+len(s)-1 # end position
            c_pos[chain]=(b, e)
            b=e+gap+1
        seq2 = seq.replace("X", "G").replace(":", "G"*gap)

        if len(seq2)>400:
            raise Exception(f"ESMFold service cannot handle sequence (including gaps) longer than 400 residues.")
        try:
            foldedP = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=seq2, verify=False)
        except Exception as error:
            print("An exception was raised when calling the API", error)
            return None

        pdbstr=foldedP.content.decode("utf-8")
        if 'TITLE     ESMFOLD V1 PREDICTION FOR INPUT' not in pdbstr:
            raise Exception(f"ESMFold service did not return valid PDB content:\n{pdbstr}")
        #util.save_string('t.pdb', pdbstr)
        p=Protein(pdbstr)
        p.split_chains(c_pos, inplace = True)
        return (p.extract(~p.rs(X_pos)))

    @staticmethod
    def find_aligned_positions(seq1: str, seq2: str, is_global: bool = False, match: int = 2,
                             xmatch: int = -1, gap_open: float = -1.5, gap_ext: float = -0.2) -> Tuple[Tuple[Any, float], List[int], List[int]]:
        """
        is_global=True: Perform global alignment, otherwise, local alignment
        """

        #from Bio import pairwise2
        #from Bio.pairwise2 import format_alignment

        #if is_global:
        #    alignments = pairwise2.align.globalms(seq1, seq2, match, xmatch, gap_open, gap_ext)
        #else:
        #    alignments = pairwise2.align.localms(seq1, seq2, match, xmatch, gap_open, gap_ext)
        from Bio.Align import PairwiseAligner

        aligner = PairwiseAligner()
        aligner.match_score = match
        aligner.mismatch_score = xmatch
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_ext

        # Set alignment mode
        aligner.mode = 'global' if is_global else 'local'

        # Perform alignment
        alignments = aligner.align(seq1, seq2)

        # Get the best alignment
        best_alignment = alignments[0]
        seq_a, seq_b = best_alignment[0], best_alignment[1]

        total = identical = 0
        aligned_positions_seq1 = []
        aligned_positions_seq2 = []
        seq1_index = seq2_index = 0
        for res1, res2 in zip(seq_a, seq_b):
            l_next1=res1 not in ('-',)
            l_next2=res2 not in ('-',)
            if l_next1 and l_next2:
                if res1!='X' and res2!='X':
                    aligned_positions_seq1.append(seq1_index)
                    aligned_positions_seq2.append(seq2_index)
                    total += 1
                    if res1 == res2:
                        identical+=1
            if l_next1: seq1_index += 1
            if l_next2: seq2_index += 1
        identity = identical/total *100 if total > 0 else 0.0
        return ((best_alignment, identity), aligned_positions_seq1, aligned_positions_seq2)

    def _rl_align(self, obj_b, chain_a, chain_b, is_global=False, match=2, xmatch=-1, gap_open=-1.5, gap_ext=-0.2):
        """Align chain_a and chain_b of two objects, return the aligned residue lists
            This can be used to generate RL for alignment between two proteins where sequence are not identifcal
            This method is to be used by rl_align()
        """
        seq1=self.seq_dict()[chain_a]
        seq2=obj_b.seq_dict()[chain_b]
        posX2pos1=Protein.posX2pos(seq1)
        posX2pos2=Protein.posX2pos(seq2)
        _, pos1, pos2 = Protein.find_aligned_positions(seq1, seq2, is_global=is_global, match=match, xmatch=xmatch, gap_open=gap_open, gap_ext=gap_ext)

        print(_[0])

        pos1=[posX2pos1[x] for x in pos1]
        pos2=[posX2pos2[x] for x in pos2]
        pos1=np.array(pos1)+self.chain_pos()[chain_a][0]
        pos2=np.array(pos2)+obj_b.chain_pos()[chain_b][0]
        return RL(self, pos1), RL(obj_b, pos2)

    def rl_align(self, obj_b, chains_a=None, chains_b=None, is_global=False, match=2, xmatch=-1, gap_open=-1.5, gap_ext=-0.2):
        """Align two objects. Specific the list of chains to be aligned, so
            chains_a and chains_b are two lists with corresponding chains in the same order
            if set to None, we will find all common chain names and assume they are used.
                chains_a=["A","B","C"] chains_b=["H","L","P"] will match A to H
                In this case, if left as None, it's an error as there is no shared chain name
            If leave as None, both should be None
        """
        if chains_a is None or chains_b is None:
            chains_a=chains_b=[x for x in self.chain_id() if x in obj_b.chain_id() ]
        else:
            if type(chains_a) is str: chains_a=[chains_a]
            if type(chains_b) is str: chains_b=[chains_b]
        out_a=[]
        out_b=[]
        for x,y in zip(chains_a, chains_b):
            rl_a, rl_b=self._rl_align(obj_b, x, y, is_global=is_global, match=match, xmatch=xmatch, gap_open=gap_open, gap_ext=gap_ext)
            if len(rl_a)>0:
                out_a.append(rl_a)
                out_b.append(rl_b)
            else:
                print(f"WARNING> No aligned residues between (obj_a, chain {x}) and (obj_b, chain {y})")
        if len(out_a)==0:
            return self.rs('NONE'), obj_b.rs('NONE')
        return RL._or(*out_a), RL._or(*out_b)

    def rl_align_multi(self, obj_b, is_global=False, match=2, xmatch=-1, gap_open=-1.5, gap_ext=-0.2, min_identity=0.3):
        """Automatically map chains between two objects. Ignore chain labels, but consider both sequence similarity
        and structure similarity.

        For the first aligned pair, we accept them without apply identity cutoff
        Subsequent pairs must have identity above min_identity to be considered

        Return: (chain_map_a, chain_map_b, rl_a, rl_b)
            chain_map_a: the list of chains from self
            chain_map_b: the list of chains from obj_b
            rl_a, rl_b: corresponding RL objects
        """
        p, q=self, obj_b
        c_pos_a=p.chain_pos()
        c_pos_b=q.chain_pos()
        c_seq_a=p.seq_dict()
        c_seq_b=q.seq_dict()
        chain_a=[x for x in p.chain_id()]
        chain_b=[x for x in q.chain_id()]
        n,m = len(c_seq_a), len(c_seq_b)
        sim = np.zeros((n,m), dtype=float)
        identity = np.zeros((n,m), dtype=float)
        rs_pair = np.zeros((n, m), dtype=object)

        # sequence align all chain pairs, memorize algined residues in rs_pair
        for i,a in enumerate(chain_a):
            seq_a=c_seq_a[a]
            posX2pos_a=Protein.posX2pos(seq_a)
            for j,b in enumerate(chain_b):
                seq_b=c_seq_b[b]
                posX2pos_b=Protein.posX2pos(seq_b)
                _, pos1, pos2 = Protein.find_aligned_positions(seq_a, seq_b, is_global=is_global, match=match, xmatch=xmatch, gap_open=gap_open, gap_ext=gap_ext)
                #N=len(_[0])
                #print(a, b, _[1])
                sim[i,j]=_[0].score
                identity[i,j]=_[1]
                pos1=[posX2pos_a[x] for x in pos1]
                pos2=[posX2pos_b[x] for x in pos2]
                rs_pair[i,j]=(p.rl(np.array(pos1)+c_pos_a[a][0]), q.rl(np.array(pos2)+c_pos_b[b][0]))

        chain_map_a=[]
        chain_map_b=[]
        rl_a=p.rl("NONE")
        rl_b=q.rl("NONE")

        while sim.shape[0]>0 and sim.shape[1]>0:
            max_score = np.max(sim)

            # if sequence pair are similar within 10%, we go for structure RMSD similarity
            # this is to break ties for homomultimers
            threshold = max_score * 0.9

            # Find all indices within 10% of the max score
            if len(rl_a)==0: # we don't apply min_identity for the first pair
                candidate_indices = np.argwhere(sim >= threshold)
            else:
                candidate_indices = np.argwhere((sim >= threshold) & (identity >= min_identity))
            # Among candidates, find the one with the highest structure score

            #print(">>>>>>", sim, candidate_indices)

            if len(candidate_indices)==0:
                break
            elif len(candidate_indices)>1:
                # if ties, we choose the pair with smallest RMSD
                # This is useful, if there multiple copies of antibodies, if we align two H chains first
                # then this helps to picking the right pair of L chains from the same antibody
                best_pair = None
                best_rl = None
                best_struct_score = np.inf
                for i, j in candidate_indices:
                    rl_a2=rl_a+rs_pair[i,j][0]
                    rl_b2=rl_b+rs_pair[i,j][1]
                    #print(rl_a2, rl_b2)
                    struct_score=p.rmsd(q, rl_a2, rl_b2, ats="N,CA,C,O", align=True)
                    #print("sssssssss>", struct_score, (i, j), (chain_a[i], chain_b[j]))
                    if struct_score < best_struct_score:
                        best_struct_score = struct_score
                        best_pair = (i, j)
                        best_rl = rs_pair[i,j]
            else:
                best_pair = candidate_indices[0]
                best_rl = rs_pair[best_pair[0], best_pair[1]]

            chain_map_a.append(chain_a[best_pair[0]])
            chain_map_b.append(chain_b[best_pair[1]])
            rl_a += best_rl[0]
            rl_b += best_rl[1]

            # Remove the selected row and column from sim and rs_pair
            (i, j) = best_pair
            sim = np.delete(sim, i, axis=0)
            sim = np.delete(sim, j, axis=1)
            identity = np.delete(identity, i, axis=0)
            identity = np.delete(identity, j, axis=1)
            rs_pair = np.delete(rs_pair, i, axis=0)
            rs_pair = np.delete(rs_pair, j, axis=1)
            ka=chain_a.pop(i)
            kb=chain_b.pop(j)

            # print alignment for visual inspection
            _, pos1, pos2 = Protein.find_aligned_positions(c_seq_a[ka], c_seq_b[kb], is_global=is_global, match=match, xmatch=xmatch, gap_open=gap_open, gap_ext=gap_ext)
            print(f"Chain Mapped: {ka} <> {kb}")
            print(_[0])
        return (chain_map_a, chain_map_b, rl_a, rl_b)


    def align_two(self, obj_b, chain_a=None, chain_b=None, is_global=False, match=2, xmatch=-1, gap_open=-1.5, gap_ext=-0.2, auto_chain_map=False):
        """
        self is the reference protein, chains in obj_b will be renamed to match chains in self
        return two new objects: (new_reference, new_obj_b, RL_old_ref, RL_old_b), where only aligned residues are kept
        new pairs can be directly used for rmsd and dockQ calculation
        """
        if auto_chain_map:
            chain_a, chain_b, rl_a, rl_b = self.rl_align_multi(obj_b, is_global=is_global, match=match, xmatch=xmatch, \
                                    gap_open=gap_open, gap_ext=gap_ext)
        else:
            if chain_a is None:
                chain_a=chain_b=[x for x in self.chain_id() if x in obj_b.chain_id() ]
            rl_a, rl_b = self.rl_align(obj_b, chain_a, chain_b, is_global, match, xmatch, gap_open, gap_ext)
        p_a=self.extract(rl_a)
        p_b=obj_b.extract(rl_b)
        p_b.rename_chains({k:v for k,v in zip(chain_b, chain_a)}, inplace=True)
        return (p_a, p_b, rl_a, rl_b)

    @staticmethod
    def posX2pos(seq: str) -> np.ndarray:
        """creat a mapping betweem positions in seq containing X and seq without X"""
        X=np.arange(len(seq))
        noX=[]
        pos=0
        for i,x in enumerate(seq.upper()):
            noX.append(pos)
            if x!="X": pos+=1
        return np.array(noX, dtype=int)

    @staticmethod
    def seq2antibody(seq, chain_type=None, species="human", scheme="chothia"):
        """
        If you are looking for a specific chain_type, you can specify "H", ["H", "L"], etc.
                The default None means ["H", "L", "K"]. "L" will include "K" as well.

        return (chain_type, (var_start, var_end), ((b1,e1,s1), (b2,e2,s2), (b3,e3,s3)))
        """
        check_Ab()
        if chain_type is None:
            chain_type=["H","L","K"]
        elif type(chain_type) is str and chain_type in("L", "K"):
            chain_type=["L","K"]
        x=antibody.Antibody(seq, chain_type=chain_type, species=species, scheme=scheme)
        if x.chain_type is None:
            return (None, None, None, None)
        numbering=np.array([str(a)+b for a,b in zip(x.numbering.num_int, x.numbering.num_letter)])
        return (x.chain_type, (x.start, x.end), x.cdrs, numbering)

    def rs_antibody(self, chains=None, chain_type=None, species="human", scheme="chothia", set_b_factor=False):
        """Identify and extract antibody variable domains and CDR regions from protein structure.

        Automatically detects antibody chains in the structure using sequence-based
        analysis and returns residue selections for CDR regions and variable domains.
        Supports multiple antibody numbering schemes and can set B-factors for
        visualization purposes.

        Args:
            chains (str, list, optional): Specific chains to analyze for antibody content:
                - None: Analyze all chains in structure (default)
                - String: Single chain like "H"
                - List: Multiple chains like ["H", "L", "P"]
            chain_type (str, list, optional): Expected antibody chain types to search for:
                - None: Search for ["H", "L", "K"] (heavy, lambda, kappa)
                - "H": Heavy chains only
                - ["H", "L"]: Heavy and light chains
                - ["L", "K"]: Light chains (both lambda and kappa)
            species (str, optional): Species for antibody detection algorithms:
                - "human": Human antibody sequences (default)
                - "mouse": Mouse antibody sequences
                - "rabbit": Rabbit antibody sequences
            scheme (str, optional): Antibody numbering scheme for CDR definition:
                - "chothia": Chothia scheme (structural, default)
                - "kabat": Kabat scheme (sequence-based)
                - "imgt": IMGT scheme (international standard)
                - "chothia_consensus": Chothia with consensus boundaries
                - combinations, e.g., "chothia,imgt" use the conservative definition by combining both schemes
            set_b_factor (bool, optional): Whether to modify B-factors for visualization:
                - False: Leave B-factors unchanged (default)
                - True: Set B-factors to color-code CDRs and chains

        Returns:
            tuple: (rs_cdr, rs_var, chain_type_dict, c_cdr_by_chain) where:
                - rs_cdr (RS): Combined residue selection for all CDR regions
                - rs_var (RS): Combined residue selection for all variable domains
                - chain_type_dict (dict): {chain_id: type} mapping ("H", "L", "K")
                - c_cdr_by_chain (dict): {chain_id: [cdr1_rs, cdr2_rs, cdr3_rs]}

        CDR Numbering Schemes:
            Chothia Scheme (structural):
                - Based on 3D structural analysis
                - CDRs defined by loop conformations
                - Most accurate for structural studies

            Kabat Scheme (sequence):
                - Based on sequence variability
                - Traditional immunology numbering
                - Good for sequence analysis

            IMGT Scheme (standardized):
                - International standardization
                - Consistent across species
                - Useful for comparative studies

        B-factor Color Coding (set_b_factor=True):
            Variable Domains:
                - Heavy chains: Green (B-factor = 0.4)
                - Light chains: Yellow (B-factor = 0.25)

            CDR Regions:
                - CDR1: Light blue (B-factor = 0.8)
                - CDR2: Medium blue (B-factor = 0.9)
                - CDR3: Dark blue (B-factor = 1.0)

            Non-Antibody Chains:
                - Warm colors (B-factors 0.0-0.25)

        Examples:
            >>> p = Protein("5cil.pdb")  # Antibody-antigen complex

            # Basic antibody detection
            >>> rs_cdr, rs_var, chain_types, cdrs = p.rs_antibody()
            >>> print(f"Found antibody chains: {list(chain_types.keys())}")
            >>> print(f"Chain types: {chain_types}")

            # Extract only CDR regions
            >>> cdr_structure = p.extract(rs_cdr)
            >>> cdr_structure.save("cdrs_only.pdb")

            # Extract variable domains only
            >>> var_structure = p.extract(rs_var)
            >>> print(f"Variable domains: {len(var_structure)} residues")

            # Analyze specific chains with IMGT numbering
            >>> rs_cdr, rs_var, types, cdrs = p.rs_antibody(
            ...     chains=["H", "L"], scheme="imgt", set_b_factor=True
            ... )
            >>> p.show(color="b")  # Visualize with CDR coloring

        Chain-Specific Analysis:
            Individual CDR Analysis:
                >>> rs_cdr, rs_var, chain_types, cdrs_by_chain = p.rs_antibody()
                >>> for chain, cdr_list in cdrs_by_chain.items():
                ...     print(f"Chain {chain} ({chain_types[chain]}):")
                ...     for i, cdr_rs in enumerate(cdr_list, 1):
                ...         if cdr_rs is not None:
                ...             print(f"  CDR{i}: {len(cdr_rs)} residues")
                ...             print(f"  Sequence: {cdr_rs.seq() if hasattr(cdr_rs, 'seq') else 'N/A'}")

            Heavy vs Light Chain Separation:
                >>> heavy_cdrs = rs_or(*[cdrs_by_chain[c] for c in cdrs_by_chain
                ...                     if chain_types.get(c) == "H"])
                >>> light_cdrs = rs_or(*[cdrs_by_chain[c] for c in cdrs_by_chain
                ...                     if chain_types.get(c) in ["L", "K"]])

        Visualization and Analysis:
            CDR-Focused Visualization:
                >>> rs_cdr, _, _, _ = p.rs_antibody(set_b_factor=True)
                >>> p.show(color="b", style="cartoon")  # Color by CDR regions

        Species-Specific Detection:
            Different species have different antibody sequence patterns:
                >>> # Mouse antibody analysis
                >>> rs_cdr, _, types, _ = p.rs_antibody(species="mouse")

                >>> # Rabbit antibody (often VHH/single-domain)
                >>> rs_cdr, _, types, _ = p.rs_antibody(species="rabbit", chain_type=["H"])

        Technical Details:
            Detection Algorithm:
                1. Extracts sequences from specified chains
                2. Runs ANARCI antibody numbering algorithm
                3. Identifies variable domain boundaries
                4. Maps CDR positions based on numbering scheme
                5. Converts sequence positions to structure indices

        Note:
            This method requires the ANARCI package for antibody sequence analysis.
            Install with: `conda install bioconda::anarci`. The detection algorithm
            works best with complete variable domains and may miss heavily truncated
            or highly modified antibody sequences. For nanobodies (VHH), use
            chain_type=["H"] and species="llama" if available.
        """
        check_Ab()
        if chains is None:
            chains=self.chain_id()
        elif type(chains) is str:
            chains=[chains]
        c_seq=self.seq_dict(gap="X")
        c_pos=self.chain_pos()
        rs_var=self.rs("NONE")
        rs_cdr=self.rs("NONE")

        cdrs_pos={}
        c_chain_type={}
        c_cdr_by_chain={}
        for k in chains:
            out=Protein.seq2antibody(c_seq[k], chain_type=chain_type, species=species, scheme=scheme)
            if out[0] is None:
                continue
            posX2pos=Protein.posX2pos(c_seq[k])
            c_chain_type[k], (b,e), cdrs, _ = out
            b,e=posX2pos[b], posX2pos[e]
            rs_var+=self.rs(np.arange(b+c_pos[k][0], e+c_pos[k][0]+1, dtype=int))

            cdrs_pos[k]=[]
            _cdrs=[]
            for (b,e,s) in cdrs:
                if b is None:
                    _cdrs.append(self.rs('NONE'))
                    continue
                b,e=posX2pos[b], posX2pos[e]
                cdrs_pos[k].append((b+c_pos[k][0], e+c_pos[k][0]))
                _cdrs.append(self.rs(np.arange(b+c_pos[k][0], e+c_pos[k][0]+1, dtype=int)))
                rs_cdr+=_cdrs[-1]
            c_cdr_by_chain[k]=_cdrs

        if set_b_factor and len(rs_var):
            B=self.b_factors()
            B[:]=0.2
            # L/K: i=0,1,2, H: 3,4,5
            c_pos=self.chain_pos()
            ag=[x for x in self.chain_id() if x not in c_chain_type]
            n_ag=len(ag)
            # non-Ab chains colored as [0, 0.4)
            for i,k in enumerate(ag):
                b,e=c_pos[k]
                B[b:e+1]=i/n_ag*0.25
            for k, cdrs in c_cdr_by_chain.items():
                b,e=c_pos[k]
                # Ab light chain yellow, heavy chain green
                B[b:e+1]=0.4 if c_chain_type[k]=='H' else 0.25
                for i,rs in enumerate(cdrs):
                    v=[0.8, 0.9, 1.0][i]
                    if len(rs):
                        B[rs.data]=v
            self.b_factors(B)

        return rs_cdr, rs_var, c_chain_type, c_cdr_by_chain

    def truncate_antibody(self, chains=None, chain_type=None, species="human", scheme="chothia", set_b_factor=False, renumbering=False, inplace=False):
        """It first run rs_antibody, then truncate H or L/K chains removing any residues that are not part of the variable domain.
            Non-Ab chains are untouched.
        """
        check_Ab()
        rs_cdr, rs_var, c_chain_type, c_cdr = self.rs_antibody(chains=chains, chain_type=chain_type, species=species, scheme=scheme, set_b_factor=set_b_factor)
        contigs=[]
        for x in self.chain_id():
            if x not in c_chain_type:
                contigs.append(x)
        if len(contigs)>0:
            rs_var+=self.rs(":".join(contigs))
        obj=self.extract(rs_var, inplace=inplace)
        if renumbering:
            c_seq=obj.seq_dict()
            for k,v in c_chain_type.items():
                seq=c_seq[k]
                out=Protein.seq2antibody(seq, chain_type=v, species=species, scheme=scheme)
                #print(out[3])
                resn=[x for s,x in zip(seq, out[3]) if s!='X']
                obj.resn(resn, obj.rl(k))
        return obj

    @staticmethod
    def seq2Fv(seq, species="human"):
        """Extract Fv (variable fragment) regions from antibody sequence.

        Args:
            seq: Antibody sequence string.
            species: Species for numbering scheme, default "human".

        Returns:
            list: List of tuples containing (start_pos, end_pos, chain_type) for each Fv region found.
        """
        check_Ab()
        out=[]
        while True:
            chain_type, var_pos, cdrs, num = Protein.seq2antibody(seq, species=species, scheme="chothia")
            if chain_type is None: break
            out.append((var_pos[0], var_pos[1], chain_type))
            S=["X" if (i>=var_pos[0] and i<=var_pos[1]) else x for i,x in enumerate(seq) ]
            seq="".join(S)
        return out

    def rs_Fv(self, chain, species="human"):
        """Extract variable regions from a sequence, there can be multiple variable regions,
        e.g., heavy and light chains are fused in scFv
        """
        check_Ab()
        c_pos=self.chain_pos()
        c_seq=self.seq_dict()
        if chain not in c_pos:
            raise Exception(f"Chain name {chain} not found in protein")
        b,e=c_pos[chain]
        s_seq=c_seq[chain]
        posX2pos=Protein.posX2pos(s_seq)
        out = Protein.seq2Fv(s_seq, species=species)
        var_pos = [self.rs(np.arange(posX2pos[x]+b,posX2pos[y]+b+1))  for x,y,s in out]
        chain_type = [s for x,y,s in out]
        return (var_pos, chain_type)

    def scFv2mAb(self, chain=None, species="human"):
        """Convert scFv to mAb by splitting into heavy and light chains.
        If chain is None, it will automatically identify all possible scFv chains found in the structure.
        We assume there can only be at most one heavy and one light chain domain within an scFv chain.
        The heavy domain will inherit the original chain name and the light chain will be named as the lower case.

        Return: a new mAb object

        Example:
            >>> p = Protein("9cn2")
            >>> p = p.scFv2mAb()
            >>> print(p.seq_dict())
        """
        check_Ab()
        if chain is None:
            chain = self.chain_id()
        elif type(chain) is str:
            chain = [ chain ]

        objs = []
        for x in self.chain_id():
            if x in chain:
                var_pos, chain_type = self.rs_Fv(x, species=species)
                if len(var_pos) == 2: # an scFv chain is found
                    for rs, ty in zip(var_pos, chain_type):
                        p = self.extract(rs)
                        chain_name = x.upper() if ty=="H" else x.lower()
                        p.rename_chains({x: chain_name}, inplace=True)
                        objs.append(p)
                else:
                    objs.append(self.extract(x))
        return Protein.merge(objs)

    @staticmethod
    def abag_units(pdb, min_within_ab_contacts=20, min_ab_ag_contact=1, min_contact_dist=4.5):
        """
        Identify antibody-antigen interaction units from a PDB structure.

        This function analyzes a protein structure to identify antibody chains, group them
        into heavy/light pairs, and associate them with their corresponding antigen chains
        based on residue contact analysis.

        Args:
            pdb (str): Path to PDB/mmCIF file or PDB ID
            min_within_ab_contacts (int, optional): Minimum number of contacts required
                between heavy and light chains to be considered a pair. Defaults to 10.
            min_ab_ag_contact (int, optional): Minimum number of contacts required
                between antibody and antigen chains. Defaults to 1.
            min_contact_dist (float, optional): Maximum distance (Å) to consider
                residues in contact. Defaults to 4.5.

        Returns:
            list: List of dictionaries, each representing an Ab/Ag unit:
                [
                    {
                        'ab': [heavy_chain_id, light_chain_id],  # '' if no partner
                        'ag': [antigen_chain_id1, antigen_chain_id2, ...]
                    },
                    ...
                ]

        Example:
            >>> units = abag_units("8zua", min_within_ab_contacts=15, min_ab_ag_contact=2)
            >>> print(units)
            [{'ab': ['A', 'G'], 'ag': ['F']}, {'ab': ['B', 'D'], 'ag': ['C']},
            {'ab': ['E', 'I'], 'ag': ['H']}, {'ab': ['J', 'L'], 'ag': ['K']}]

            This output indicates:
            - Unit 1: Heavy chain 'A' paired with light chain 'G', binding antigen 'F'
            - Unit 2: Heavy chain 'B' paired with light chain 'D', binding antigen 'C'
            - Unit 3: Heavy chain 'E' paired with light chain 'I', binding antigen 'H'
            - Unit 4: Heavy chain 'J' paired with light chain 'L', binding antigen 'K'

        Note:
            - Heavy chains are always listed first in the 'ab' pair
            - Antigen chains can appear in multiple units if they contact different antibodies
            - Empty string ('') indicates missing heavy or light chain partner
            - The function uses rs_antibody() detection to identify antibody chains
            - scFv might be treated as either heavy or light based on anarci classification
        """
        p=Protein(pdb)
        rs_cdr, rs_var, c_chain_type, c_cdr_by_chain = p.rs_antibody()
        chain_used=[]
        n = len(p.chain_id())
        contact = np.zeros((n, n), dtype=int)
        ab_chains = list(c_chain_type.keys())
        all_chains = list(p.chain_id())
        for i,a in enumerate(all_chains):
            for j,b in enumerate(all_chains):
                if i < j:
                    rs_a, rs_b, dist = p.rs_around(a, rs_within=b, dist=min_contact_dist)
                    contact[i, j] = contact[j, i] = len(rs_a | rs_b)

        units = []
        ab_chain_indices = [all_chains.index(c) for c in ab_chains]
        remaining_ab = set(ab_chain_indices)

        while remaining_ab:
            ab1 = remaining_ab.pop()

            # Find the Ab partner with max contact (must meet minimum threshold)
            ab_contacts = [(i, contact[ab1, i]) for i in remaining_ab if contact[ab1, i] >= min_within_ab_contacts]
            if ab_contacts:
                ab2 = max(ab_contacts, key=lambda x: x[1])[0]
                remaining_ab.remove(ab2)
                # heavy chain first
                ab_pair = [ab1, ab2] if c_chain_type[all_chains[ab1]] == "H" else [ab2, ab1]
            else:
                ab_pair = [ab1, ""] if c_chain_type[all_chains[ab1]] == "H" else ["", ab1]

            # Find all Ag chains in contact with either Ab chain
            ag_chains = []
            for i in range(len(all_chains)):
                # Skip if this chain is an antibody chain
                if i in ab_chain_indices:
                    continue
                # Check if this chain has contact with any Ab chain in the pair
                if any(contact[ab, i] > min_ab_ag_contact for ab in ab_pair):
                    ag_chains.append(i)

            # Only remove used antibody chains (antigen chains can be shared)
            units.append({
                'ab': [all_chains[i] for i in ab_pair],
                'ag': [all_chains[i] for i in ag_chains]
            })

        return units

class ATS:
    """
    Atom Selection class for selecting specific atom types across residues.

    ATS (Atom Type Selection) provides a way to select specific atoms by their
    standard PDB names (e.g., "N", "CA", "C", "O"). The class stores atom selections
    as NumPy arrays of column indices corresponding to the 37 unique atom types
    found across all 20 amino acids.

    Data Structure:
        The .data attribute contains integer indices (0-36) representing atom types.
        These indices correspond to columns in the protein's atom_positions array.
        For example: N=0, CA=1, C=2, O=4, etc.

    Input Formats:
        - String: "N,CA,C,O" or "N CA C O" (comma or space separated)
        - List: ["N", "CA", "C", "O"]
        - Another ATS object: ATS(existing_ats)
        - Integer array: np.array([0, 1, 2, 4]) for N, CA, C, O
        - Special keywords: "ALL" (all atoms), "NONE"/"NULL"/"" (no atoms)

    Common Usage:
        Backbone atoms: ATS("N,CA,C,O")
        Side chain atoms: ~ATS("N,CA,C,O")
        Alpha carbons only: ATS("CA")
        All atoms: ATS("ALL") or ATS()

    Boolean Operations:
        - & (AND): atoms present in both selections
        - | (OR): atoms present in either selection
        - ~ (NOT): complement of the selection

    Examples:
        >>> ats = ATS("N,CA,C,O")              # backbone atoms
        >>> print(ats)                        # "N,CA,C,O"
        >>> ats_heavy = ATS("CA,CB,CG")       # heavy atoms
        >>> ats_combined = ats | ats_heavy    # union
        >>> ats_ca_only = ATS("CA")          # alpha carbons
        >>> ats_sidechain = ~ATS("N,CA,C,O") # all non-backbone atoms

        # Use with protein coordinates
        >>> p = Protein("1crn")
        >>> coords = p.data.atom_positions[:, ats.data]  # shape: (N_res, 4, 3)
        >>> mask = p.data.atom_mask[:, ats.data]        # shape: (N_res, 4)

    Note:
        Atom names are case-insensitive and converted to uppercase internally.
        The same ATS object can be applied to any protein structure regardless
        of which specific amino acids are present.
    """

    def __init__(self, ats: AtomSelectionType = None) -> None:
        """atoms should be comma/space-separated str, or a list of atoms"""
        if ats is None or (type(ats) in (str, np.str_) and ats.upper()=="ALL"):
            self.data=np.arange(afres.atom_type_num)
        elif type(ats) is str and ats.upper() in ("","NULL","NONE"):
            self.data=np.array([])
        # in jupyter, isinstance is not always work, probably due to auto reload
        # isinstance(ats, ATS) does not work, ats.__class__.__name__ is afpdb.afpdb.ATS, which ATS.__name__ is ATS
        elif type(ats).__name__=='ATS':
            self.data=ats.data.copy()
        elif isinstance(ats, np.ndarray):
            self.data=ats.copy()
        elif isinstance(ats, (int, np.integer)):
            self.data=np.array([ats])
        elif type(ats) is str:
            self.data=np.array([afres.atom_order[x.upper()] for x in re.split(r'\W+', ats)])
        elif type(ats) in (set,):
            self.data=np.array(list(ats))
        elif type(ats) in (list, tuple, pd.core.series.Series):
            # can be a list of atom names or atom ids
            self.data=np.array([afres.atom_order.get(x,x) for x in ats])
        elif isinstance(ats, (int, np.integer)):
            self.data=np.array([ats])
        self.data=np.unique(self.data)

    def is_empty(self) -> bool:
        return len(self.data)==0

    def is_full(self) -> bool:
        return len(np.unique(self.data))==afres.atom_type_num

    def not_full(self) -> bool:
        return not self.is_full()

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return ",".join([afres.atom_types[x] for x in self.data])

    def __and__(self, ats_b: 'ATS') -> 'ATS':
        """AND operation for two residue selections"""
        ats_b=ATS(ats_b)
        out=set(self.data) & set(ats_b.data)
        return ATS(out)

    def __or__(self, ats_b: 'ATS') -> 'ATS':
        """OR operation for two atom selections"""
        ats_b=ATS(ats_b)
        out=set(self.data) | set(ats_b.data)
        return ATS(out)

    def __invert__(self) -> 'ATS':
        return ATS([i for i in range(afres.atom_type_num) if i not in self.data])

    def __contains__(self, i: Union[str, int]) -> bool:
        if type(i) is str: i=self.i(i)
        return i in self.data

    def __add__(self, ats_b: 'ATS') -> 'ATS':
        return self.__or__(ats_b)

    def __sub__(self, ats_b: 'ATS') -> 'ATS':
        """In a but not in b"""
        ats_b=ATS(ats_b)
        return ATS(set(self.data)-set(ats_b.data))

    @staticmethod
    def i(atom: str) -> int:
        return afres.atom_order.get(atom, -1)

class RL:

    def __init__(self, p: 'Protein', rs: ContigType = None) -> None:
        """Selection can have duplicates, do not need to be ordered"""
        self.p=p
        # in jupyter, isinstance is not always work, probably due to auto reload
        if isinstance(rs, RL) or isinstance(rs, RS) or type(rs).__name__ in ('RL','RS'):
            self.data=rs.data.copy()
        elif isinstance(rs, np.ndarray):
            #already a selection
            self.data=rs.copy()
        elif isinstance(rs, (int, np.integer)):
            return np.array([rs])
        elif type(rs) in (set,):
            self.data=np.array(list(rs))
        elif type(rs) in (list, tuple, pd.core.series.Series):
            self.data=np.array(rs)
        elif rs is None:
            self.data=np.arange(len(self.p))
        elif type(rs) in (str, np.str_):
            self.data=self._rs(rs)
        else:
            util.error_msg(f"Unexpected selection type: {type(rs)}!")

        if len(self.data) and np.max(self.data)>=len(self.p):
            util.error_msg(f"Selection contains index {np.max(self.data.max)} exceeding protein length {len(self.p)}.")

    def is_empty(self) -> bool:
        return len(self.data)==0

    def not_full(self) -> bool:
        return not self.is_full()

    def unique(self) -> 'RS':
        """Remove duplicates, then reorder"""
        return RS(self.p, np.unique(self.data))

    def unique2(self) -> 'RS':
        """Remove duplicates without reordering"""
        return RS(self.p, self.data[np.sort(np.unique(self.data, return_index=True)[1])])

    def _rs(self, contig):
        """Convert a contig specification to residue selection.

        Args:
            contig: Contig specification (string, list, range, etc.) defining residue selection.

        Returns:
            numpy.ndarray: Array of residue indices matching the contig specification.
        """
        """res: list of residues chain:resi, if it's a single-letter str, we select the whole chain

        contig string: "A" for the whole chain, ":" separate chains, range "A2-5,10-13" (residue by residue index)
            "A:B1-10", "A-20,30-:B" (A from beginning to number 20, 30 to end, then whole B chain)
            Warning: residues are returned in the order specified by contig, we do not sort them by chain name!

        return indicies for atom_positions
        """
        if contig is None or (type(contig) is str and contig=='ALL'): return np.arange(len(self.p))
        # '' should be empty, b/c the str(empty_selection) should return ''
        if type(contig) is str and contig in ('','NONE','NULL'): return np.array([], dtype=int)

        c_pos=self.p.chain_pos()
        if contig is None: contig=''
        c_len={k:e-b+1 for k,(b,e) in c_pos.items()}
        chain_name=self.p.chain_id()
        c_contig={}
        contig=re.sub(r'\s','', contig)
        if len(contig)==0:
            # select all
            return np.arange(self.p.data.atom_positions.shape[0])
            #contig=":".join(chain_name)
        resi=self.p.data.residue_index
        # lookup (chain, resi) => position
        idx=[]
        chain_seen=[]
        prev_chain=""
        for s in contig.split(":"):
            # first letter must be chain name
            chain=s[0]
            #print(">>>>>>>>>>>>>>>>", chain, s, c_pos)
            if (chain not in c_pos):
                raise Exception(f"Chain {chain} not found!")
            #if chain!=prev_chain and chain in chain_seen:
            #    print(f"WARNING> contig should keep the segments for the same chain together {s}, if possible!")
            chain_seen.append(chain)
            prev_chain=chain
            for pair in s[1:].split(","):
                b,e=c_pos[chain]
                b2=e2=""
                if "-" in pair:
                    b2,e2=pair.split("-")
                elif len(pair)>0: # A23
                    b2=e2=pair
                if b2=="": b2=resi[b]
                if e2=="": e2=resi[e]
                b3=self.p.res_map.get(chain+b2, None)
                e3=self.p.res_map.get(chain+e2, None)
                if (b3 is None): util.error_msg(f"Cannot find residue: chain {chain} resi {b2}.")
                if (e3 is None): util.error_msg(f"Cannot find residue: chain {chain} resi {e2}.")
                #print(chain, b, e, b2, e2, b3, e3)
                for i in range(b3, e3+1):
                    idx.append(i)

        return np.array(idx)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, item: int) -> bool:
        return item in self.data

    def unique_name(self) -> List[str]:
        """convert residue index into name {chain}{res_index}"""
        chains=self.p.chain_id()
        return [f"{chains[self.p.data.chain_index[x]]}{self.p.data.residue_index[x]}" for x in self.data]

    def name(self) -> List[str]:
        """convert residue index into name {res_index}"""
        return [f"{self.p.data.residue_index[x]}" for x in self.data]

    def namei(self) -> List[int]:
        """Return the integer part of the residue_index, type is int"""
        return [int(re.sub(r'\D+$', '', self.p.data.residue_index[x])) for x in self.data]

    def insertion_code(self) -> List[str]:
        """Return the insertion code part of the residue_index, type is str"""
        return [re.sub(r'^\d+', '', self.p.data.residue_index[x]) for x in self.data]

    def chain(self, unique: bool = False) -> List[str]:
        chains=self.p.chain_id()
        chains=[chains[self.p.data.chain_index[x]] for x in self.data]
        if unique:
            return util.unique2(chains)
        return chains

    def aa(self) -> List[str]:
        aatype=self.p.data.aatype
        return [afres.restypes_with_x[aatype[i]] for i in self.data]

    def seq(self) -> str:
        return "".join(self.aa())

    @staticmethod
    def i(restype: str) -> int:
        if len(restype)==3: restype=afres.restype_3to1.get(restype.upper(), restype)
        return afres.restype_order.get(restype.upper(), -1)

    def df(self) -> pd.DataFrame:
        """List all the residues"""
        resi=self.data
        t=pd.DataFrame({'resi':resi, 'chain':self.chain(), 'resn':self.name(), 'namei':self.namei(), 'code':self.insertion_code() ,'aa':self.aa(), 'unique_name':self.unique_name()})
        return t

    def __str__(self) -> str:
        if len(self.data)==0: return ''

        t=self.df()

        c_len=self.p.len_dict()
        def contig(b, e):
            #Create contig for a continous segment within the same chain
            chain=t.loc[b, 'chain']
            if (e-b+1)==1: return f"{chain}{t.loc[b, 'resn']}"
            # full length
            if (e-b+1)==c_len[chain]: return chain
            if t.loc[b, 'resi']<0 or t.loc[e, 'resi']<0:
                raise Exception("Contig cannot represent residue with negative numbering: ({chain}{t.loc[b, 'resi']} - {chain}{t.loc[e, 'resi']})")
            return f"{chain}{t.loc[b, 'resn']}-{t.loc[e, 'resn']}"

        out=[]
        b=prev_resi=0
        prev_chain=''
        n=len(t)
        for i,r in t.iterrows():
            if i==0 or r['resi']!=prev_resi+1 or r['chain']!=prev_chain: # a new segment
                if i>0: out.append(contig(b, i-1))
                b=i
                prev_chain=r['chain']
            prev_resi=r['resi']
        out.append(contig(b, n-1))

        return ":".join(out)

    def __or__(self, rl_b: 'RL') -> 'RL':
        """or/add for two residue lists means we concatenate them"""
        return RL(self.p, np.concatenate((self.data, rl_b.data)))

    def __add__(self, rl_b: 'RL') -> 'RL':
        return self.__or__(rl_b)

    def __sub__(self, rl_b: 'RL') -> 'RL':
        """In a but not in b"""
        rs_b=RS(self.p, rl_b)
        out=[x for x in self.data if x not in rs_b.data]
        return RL(self.p, out)

    @staticmethod
    def _or(*L_rl: 'RL') -> 'RL':
        """or operation on a list of RL objects"""
        if len(L_rl)==0: util.error_msg("At least one RL objective should be provided!")
        if not isinstance(L_rl[0], RL): util.error_msg("The first objective must be an RL instance!")
        p=L_rl[0].p
        out=np.concatenate([RL(p, x).data for x in L_rl])
        return RL(p, out)

    def cast(self, q: 'Protein') -> 'RL':
        """Cast the selection to a selection of object q
            We convert the select to contig str, not via array indices, so that residue A10 remains residue A10
            regardless whether the A chain is in object q"""
        return q.rl(self.__str__())

    def split(self) -> List['RL']:
        """split a selection into continous residue selection"""
        resi=self.data
        resc=self.chain()
        out=[]
        prev_i=-np.inf
        prev_c=''
        seg=[]
        for k,i in zip(resc, resi):
            if k!=prev_c or i>prev_i+1:
                # start a new segment
                if len(seg):
                    out.append(seg)
                    seg=[]
            seg.append(i)
            prev_c, prev_i=k,i
        if len(seg):
            out.append(seg)
        return [self.__class__(self.p, X) for X in out]

class RS(RL):

    def __init__(self, p: 'Protein', rs: ContigType = None) -> None:
        """Selection can have duplicates, do not need to be ordered"""
        super().__init__(p, rs)
        self.data=np.unique(self.data)

    def is_full(self) -> bool:
        return len(self.data)==len(self.p)

    def __and__(self, rs_b: 'RS') -> 'RS':
        """AND operation for two residue selections, we aim to preserve the order in a"""
        rs_b=RS(self.p, rs_b)
        out=set(self.data) & set(rs_b.data)
        return RS(self.p, out)

    def __or__(self, rs_b: 'RS') -> 'RS':
        """OR operation for two residue selections, we aim to preserve the order in a, then b"""
        rs_b=RS(self.p, rs_b)
        out=set(self.data) | set(rs_b.data)
        return RS(self.p, out)

    def __iand__(self, rs_b: 'RS') -> 'RS':
        o=self.__and__(rs_b)
        self.data=o.data
        return self

    def __ior__(self, rs_b: 'RS') -> 'RS':
        o=self.__or__(rs_b)
        self.data=o.data
        return self

    @staticmethod
    def _and(*L_rs: 'RS') -> 'RS':
        """and operation on a list of RS objects"""
        if len(L_rs)==0: util.error_msg("At least one RS objective should be provided!")
        if not isinstance(L_rs[0], RS): util.error_msg("The first objective must be an RS instance!")
        p=L_rs[0].p
        L_rs=[set(RS(p, x).data) for x in L_rs]
        out=L_rs[0]
        for rs in L_rs[1:]:
            out &= set(rs)
            if len(out)==0: break
        return RS(p, out)

    @staticmethod
    def _or(*L_rs: 'RS') -> 'RS':
        """or operation on a list of RS objects"""
        if len(L_rs)==0: util.error_msg("At least one RS objective should be provided!")
        if not isinstance(L_rs[0], RS): util.error_msg("The first objective must be an RS instance!")
        p=L_rs[0].p
        L_rs=[RS(p, x).data for x in L_rs]
        out=[ x for X in L_rs for x in X ]
        return RS(p, out)

    def _not(self, rs_full: ContigType = None) -> 'RS':
        """NOT operation for a residue selection, if rs_full provided, only pick residues within those chains"""
        rs_full=RS(self.p, rs_full)
        out=[x for x in rs_full.data if x not in self.data]
        return RS(self.p, out)

    def __invert__(self) -> 'RS':
        return self._not()

    def __add__(self, rs_b: 'RS') -> 'RS':
        return self.__or__(rs_b)

    def __sub__(self, rs_b: 'RS') -> 'RS':
        """In a but not in b"""
        rs_b=RS(self.p, rs_b)
        return RS(self.p, set(self.data)-set(rs_b.data))

    def cast(self, q: 'Protein') -> 'RS':
        """Cast the selection to a selection of object q
            We convert the select to contig str, not via array indices, so that residue A10 remains residue A10
            regardless whether the A chain is in object q"""
        return q.rs(str(self))

    def str(self, format: str = "CONTIG", rs_name: str = "rs", ats: Optional['ATS'] = None, obj_name: str = "") -> str:
        return self.__str__(format, rs_name=rs_name, ats=ats, obj_name=obj_name)

    def chain(self, unique: bool = False) -> List[str]:
        chains=self.p.chain_id()
        chains=[chains[self.p.data.chain_index[x]] for x in self.data]
        if unique:
            return util.unique2(chains)
        return chains

    def __str__(self, format: str = "CONTIG", rs_name: str = "rs", ats: Optional['ATS'] = None, obj_name: str = "") -> str:
        """format: CONTIG, PYMOL
            if PYMOL, rs_name define the name of the pymol selection, ats for optional atom selection
            for PYMOL format, users can optionally specify residue name and structure object name,
                those are names for PyMOL objects.
        """
        mask=np.zeros_like(self.p.data.chain_index)
        if self.is_empty():
            if format=="CONTIG": return ""
            # create an empty selection object
            return f"select {rs_name}, none"

        mask[self.data]=1
        c_pos=self.p.chain_pos()
        out=[]
        g=self.p.data
        idx,code=self.p.split_residue_index(g.residue_index)
        for k,(b,e) in c_pos.items():
            M=mask[b:e+1]
            if np.all(M==0):
                continue
            elif np.all(M==1):
                if format=="PYMOL":
                    out.append(f"(chain {k})")
                else:
                    out.append(k)
            else:
                b0, e0 = b, e
                in_select=False
                seg=[]
                for m,i in zip(M, range(b,e+1)):
                    if in_select and not m: #break select
                        seg.append((b,i-1))
                        in_select=False
                    elif not in_select and m:
                        b=i
                        in_select=True
                if in_select:
                    seg.append((b,e))
                if format=="PYMOL":
                    # pymol has a bug, if residues are 100, 100A, 100B, 100C, 100D
                    # 100A-100D will include 100, insertion code in range is ignored
                    # so if range does not completely cover all resn_i,
                    # we will convert range to individual selection
                    X=[]
                    for b,e in seg:
                        prefix=[]
                        if b>b0 and idx[b]==idx[b-1] and b<e: # left boundary breaks code segment
                            while b<e and idx[b]==idx[b-1]:
                                prefix.append((b,b))
                                b+=1
                        suffix=[]
                        if e<e0 and idx[e]==idx[e+1] and b<e: # right boundary breaks code segment
                            while b<e and idx[e]==idx[e+1]:
                                suffix.append((e,e))
                                e-=1
                        X.extend(prefix+[(b,e)]+suffix[::-1])
                    seg=X
                # check, we cannot deal with negative idx, which was used in some PDB
                for b,e in seg:
                    if idx[b]<0 or idx[e]<0:
                        raise Exception(f"Cannot represent negative residue numbering: {g.residue_index[b]}-{g.residue_index[e]}")
                seg=[(g.residue_index[b] if b==e else f"{g.residue_index[b]}-{g.residue_index[e]}") for b,e in seg]
                if format=="PYMOL":
                    out.append(f"(chain {k} and resi "+"+".join(seg)+")")
                else:
                    out.append(k+",".join(seg))
        if format=="PYMOL":
            ats=ATS(ats)
            if ats.is_empty(): raise Exception("ats is empty in __str__")
            s=" or ".join(out)
            if ats.is_full():
                s_out=f"select {rs_name}, {s}"
            else:
                if len(out)>1:
                    s_out=f"select {rs_name}, ({s}) and name {str(ats).replace(',', '+')}"
                else:
                    s_out=f"select {rs_name}, {s} and name {str(ats).replace(',', '+')}"
            if obj_name!='':
                s_out+=f" and {obj_name}"
            return s_out
        else:
            return ":".join(out)

    def contig_mpnn(self):
        """output contig for the ProteinMPNN part of the ColabDesign, notice the contig format there
        cannot handle insertion code, so if you have insertion code, renumber it first
            q, old=p.renumber(renumber='NOCODE')

        rs: residues that we do not redesign (fixed), MPNN only redesigns the remaining residues
        """
        mask=np.zeros_like(self.p.data.chain_index)
        if not self.is_empty(): mask[self.data]=1
        idx,code=self.p.split_residue_index(self.p.data.residue_index)
        out=[]
        c_pos=self.p.chain_pos()
        #print(c_pos, mask, idx, code)
        chains=self.p.chain_id()
        for k in chains:
            (b,e) = c_pos[k]
            out2=[]
            prev_m=-1
            prev_idx=-99999
            M=mask[b:e+1]
            one=""
            for m,i in zip(M, range(b,e+1)):
                #print(">>>", i, m, idx[i], one, prev_m, prev_idx, e)
                if i==b:
                    one=(k if m>0 else "")+str(idx[i])
                elif i==e or m!=prev_m or idx[i]-prev_idx>1:
                    if prev_m>0:
                        if i==e:
                            out2.append(one+f"-{idx[i]}")
                        else:
                            out2.append(one+f"-{idx[i-1]}")
                    else:
                        if i==e:
                            L=idx[i]-int(one)+1
                        else:
                            L=idx[i-1]-int(one)+1
                        out2.append(f"{L}-{L}")
                    one=(k if m>0 else "")+str(idx[i])
                if i==e: break
                prev_m=m
                prev_idx=idx[i]
                #print(out2, one)
            out.append("/".join(out2))
        return " ".join(out)

if __name__=="__main__":
    pass
