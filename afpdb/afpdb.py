#!/usr/bin/env python
from .myalphafold.common import protein as afprt
from .myalphafold.common import residue_constants as afres
from .mycolabdesign import getpdb as cldpdb
from .mycolabdesign import utils as cldutl
from .mycolabdesign import protein as cldprt
from .mycolabfold.utils import CFMMCIFIO
try:
    from .mol3D import Mol3D
    _Mol3D_=True
except:
    _Mol3D_=False
from afpdb import util
from scipy.stats import rankdata
from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict, PDBIO, Structure
import numpy as np, pandas as pd
import tempfile
import os,re,traceback,pathlib

def check_Mol3D():
    if not _Mol3D_:
        print("Requirement Py3DMol is not installed")
        exit()

def rot_a2b(a, b):
    """Return (R, T) that can transform coordinates in a into coordinates in b by a@R+T ~ b
    """
    b_c=np.reshape(np.mean(b, axis=0), (1,3))
    a_c=np.reshape(np.mean(a, axis=0), (1,3))
    R=cldprt._np_kabsch(a-a_c, b-b_c, use_jax=False)
    #print("OUT>\n", a_c, "\n", R, "\n", t)
    # b=(a-a_c)@R+b_c
    # so, t=b_c-a_c@R
    t=b_c-a_c@R
    return (R, t)

def rot_a(a, R, t):
    return a@R+t

def _test_rot_a2b():
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

# stores the standard 3D coordinates for each amino acid
RESIDUE_POSITION={}
def create_residue_position():
    global RESIDUE_POSITION
    fn=os.path.join(os.path.dirname(__file__), "myamino/amino.pickle")
    if os.path.exists(fn):
        RESIDUE_POSITION=util.load_object(fn)
        return
    for res in afres.restypes:
        res3=afres.restype_1to3[res]
        atom=afres.residue_atoms[res3]
        atomi=np.array([afres.atom_order[x] for x in atom])
        # reorder them so that N,CA,C are the first 3 atoms, as we assume this in thread_sequence()
        o=np.argsort(atomi)
        atom=[atom[i] for i in o]
        atomi=atomi[o]
        p=afprt.from_pdb_string(cldprt.pdb_to_string(f"myamino/{res}.pdb"))
        RESIDUE_POSITION[res3]=[[a, p.atom_positions[0][i]] for i,a in zip(atomi,atom) ]
    util.dump_object(RESIDUE_POSITION, fn)

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

    # Residue renumbering mode
    #    None: keep original numbering
    #    RESTART: 1, 2, 3, ... for each chain
    #    CONTINUE: 1 ... 100, 101 ... 140, 141 ... 245
    #    GAP200: 1 ... 100, 301 ... 340, 541 ... 645 # mimic AlphaFold gaps
    #    You can define your own gap by replacing GAP200 with GAP{number}, e.g., GAP10
    #    NOCODE: remove insertion code, 6A, 6B will become 6, 7

    RENUMBER={'NONE':None, 'RESTART':'RESTART', 'CONTINUE':'CONTINUE', 'GAP200':'GAP200', 'NOCODE':'NOCODE'}

    def __init__(self, pdb_str=None, contig=None):
        """pdb_str can also be a file name ends with .pdb (or .ent)/.cif, but chain_id will be ignored in that case"""
        # contains two data members
        # self.data # AlphaFold Protein instance
        self._set_data(pdb_str, contig)
        self.chain_id() # make sure we have self.data.chain_id

    def _set_data(self, data, contig=None):
        cls=type(data)
        if data is None or ((type(data) is str) and data ==''):
            data="MODEL     1\nENDMDL\nEND"
        if type(data) is pathlib.PosixPath:
            data=str(data) 
        if type(data) is str:
            if "\n" not in data:
                ext=data.lower()[-4:]
                if ext.endswith(".pdb") or ext.endswith(".ent"): # pdb file name
                    self.from_pdb(data)
                elif ext.endswith(".cif"):
                    self.from_cif(data)
                else: # we assume it's a pdb code, we fetch
                    self.fetch_pdb(data)
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

    def __getattr__(self, key):
        if key=='data_prt': return self.data
        raise AttributeError(key)

    def _make_res_map(self):
        # map residue ID (format B34A means B chain, 34A residue) into its position in the data arrays
        self.res_map={}
        c_pos=self.chain_pos()
        resi=self.data.residue_index
        #print(c_pos)
        for k,(b,e) in c_pos.items():
            for i in range(b, e+1):
                self.res_map[f'{k}{resi[i]}']=i

    def split_residue_index(self, residue_index):
        # residue index may contain insertion code, we split it into two lists, one contain the integer numbers, another contain only insertion code
        idx=[]
        code=[]
        for x in residue_index:
            m=re.search(r'(?P<idx>-?\d+)(?P<code>\D*)$', x)
            idx.append(int(m.group('idx')))
            code.append(m.group('code'))
        return np.array(idx), np.array(code)

    def merge_residue_index(self, idx, code):
        out=[ f"{a}{b}" for a,b in zip(idx, code) ]
        return np.array(out, dtype=np.dtype('<U6'))

    def b_factors(self, data=None, rs=None):
        """if data is None, return b_factor numpy array, one array for the whole structure
        otherwise, data must be a numpy array matches the length of the whole strucutre
        currently, one residue has one b-factor, we do not deal with atom-level b-factors

        # to color by b-factor in pymol

        spectrum b, deepsalmon_lightpink_paleyellow_palegreen_marine_deepblue, [selection]

        if rs is not None, it applies to the selected residues only
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

    def b_factors_by_chain(self, data=None):
        """if data is None, return b_factor as a dict, keys are chain name, values are numpy array
        otherwise, data must be a dict, where the b factors for the corresponding chains are set
        currently, one residue has one b-factor, we do not deal with atom-level b-factors
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

    def search_seq(self, seq, in_chains=None):
        """Search a sequence fragment, return all matches"""
        util.warn_msg("search_seq() is replaced by rs_seq, it returns a list of selections, not a sequence any more!")
        return self.rs_seq(seq, in_chains=in_chains)

    def chain_id(self):
        if self.data is not None and hasattr(self.data, 'chain_id'):
            return self.data.chain_id
        else:
            n=len(np.unique(self.data.chain_index))
            self.data.chain_id=list(afprt.PDB_CHAIN_IDS)[:n]
            return self.data.chain_id

    def chain_list(self):
        # list of chains in the order of their appearance in PDB
        chains=self.chain_id()
        # PDB order
        return util.unique2([chains[x] for x in self.data.chain_index ])

    def rs_missing_atoms(self, debug=False):
        """a residue selection object pointing to residues with missing atoms"""
        p=self.data
        n,m=p.atom_positions.shape[:2]
        out=[]
        chains=self.chain_id()
        for i,res in enumerate(p.aatype):
            if res >= 20: continue # residu X
            aa=afres.restype_1to3[afres.restypes[res]]
            atoms=afres.residue_atoms[aa]
            i_atoms=np.array([afres.atom_order[x] for x in atoms])
            mask=p.atom_mask[i, i_atoms]
            if not np.all(mask):
                if debug:
                    out=[ a for a,x in zip(atoms, mask) if x<1 ]
                    print(f"Missing atoms: {out}")
                out.append(i)
        return RS(self, out)

    def chain_pos(self): # return a dict of the residue position of each chain
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

    def merge_chains(self, gap=200, inplace=True):
        """
            merge_chains is meant for preparing PDB so that it complexes can be model with AF monomer
            please keep the original copy or save chain_pos, you will need chain_pos to break the
            monomer back into a multi-chain complex later

            residue numbering will be redone as the result, for AF monomer
            you may want to provide a 200 gap in numbering bewteen original chains
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

    def _residue_range(s):
        if s is None: return (None, None, None)
        chain, s=s.split(":")
        b,e = s.split("-")
        return (chain, b, e)

    def peptide_bond_length(self, residue_range=None):
        # return array of peptide bond distance for each chain, array size is (chain length-1)
        """If residue_range is provided, it must be a single region following the format:
            C:34-56A
            i.e., {chain id}:{start residue id}-{end residue id}
            We support insertion code in residue id
        """
        from numpy.linalg import norm
        c_pos=self.chain_pos()
        my_chain, my_b, my_e = Protein._residue_range(residue_range)
        #print(c_pos)
        N,C=afres.atom_order['N'],afres.atom_order['C']
        data={}
        for c,(b,e) in c_pos.items():
            if my_chain is not None and c!=my_chain: continue
            data[c]={}
            out=[]
            if my_b is not None: # user specified a range
                b2=util.index(my_b, self.data.residue_index[b:e+1])
                e2=util.index(my_e, self.data.residue_index[b:e+1])
                if b2<0:
                    util.warn_msg(f"Bad residue id: {my_b}, use 0")
                    b2=0
                if e2<0:
                    util.warn_msg(f"Bad residue id: {my_e}, use {e}")
                    e2=e-b
                b,e=b+b2,b+e2
            for pos in range(b+1,e+1):
                prev_c=self.data.atom_positions[pos-1, C]
                this_n=self.data.atom_positions[pos, N]
                out.append(norm(this_n - prev_c))
            data[c]["dist"]=np.array(out)
            idx=np.argmax(out)+1 # if max is 1st, then it's actually the 2nd residue
            resi=self.data.residue_index[b+idx]
            data[c]["max_resi"]=f"{c}:{resi}"
            data[c]["max_dist"]=out[idx-1]
            idx=np.argmin(out)+1 # if max is 1st, then it's actually the 2nd residue
            resi=self.data.residue_index[b+idx]
            data[c]["min_resi"]=f"{c}:{resi}"
            data[c]["min_dist"]=out[idx-1]
        return data

    def backbone_bond_length(self, residue_range=None):
        # return array of peptide bond distance for each chain, array size is (chain length-1)
        """Please see peptide_bond_length regarding the format of residue_range"""
        from numpy.linalg import norm
        c_pos=self.chain_pos()
        #print(residue_range)
        my_chain, my_b, my_e = Protein._residue_range(residue_range)
        #print(c_pos)
        N,C,CA=afres.atom_order['N'],afres.atom_order['C'],afres.atom_order['CA']
        data={}
        for c,(b,e) in c_pos.items():
            if my_chain is not None and c!=my_chain: continue
            data[c]={}
            out_min=[]
            out_max=[]
            if my_b is not None: # user specified a range
                b2=util.index(my_b, self.data.residue_index[b:e+1])
                e2=util.index(my_e, self.data.residue_index[b:e+1])
                if b2<0:
                    util.warn_msg(f"Bad residue id: {my_b}, use 0")
                    b2=0
                if e2<0:
                    util.warn_msg(f"Bad residue id: {my_e}, use {e}")
                    e2=e-b
                b,e=b+b2,b+e2
            for pos in range(b,e+1):
                bond_n_ca=norm(self.data.atom_positions[pos, CA]-self.data.atom_positions[pos, N])
                bond_ca_c=norm(self.data.atom_positions[pos, CA]-self.data.atom_positions[pos, C])
                if pos>b:
                    prev_c=self.data.atom_positions[pos-1, C]
                    this_n=self.data.atom_positions[pos, N]
                    bond_pept=norm(this_n - prev_c)
                    out_min.append(min(bond_n_ca, bond_ca_c, bond_pept))
                    out_max.append(max(bond_n_ca, bond_ca_c, bond_pept))
                else:
                    out_min.append(min(bond_n_ca, bond_ca_c))
                    out_max.append(max(bond_n_ca, bond_ca_c))
            data[c]["min_dist"]=np.array(out_min)
            data[c]["max_dist"]=np.array(out_max)
            #print(out_min, out_max)
            idx=np.argmax(out_max)
            resi=self.data.residue_index[b+idx]
            data[c]["max_resi"]=f"{c}:{resi}"
            data[c]["max_dist"]=out_max[idx]
            idx=np.argmin(out_min)
            resi=self.data.residue_index[b+idx]
            data[c]["min_resi"]=f"{c}:{resi}"
            data[c]["min_dist"]=out_min[idx]
        return data

    def renumber(self, renumber=None, inplace=True):
        obj = self if inplace else self.clone()
        old_num=obj._renumber(renumber)
        return (obj, old_num)

    def _renumber(self, renumber=None):
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
                    new_idx.append(new_idx[-1]+max(idx[j+1]-idx[j],1))
                idx=np.array(new_idx)
                code=['']*len(idx)
            else:
                idx=idx-idx[0]+1 # change to start from 1
            n_cnt=idx[-1]-idx[0]+1 # including missing residues
            if renumber == Protein.RENUMBER['RESTART']:
                self.data.residue_index[b:e+1]=self.merge_residue_index(idx, code)
            else: # continue or gap200
                self.data.residue_index[b:e+1]=self.merge_residue_index(idx+base, code)
            base+=n_cnt+gap # gap=0 should be equivalent to CONTINUE
        self._make_res_map()
        return old_num

    def split_chains(self, c_pos, renumber=None, inplace=True):
        """
            The key for c_pos are the desired chain name for the output structure
            The value for c_pos are the residue index in the original structure

            takes chain positions and undo merge_chains()
            if renumber, residue numbering will start from 1 for each chain
        """
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

    def reorder_chains(self, chains, renumber='RESTART', inplace=True):
        """
            chains is the list of chain names in the new order
            if renumber, residue number will start from 1 for each chain
        """
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

    def extract_by_contig(self, rs, ats=None, as_rl=False, inplace=False):
        util.warn_msg('extract_by_contig will be depreciated, please use extract() instead.')
        return self.extract(rs, ats=ats, as_rl=as_rl, inplace=inplace)

    def extract(self, rs, ats=None, as_rl=False, inplace=False):
        #for i in idx: print(chain_name[self.data.chain_index[i]]+resi[i])
        """create a new object based on residues specified by the contig string
        contig string: "A" for the whole chain, ":" separate chains, range "A2-5,10-13" (residue by residue index)
            "A:B1-10", "A-20,30-:B" (A from beginning to number 20, 30 to end, then whole B chain)
            Warning: residues are returned in the order specified by contig, we do not sort them by chain name!

            Example: p=p.extract("P-3:H3-8,10-12")
        """
        def validate_rl(rl):
            # make sure no duplicates
            t=pd.DataFrame({"resi":rl.data})
            t['chain']=rl.chain()
            if len(t.resi.unique())!=len(t):
                v,cnt=np.unique(t.resi.values, return_counts=True)
                v=v[cnt>1]
                raise Exception(f"There are duplicate residues: {v}")
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
        if len(rs)==0:
            raise Exception("No residue is selected!")
        obj=self if inplace else self.clone()
        chains=obj.chain_id()
        obj.data.aatype=obj.data.aatype[rs.data]
        obj.data.atom_positions=obj.data.atom_positions[rs.data]
        obj.data.atom_mask=obj.data.atom_mask[rs.data]
        obj.data.residue_index=obj.data.residue_index[rs.data]
        obj.data.b_factors=obj.data.b_factors[rs.data]
        obj.data.chain_index=obj.data.chain_index[rs.data]
        if ats.not_full():
            notats=(~ats).data
            obj.data.atom_mask[:, notats]=0
        # renumber chain index and chain name
        chain_index=util.unique2(obj.data.chain_index)
        c_map={x:i for i,x in enumerate(chain_index) }
        obj.data.chain_id=np.array([chains[x] for x in chain_index])
        obj.data.chain_index=np.array([c_map[x] for x in obj.data.chain_index])
        obj._make_res_map()
        return obj

    def fetch_pdb(self, code, remove_file=True):
        fn=cldpdb.get_pdb(code)
        if fn is None: return fn
        if fn.lower().endswith(".cif"):
            self.from_cif(fn)
        else:
            self.from_pdb(fn)
        if remove_file:
            os.remove(fn)
        else:
            return fn

    def __len__(self):
        return len(self.data.aatype)

    def len_dict(self):
        c,n=np.unique(self.data.chain_index, return_counts=True)
        chains=self.chain_id()
        return {chains[k]:v for k,v in zip(c,n)}

    @staticmethod
    def create_from_file(fn):
        if str(fn).lower().endswith('.cif'):
            return Protein().from_cif(fn)
        else:
            return Protein().from_pdb(fn)

    def from_pdb(self, fn, chains=None, models=None):
        self._set_data(cldprt.pdb_to_string(str(fn), chains, models))
        return self

    def to_pdb_str(self):
        data_str=afprt.to_pdb(self.data)
        return data_str

    def from_cif(self, fn, model=None, chains=None):
        parser = MMCIFParser()
        structure=parser.get_structure("", str(fn))
        return self.from_biopython(structure, model=model, chains=chains)

    def from_biopython(self, structure, model=None, chains=None):
        """convert an BioPython structure object to Protein object"""
        # code taken from Bio/PDB/PDBIO.py
        # avoid converting to PDB to support huge strutures
        self._set_data(afprt.Protein.from_biopython(structure, model=model, chains=chains))
        return self

    def to_biopython(self):
        """convert the object to BioPython structure object"""
        tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=True, suffix=".pdb").name
        self.save(tmp)
        # Load the PDB structure
        parser = PDBParser()
        structure = parser.get_structure("mypdb", tmp)
        return structure

    def html(self, show_sidechains=False, show_mainchains=False, color="chain", style="cartoon", width=320, height=320):
        check_Mol3D()
        obj=Mol3D()
        return obj.show(self.to_pdb_str(), show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color, style=style, width=width, height=height)

    def show(self, show_sidechains=False, show_mainchains=False, color="chain", style="cartoon", width=320, height=320):
        try:
            import IPython.display
            _has_IPython = True
        except ImportError:
            _has_IPython = False
        check_Mol3D()
        obj=Mol3D()
        html=obj.show(self.to_pdb_str(), show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color, style=style, width=width, height=height)
        return IPython.display.publish_display_data({'application/3dmoljs_load.v0':html, 'text/html': html},metadata={})

    def sasa(self, in_chains=None):
        """in_chains is a contig that describe what to be used as the object"""
        if in_chains is not None:
            obj=self.extract(in_chains)
        else:
            obj=self
        structure=obj.to_biopython()
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
        t=t[['chain','resn','resn_i','resi','SASA']]
        #t.display()
        return t

    def save(self, filename, format=None):
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

    def clone(self):
        return Protein(self.data.clone())

    @staticmethod
    def pdb2cif(pdb_file, cif_file):
        """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
        # code copied from colabfold/batch.py
        parser = PDBParser(QUIET=True)
        code=os.path.splitext(os.path.basename(pdb_file))[0]
        structure = parser.get_structure(code, pdb_file)
        cif_io = CFMMCIFIO()
        cif_io.set_structure(structure)
        cif_io.save(str(cif_file))

    @staticmethod
    def cif2pdb(cif_file, pdb_file):
        parser = MMCIFParser()
        structure = parser.get_structure("", cif_file)
        io=PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_file))

    @staticmethod
    def remove_hetero(pdb_file, pdb_file2):
        # Remove hetatms
        p=Protein()
        p.from_pdb(pdb_file)
        p.save(pdb_file2)

    @staticmethod
    def renumber_residue(pdb_file, pdb_file2, keep_chain_name=False):
        # loop over chains and number aa for spectrum display
        # Create a dictionary of old residue numbers to new residue numbers
        parser = PDBParser(QUIET=True)
        code=os.path.splitext(os.path.basename(pdb_file))[0]
        structure = parser.get_structure(code, pdb_file)
        res_dict = {}
        for model in structure:
            #residue.id, hetfield is a string that specifies whether the residue is a heterogen or not. resseq is an integer that specifies the residue sequence identifiericode is a string that specifies the insertion code of the residu
            # we find the max number, so that renumbering doesnot create a conflict
            for chain in model:
                res_num=max([ residue.id[1] for residue in chain ])+1
                for residue in chain:
                    residue.id = (' ', res_num, ' ')
                    res_num += 1
            for chain in model:
                res_num = 1
                for residue in chain:
                    #print(chain, residue)
                    residue.id = (' ', res_num, ' ')
                    res_num += 1
        io=PDBIO()
        io.set_structure(structure)
        io.save(pdb_file2)

    def _in_atom_list(self, atom_list, inplace=True):
        not_id=np.array([ i for i,x in enumerate(afres.atom_types) if x not in atom_list ])
        obj=self if inplace else self.clone()
        obj.data.atom_positions[:, not_id, :]=0.0
        obj.data.atom_mask[:, not_id]=0
        obj.data.b_factors[:, not_id]=0
        obj.data_str=obj.to_pdb_str()
        return obj

    def ca_only(self, inplace=True):
        return self._in_atom_list(['CA'], inplace=inplace)

    def backbone_only(self, inplace=True):
        return self._in_atom_list(['N','CA','C'], inplace=inplace)

    def reverse(self, chains=None, inplace=True):
        """chains: list of chain ids to reverse, if None, all chains,
            reverse means the chain will go from C to N term.
            residue will be renumbered as well
        """

        def flip_array(m, np_chain, chains=None, axis=0):
            m2=m.copy()
            if chains is None:
                chains=np.unique(np_chain)
            for c in chains:
                idx=np.where(np_chain == c)
                if len(idx):
                    m2[idx]=np.flip(m2[idx], axis=axis)
            m[...]=m2

        obj=self if inplace else self.clone()
        np_chain=obj.data.chain_index
        flip_array(obj.data.aatype, np_chain, chains=chains)
        flip_array(obj.data.atom_positions, np_chain, chains=chains)
        flip_array(obj.data.atom_mask, np_chain, chains=chains)
        flip_array(obj.data.b_factors, np_chain, chains=chains)
        obj.data_str=obj.to_pdb_str()
        return obj

    def fill_pos_with_ca(self, backbone_only=False, cacl_cb=False, inplace=True):
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
            obj.data.atom_positions[i,mask]=obj.data.atom_positions[i,CA] # 1 is the CA positon
            obj.data.b_factors[i,mask]=obj.data.b_factors[i,CA] # 1 is the CA positon
            # check CB
            if cacl_cb and (obj.data.atom_mask[i][CB]==0) and (np.all(obj.data.atom_mask[i][:3])):
                # N, CA, C were all present
                if obj.data.aatype[i]!=GLY:
                    pos=obj.data.atom_positions[i, :3]
                    obj.data.atom_positions[i, CB]=cldprt._np_get_cb(pos[0],pos[1],pos[2], use_jax=False)
            obj.data.atom_mask[i,mask]=1
        return obj

    def _thread_sequence(self, seq, inplace=True):
        """This does not work, b/c the peptide bond is not sured to be planary, should use pyrosetta instead"""
        create_residue_position()
        obj=self if inplace else self.clone()
        if len(self.data.aatype)!=len(seq):
            s=f"{len(self.data.aatype)} residues in the structure does not match sequence length {len(seq)}"
            raise Exception(s)

        CA=1
        for i,aa in enumerate(seq):
            res=afres.restype_1to3.get(aa, 'GLY') # if error, use GLY, i.e., no side chain
            resi=afres.restype_order[aa]
            atoms=np.array([afres.atom_order[atom] for atom,pos in RESIDUE_POSITION[res]])
            a=np.array([pos for atom,pos in RESIDUE_POSITION[res]])
            #print(a)
            #print(atoms)
            # first 3 atoms are N, CA, C
            b=self.data.atom_positions[i, :3]
            #print(">a,b\n", a, "\n", b)
            R,t=rot_a2b(a[:3], b)
            a2=rot_a(a, R, t)
            #print(">a2, b\n", a2, "\n", b)

            # copy a2 positions
            std_mask=afres.STANDARD_ATOM_MASK[resi].copy()
            not_in=np.where(std_mask==0)
            obj.data.atom_positions[i,not_in]=0.0
            obj.data.b_factors[i,not_in]=0
            obj.data.atom_mask[i,not_in]=0

            # copy b factors
            mask=(std_mask==1) & (obj.data.atom_mask[i]==0)

            obj.data.b_factors[i,mask]=obj.data.b_factors[i,CA]

            # now set non-backbone atom positoins
            idx=np.array([atoms[j] for j in range(3, a2.shape[0])])
            obj.data.atom_positions[i,idx]=a2[3:]
            obj.data.atom_mask[i,idx]=1
            obj.data.aatype[i]=resi
        return obj

    def thread_sequence(self, seq, output, relax=1, replace_X_with='', seq2bfactor=True, amber_gpu=False, cores=1, side_chain_pdb=None, chain_map=None):
        tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".pdb").name
        self.save(tmp, format="pdb")
        data=Protein.thread_sequence2(tmp, seq, output, relax=relax, replace_X_with=replace_X_with, seq2bfactor=seq2bfactor, amber_gpu=amber_gpu, cores=cores, side_chain_pdb=side_chain_pdb, chain_map=chain_map)
        os.remove(tmp)
        return data

    @staticmethod
    def thread_sequence2(pdb, seq, output, relax=1, replace_X_with='', seq2bfactor=True, amber_gpu=False, cores=1, side_chain_pdb=None, chain_map=None):
        """seq: can be a string, multiple chains are ":"-separated
                ELTQSPATLSLSPGERATLSCRASQSVGRNLGWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQARLLLPQTFGQGTKVEIKRTV:EVQLLESGPGLLKPSETLSLTCTVSGGSMINYYWSWIRQPPGERPQWLGHIIYGGTTKYNPSLESRITISRDISKNQFSLRLNSVTAADTAIYYCARVAIGVSGFLNYYYYMDVWGSGTAVTVSS:WNWFDITNK
            We strongly recommend to use dictionary or JSON string
                {"A": "ELTQSPATLSLSPGERATLSCRASQSVGRNLGWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQARLLLPQTFGQGTKVEIKRTV", "B": "EVQLLESGPGLLKPSETLSLTCTVSGGSMINYYWSWIRQPPGERPQWLGHIIYGGTTKYNPSLESRITISRDISKNQFSLRLNSVTAADTAIYYCARVAIGVSGFLNYYYYMDVWGSGTAVTVSS", "C": "WNWFDITNK"}

        """
        from afpdb.thread_seq import ThreadSeq
        ts=ThreadSeq(pdb)
        data=ts.run(output, seq, relax=relax, replace_X_with=replace_X_with, seq2bfactor=seq2bfactor, amber_gpu=amber_gpu, cores=cores, side_chain_pdb=side_chain_pdb, chain_map=chain_map)
        return data

    def center(self):
        """center of CA mass"""
        return np.mean(self.data.atom_positions[:,1], axis=0)

    def translate(self, v, inplace=True):
        """Move structure by a vector v"""
        obj=self if inplace else self.clone()
        obj.data.atom_positions[obj.data.atom_mask>0]= \
            obj.data.atom_positions[obj.data.atom_mask>0]+v
        return obj

    def center_at(self, v=None, inplace=True):
        '''position the structure, so that the new center is at v'''
        obj=self if inplace else self.clone()
        c=obj.center()
        if v is None: v=np.zeros(3)
        obj.translate(v-c)
        return obj

    def rotate(self, ax, theta, inplace=True):
        """Rotate by axis ax with theta angle, note: center is the origin
            theta should be in the unit of degree, e.g., 90
        """
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
    def rotate_a2b(a, b):
        '''Rotate so that vector a will point at b direction'''
        a=Protein._unit_vec(a)
        b=Protein._unit_vec(b)
        ang=Protein._np_ang_acos(a, b)
        ax=np.cross(a, b)
        return (ax, ang)

    def reset_pos(self):
        '''Set center at origin, align the PCA axis along, Z, X and Y'''
        self.center_at()
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
            X=self.data.atom_positions[:, 1] # CA
            pmi=get_pmi(X)
            # rotate major axis to Z, minor to X (in PyMOL, X goes out of the screen, Z at right, Y at Up)
            if i==0:
                b=np.array([1.,0.,0.])
                idx=2
            else:
                b=np.array([0.,0.,1.])
                idx=0
            ax, ang = Protein.rotate_a2b(pmi[idx], b)
            self.rotate(ax, ang*180/np.pi)
        #X=self.data.atom_positions[:, 1] # CA
        #mi=calc_m_i(X)
        #X=p.data.atom_positions[:,1]
        #print(np.min(X, axis=0), np.max(X, axis=0))
        #print(mi)

    def spin_to(self, phi, theta):
        """Assume the molecule is pointing at Z, rotate it to point at phi, theta,
            theta is the angle wrt Z, phi is the angle wrt to X,
            see https://math.stackexchange.com/questions/2247039/angles-of-a-known-3d-vector
        You should first center the molecule!
        """
        self.rotate(np.array([0.,0.,1.]), phi)
        ax=np.array([np.sin(phi), -np.cos(phi), 0])
        self.rotate(ax, theta)

    @staticmethod
    def _unit_vec(v):
        return (v/cldprt._np_norm(v)+1e-8)

    @staticmethod
    def _np_ang_acos(a, b):
        '''compute the angle between two vectors'''
        cos_ang=np.clip((Protein._unit_vec(a)*Protein._unit_vec(b)).sum(), -1, 1)
        return np.arccos(cos_ang)

    def range(self, ax, CA_only=True):
        """Return (min, max) coordinates along the ax direction"""
        if CA_only:
            X=self.data.atom_positions[:, 1] # CA
        else:
            X=self.data.atom_positions[ self.data.atom_mask>0 ]
        ax=self._unit_vec(ax)
        Y=np.sum(X*ax, axis=1)
        return (np.min(Y), np.max(Y))

    @staticmethod
    def from_atom_positions(atom_positions, atom_mask=None, aatype=None, residue_index=None, chain_index=None, b_factors=None):
        """Purely for visualization purpose, crease a strucutre from atom positions, guess atom_mask"""
        n,m=atom_positions.shape[:2]
        if atom_mask is None:
            atom_mask=np.any(atom_positions, axis=-1)
            atom_mask[:,:3]=1 # assume N, CA, C has positions
            atom_mask[:,36]=0 # remove possible OXT

        def match_res(mask, pos):
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
            print("Cloest match: (shared, input, expect)", aa, score)
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

    def seq_dict(self, keep_gap=True, gap="X"):
        chains=self.chain_list()
        seq=self.seq(keep_gap=keep_gap, gap=gap).split(":")
        return {k:v for k,v in zip(chains, seq)}

    def seq_add_chain_break(self, seq):
        """Given a multichain sequence without ':', we add ':'"""
        chains=self.data.chain_index
        idx=np.where(chains[1:]!=chains[:-1])[0]+1
        S=[]
        for i,aa in enumerate(seq):
            if i in idx:
                S.append(":")
            S.append(aa)
        return "".join(S)

    def rename_chains(self, c_chains, inplace=True):
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

    def seq(self, keep_gap=True, gap="X"):
        """Return sequence, chains are separated by :, gaps represented by character defined by "gap"
            gap: controls what char is used for gap, AlphaFold does nto take -, so we use X instead"""
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
                    for j in range(resi[i-1], resi[i]-1):
                        S.append(gap)
            S.append(afres.restypes_with_x[aa])
        return "".join(S)

    @staticmethod
    def copy_side_chain(to_pdb, src_pdb, chain_map=None, c_seq=None):
        """
            RFDiffusion output PDB: to_pdb does not contain side chain atoms
            We would like to preserve side chains for the fixed positions
            This method is to copy side chain for fixed residues based on mask provided in c_seq
            This method is used in thread_seq.py --side_chain

            chain_map: dict maps src chain name to to chain name
            c_seq is a dictionary, specifying sequence for each chain, lower case means do not copy
            If None, all residues are copied
            We only copy residue side chains when to_pdb and src_pdb has the same residue and c_seq is upper case

            The chain names in c_seq must match those in to_pdb
            The chain names in src_pdb maps to to_pdb according to chain_map
        """
        b=Protein(src_pdb)
        seq_b=b.seq_dict(gap='')
        chains_b=b.chain_list()
        e=Protein(to_pdb)
        seq_e=e.seq_dict(gap='')
        chains_e=e.chain_list()
        if c_seq is None: c_seq=seq_e
        if chain_map is None:
            chain_map={k:v for k,v in zip(chains_b, chains_e)}
        for s,t in chain_map.items():
            assert(len(seq_b[s]) == len(seq_e[t]))
            assert(len(c_seq[t]) == len(seq_e[t]))
        # create selection based on capital sequence letters
        pos_b=b.chain_pos()
        pos_e=e.chain_pos()
        sel_idx_b=[]
        sel_idx_e=[]
        for s,t in chain_map.items():
            if t not in c_seq: continue # skip the chain
            idx_b=pos_b[s][0]
            idx_e=pos_e[t][0]
            for x,y,mask in zip(list(seq_b[s]), list(seq_e[t]), list(c_seq[t])):
                #print("chains", s, t, "residues", x, y, mask)
                if mask.isupper() and x==y:
                    sel_idx_b.append(idx_b)
                    sel_idx_e.append(idx_e)
                idx_b+=1
                idx_e+=1
        if len(sel_idx_b)==0:
            print("No conserved residues, no side chain to copy...")
            return e

        sel_idx_b=np.array(sel_idx_b)
        sel_idx_e=np.array(sel_idx_e)
        b.align(e, sel_idx_b, sel_idx_e, ats="N,CA,C,O")
        print("RMSD after backbone alignment:", b.rmsd(e, sel_idx_b, sel_idx_e, ats="N,CA,C,O"))
        e.data.atom_positions[sel_idx_e,...]=b.data.atom_positions[sel_idx_b,...]
        e.data.atom_mask[sel_idx_e,...]=b.data.atom_mask[sel_idx_b,...]
        return e

    def rs(self, rs):
        return RS(self, rs)

    def rl(self, rl):
        return RL(self, rl)

    def rs_around(self, rs, dist=5, ats=None, rs_within=None, drop_duplicates=True):
        """select all residues not in rs and have at least one atom that is within dist to rs

            If drop_duplicates is True, we keep only one closest neighbor per source residue
            If drop_duplicates is False, we keep all neighbors, then all interaction pairs are kept

            return a table with 3 columns: res_id_src, res_id_target, distance, sorted by distance descend
        """
        rs=RS(self, rs)
        if rs.is_empty(): util.error_msg("rs is empty!")
        rs_b=rs._not(rs_full=rs_within)
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in rs_around()!")

        #print(rs, rs_b)
        # acceleration
        def get_xyz(rs, ats):
            """Extract atom XYZ coordinate, the corresponding residue id will also be returned"""
            xyz=self.data.atom_positions[rs.data]
            n=xyz.shape[1]
            mask=self.data.atom_mask[rs.data]
            if ats.not_full():
                xyz=xyz[:, ats.data]
                mask=mask[:, ats.data]
            n=xyz.shape[1]
            res=np.repeat(rs.data, n).reshape(-1, n)
            xyz=xyz.reshape(-1, 3)
            mask=mask.reshape(-1)>0
            res=res.reshape(-1)
            xyz=xyz[mask]
            res=res[mask]
            return (res, xyz)

        # only consider the residues within the box [-dist+xyz_min, xyz_max+dist], important for large structures
        idx_rs, xyz_rs=get_xyz(rs, ats)
        idx_b, xyz_b=get_xyz(rs_b, ats)
        xyz_min=np.min(xyz_rs, axis=0).reshape(-1,3)-dist
        xyz_max=np.max(xyz_rs, axis=0).reshape(-1,3)+dist
        mask = np.min((xyz_b>=xyz_min) & (xyz_b<=xyz_max), axis=1)
        rs_b=RS(self, np.unique(idx_b[mask]))

        t_dist=self.rs_dist(rs, rs_b, ats=ats)
        #t_dist[:10].display()
        # residues that are close and not in rs
        t_dist=t_dist[(t_dist['dist']<=dist)&(~ t_dist['resi_b'].isin(rs.data))].copy()
        if drop_duplicates:
            t_dist.drop_duplicates('resi_b', inplace=True)
        return RS(self, np.unique(t_dist['resi_b'].values)), t_dist

    def rs_missing(self):
        """Return a selection array for missing residues.
            self (p_old) contains missing residues.
            In AlphaFold, we replace X in the sequence by G, which leads to a new object p_new.

            The returned selection array points to the Gs in p_new
            rs_m=p_old.rs_miss()
            # rs_m cannot be used on p_old, as the missing residues do not exist in the backend ndarrays
            # rs_m should be used on p_new, where the missing residues are no longer missing (replaced by G).
            # You should make sure p_old and p_new has the same chain order.
            # delete all G residues in p_new
            p_no_G=p_new.extract(p_new.rs_not(rs_m))
        """
        s=self.seq().replace(":", "") # need to remove ":", otherwise index is off
        if 'X' not in s:
            return RS(self, []) # empty selection
        return RS(self, np.array(util.index_all("X", list(s))))

    def rs_mutate(self, obj):
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

    def rs_seq(self, seq, in_chains=None):
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
                out.append(RS(self, np.array(range(c_pos[k][0]+m.start(), c_pos[k][0]+m.start()+len(m.group())))))
        return out

    def rs_not(self, rs, rs_full=None):
        return RS(self, rs)._not(rs_full=rs_full)

    def rs_notin(self, rs_a, rs_b):
        return RS(self, rs_a)-RS(self, rs_b)

    def rs_and(self, *L_rs):
        L_rs=[RS(self, x) for x in L_rs]
        return RS._and(*L_rs)

    def rs_or(self, *L_rs):
        L_rs=[RS(self, x) for x in L_rs]
        return RS._or(*L_rs)

    def rs2str(self, rs, format="CONTIG"):
        return RS(self, rs).__str__(format=format)

    def ats(self, ats):
        return ATS(ats)

    def ats_not(self, ats):
        return ~ATS(ats)

    def ats2str(self, ats):
        return str(ATS(ats))

    def _atom_dist(self, rs_a, rs_b, ats=None):
        ats=ATS(ats)
        if ats.is_empty(): return np.array()
        rs_a=RS(self, rs_a)
        rs_b=RS(self, rs_b)
        if rs_a.is_empty(): util.error_msg("rs_a is emtpy!")
        if rs_b.is_empty(): util.error_msg("rs_b is emtpy!")
        a_res = self.data.atom_positions[rs_a.data]
        a_mask = self.data.atom_mask[rs_a.data].copy()
        b_res = self.data.atom_positions[rs_b.data]
        b_mask = self.data.atom_mask[rs_b.data].copy()

        if ats.not_full():
            notats=(~ats).data
            a_mask[:,notats]=0
            b_mask[:,notats]=0

        d = cldprt._np_norm(a_res[None, :, None, :, :] - b_res[:, None, :, None, :])
        mask = a_mask[None, :, None, :] * b_mask[:, None, :, None]
        d[~mask.astype(np.bool_)] = np.inf
        return d

    def _point_dist(self, center, rs, ats=None):
        rs=RS(self, rs)
        if rs.is_empty(): util.error_msg("rs is empty!") 
        ats=ATS(ats)
        if ats.is_empty(): util.error_msg("ats cannot be empty in _point_dist()!")
        a_rs = self.data.atom_positions[rs.data]
        a_mask = self.data.atom_mask[rs.data].copy()

        if ats.not_full():
            notats=(~ats).data
            mask[:,notats]=0

        d = cldprt._np_norm(a_rs[:, :, :] - center.reshape(1,1,3))
        d[~a_mask.astype(np.bool_)] = np.inf
        return d

    def atom_dist(self, rs_a, rs_b, ats=None):
        ats=ATS(ats)
        if ats.is_empty(): util.error_msg("ats cannot be empty in atom_dist()!")
        rs_a=self.rs(rs_a)
        rs_b=self.rs(rs_b)
        if rs_a.is_empty(): util.error_msg("rs_a cannot be empty in atom_dist()!")
        if rs_b.is_empty(): util.error_msg("rs_b cannot be empty in atom_dist()!")
        d=self._atom_dist(rs_a, rs_b, ats)
        df = pd.DataFrame()
        n_a=len(rs_a)
        n_b=len(rs_b)
        n_atom=afres.atom_type_num
        df['resn_a'] = np.tile(np.repeat(self.data.residue_index[rs_a.data], n_atom*n_atom), n_b)
        df['resi_a'] = np.tile(np.repeat(rs_a.data, n_atom*n_atom), n_b)
        df['resn_b'] = np.repeat(np.repeat(self.data.residue_index[rs_b.data], n_atom*n_atom), n_a)
        df['resi_b'] = np.repeat(np.repeat(rs_b.data, n_atom*n_atom), n_a)
        df['atom_a'] = np.tile(afres.atom_types, n_a*n_b*n_atom)
        df['atom_b'] = np.tile(np.repeat(afres.atom_types, n_atom), n_a*n_b)
        df['dist'] = d.flatten()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['dist'], inplace=True)
        df=df.sort_values('dist')
        rs_a2=RL(self, df.resi_a)
        rs_b2=RL(self, df.resi_b)
        df['chain_a']=rs_a2.chain()
        df['resn_i_a']=rs_a2.namei()
        df['chain_b']=rs_b2.chain()
        df['resn_i_b']=rs_b2.namei()
        df['res_a']=rs_a2.aa()
        df['res_b']=rs_b2.aa()
        df=df[['chain_a','resn_a','resn_i_a','resi_a','res_a','chain_b','resn_b','resn_i_b','resi_b','res_b','dist','atom_a','atom_b']]
        return df

    def rs_dist(self, rs_a, rs_b, ats=None):
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in _point_dist()!")
        rs_a=self.rs(rs_a)
        rs_b=self.rs(rs_b)
        if rs_a.is_empty(): util.error_msg("rs_a is empty!")
        if rs_b.is_empty(): util.error_msg("rs_b is empty!")
        d=self._atom_dist(rs_a, rs_b, ats)
        # res_list = np.argwhere(d.min(axis=(0, 2, 3)) < 3)
        # print('+'.join(p.data.residue_index[res_list].squeeze()))
        df = pd.DataFrame()
        n_a=len(rs_a)
        n_b=len(rs_b)
        df['resn_a'] = np.tile(self.data.residue_index[rs_a.data], n_b)
        df['resi_a'] = np.tile(rs_a.data, n_b)
        df['resn_b'] = np.repeat(self.data.residue_index[rs_b.data], n_a)
        df['resi_b'] = np.repeat(rs_b.data, n_a)
        n_atom=afres.atom_type_num
        d2=d.reshape(n_a*n_b, n_atom*n_atom)
        idx_pair=d2.argmin(axis=1) #figure out which atom-pair contributes to the min
        atom_a=np.tile(afres.atom_types, n_atom)
        atom_b=np.repeat(afres.atom_types, n_atom)
        df['dist'] = np.min(d, axis=(2, 3, 4)).flatten()
        df['atom_a']=atom_a[idx_pair]
        df['atom_b']=atom_b[idx_pair]
        #df.display()
        #print(np.nanmax(d, axis=(2, 3, 4)).flatten())
        #df[:50].display()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['dist'], inplace=True)
        df=df.sort_values('dist')
        rs_a2=RL(self, df.resi_a)
        rs_b2=RL(self, df.resi_b)
        df['resn_i_a']=rs_a2.namei()
        df['resn_i_b']=rs_b2.namei()
        df['chain_a']=rs_a2.chain()
        df['chain_b']=rs_b2.chain()
        df['res_a']=rs_a2.aa()
        df['res_b']=rs_b2.aa()
        df=df[['chain_a','resn_a','resn_i_a','resi_a','res_a','chain_b','resn_b','resn_i_b','resi_b','res_b','dist','atom_a','atom_b']]
        #df.display()
        return df

    def rs_dist_to_point(self, center, rs=None, ats=None):
        """Rank residues (within rs) according to their shortest distance to a point
            center: np.array(3) for XYZ
        """
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in _point_dist()!")
        rs=self.rs(rs)
        d=self._point_dist(center, rs, ats)
        df = pd.DataFrame()

        n=len(rs)
        df['resn'] = self.data.residue_index[rs]
        df['resi'] = rs
        idx=d.argmin(axis=1).reshape(-1) #figure out which atom-pair contributes to the min
        df['dist'] = np.min(d, axis=(1)).flatten()
        print(afres.atom_types, idx)
        df['atom']=np.array(afres.atom_types)[idx]
        #df.display()
        #print(np.nanmax(d, axis=(2, 3, 4)).flatten())
        #df[:50].display()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['dist'], inplace=True)
        df=df.sort_values('dist')
        rs_b=RL(self, df.resi.values)
        df['resn_i']=rs_b.namei()
        df['chain']=rs_b.chain()
        df['res']=rs_b.aa()
        df=df[['chain','resn','resn_i','resi','res','dist','atom']]
        #df.display()
        return df

    def rmsd(self, obj_b, rl_a=None, rl_b=None, ats=None):
        """ls_a and ls_b can take RS objects,
            however, RS or set should not be used, if the one-to-one mapping orders between rl_a and rl_b should be preserved.
        """
        rl_a=RL(self, rl_a)
        rl_b=RL(obj_b, rl_b)
        assert(len(rl_a)==1 or len(rl_b)==1 or len(rl_a)==len(rl_b))
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in _point_dist()!")
        #print(self.data.atom_positions.shape, rl_a, rl_b)
        a_res = self.data.atom_positions[rl_a.data]
        a_mask = self.data.atom_mask[rl_a.data].copy()
        b_res = obj_b.data.atom_positions[rl_b.data]
        b_mask = obj_b.data.atom_mask[rl_b.data].copy()
        if ats.not_full():
            notats=(~ats).data
            a_mask[:,notats]=0
            b_mask[:,notats]=0

        if a_res.shape!=b_res.shape:
            print("WARNING> selections do not share the same shape!", a_res.shape, "vs", b_res_.shape)
        d = cldprt._np_norm(a_res - b_res)
        d=d[a_mask*b_mask>0]
        #print(d)
        n=d.shape[0]
        return np.sqrt(np.sum(d*d)/n)

    @staticmethod
    def merge(objs):
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

    def align(self, target_p, rl_a=None, rl_b=None, ats=None):
        """selection rl_a, rl_b can be RS objects.
            However, RS or set should not be used, if the one-to-one mapping orders between rl_a and rl_b should be preserved.
        """
        ats=ATS(ats)
        if ats.is_empty():
            util.error_msg("ats cannot be empty in align()!")
        res_idx_a=RL(self, rl_a).data
        res_idx_b=RL(target_p, rl_b).data
        if len(res_idx_a)!=len(res_idx_b):
            raise Exception(f"Two selections have different residues: {len(res_idx_a)}, {len(res_idx_b)}.")
        if len(res_idx_a)==0:
            raise Exception(f"Empty residue selection: {len(res_idx_a)}, {len(res_idx_b)}.")
        Ma=self.data.atom_positions
        Mb=target_p.data.atom_positions
        ma=self.data.atom_mask
        mb=target_p.data.atom_mask
        if res_idx_a is not None:
            Ma=Ma[res_idx_a]
            ma=ma[res_idx_a]
        if res_idx_b is not None:
            Mb=Mb[res_idx_b]
            mb=mb[res_idx_b]
        if ats.not_full():
            Ma=Ma[:, ats.data]
            Mb=Mb[:, ats.data]
            ma=ma[:, ats.data]
            mb=mb[:, ats.data]
        a=Ma.reshape(-1,3)
        b=Mb.reshape(-1,3)
        #print(Ma.shape, Mb.shape, a.shape, b.shape, ma.shape, mb.shape)
        m=(ma.reshape(-1)>0) & (mb.reshape(-1)>0)
        a=a[m]
        b=b[m]
        if (a.shape != b.shape):
            raise Exception(f"Two selections have different atoms after masking: {a.shape}, {b.shape}.")
        R,t=rot_a2b(a,b)
        M=self.data.atom_positions
        n=M.shape
        Ma=rot_a(M.reshape(-1,3), R, t)
        Ma=Ma.reshape(*n)
        self.data.atom_positions[...]=Ma
        return (R, t)

    def dssp(self, simplify=False):
        from Bio.PDB.DSSP import dssp_dict_from_pdb_file
        tmp=tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix=".pdb").name
        self.save(tmp)
        dssp_tuple = dssp_dict_from_pdb_file(tmp)
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

    def internal_coord(self, rs=None, MaxPeptideBond=1.4):
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

def classname(obj):
    cls = type(obj)
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name

class ATS:

    def __init__(self, ats=None):
        """atoms should be comma/space-separated str, or a list of atoms"""
        if ats is None or ats=="ALL":
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

    def is_empty(self):
        return len(self.data)==0

    def is_full(self):
        return len(np.unique(self.data))==afres.atom_type_num

    def not_full(self):
        return not self.is_full()

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return ",".join([afres.atom_types[x] for x in self.data])

    def __and__(self, ats_b):
        """AND operation for two residue selections"""
        ats_b=ATS(ats_b)
        out=set(self.data) & set(ats_b.data)
        return ATS(out)

    def __or__(self, ats_b):
        """OR operation for two atom selections"""
        ats_b=ATS(ats_b)
        out=set(self.data) | set(ats_b.data)
        return ATS(out)

    def __invert__(self):
        return ATS([i for i in range(afres.atom_type_num) if i not in self.data])

    def __contains__(self, i):
        if type(i) is str: i=self.i(i)
        return i in self.data

    def __add__(self, ats_b):
        return self.__or__(ats_b)

    def __sub__(self, ats_b):
        """In a but not in b"""
        ats_b=ATS(ats_b)
        return ATS(set(self.data)-set(ats_b.data))

    @staticmethod
    def i(atom):
        return afres.atom_order.get(atom, -1)

class RL:

    def __init__(self, p, rs=None):
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
        elif type(rs)==str:
            self.data=self._rs(rs)
        else:
            util.error_msg(f"Unexpected selection type: {type(rs)}!")

        if len(self.data) and np.max(self.data)>=len(self.p):
            util.error_msg(f"Selection contains index {np.max(self.data.max)} exceeding protein length {len(self.p)}.")

    def is_empty(self):
        return len(self.data)==0

    def not_full(self):
        return not self.is_full()

    def unique(self):
        """Remove duplicates, then reorder"""
        return RS(self.p, np.unique(self.data))

    def unique2(self):
        """Remove duplicates without reordering"""
        return RS(self.p, self.data[np.sort(np.unique(self.data, return_index=True)[1])])

    def _rs(self, contig):
        """res: list of residues chain:resi, if it's a single-letter str, we select the whole chain

        contig string: "A" for the whole chain, ":" separate chains, range "A2-5,10-13" (residue by residue index)
            "A:B1-10", "A-20,30-:B" (A from beginning to number 20, 30 to end, then whole B chain)
            Warning: residues are returned in the order specified by contig, we do not sort them by chain name!

        return indicies for atom_positions
        """
        if contig is None or contig=='ALL': return np.arange(len(self.p))
        # '' should be empty, b/c the str(empty_selection) should return ''
        if type(contig) is str and contig in ('','NONE','NULL'): return np.array([])

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
            if chain!=prev_chain and chain in chain_seen:
                print(f"WARNING> contig should keep the segments for the same chain together {s}, if possible!")
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

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data

    def unique_name(self):
        """convert residue index into name {chain}{res_index}"""
        chains=self.p.chain_id()
        return [f"{chains[self.p.data.chain_index[x]]}{self.p.data.residue_index[x]}" for x in self.data]

    def name(self):
        """convert residue index into name {res_index}"""
        return [f"{self.p.data.residue_index[x]}" for x in self.data]

    def namei(self):
        """Return the integer part of the residue_index, type is int"""
        return [int(re.sub(r'\D+$', '', self.p.data.residue_index[x])) for x in self.data]

    def chain(self, rs=None):
        chains=self.p.chain_id()
        return [chains[self.p.data.chain_index[x]] for x in self.data]

    def aa(self, rs=None):
        aatype=self.p.data.aatype
        rs = self.data if rs is None else rs
        return [afres.restypes_with_x[aatype[i]] for i in self.data]

    def seq(self, rs=None):
        return "".join(self.aa())

    @staticmethod
    def i(restype):
        if len(restype)==3: restype=afres.restype_1to3.get(restype.upper(), restype)
        return afres.restype_order.get(restype.upper(), -1)

class RS(RL):

    def __init__(self, p, rs=None):
        """Selection can have duplicates, do not need to be ordered"""
        super().__init__(p, rs)
        self.data=np.unique(self.data)

    def is_full(self):
        return len(self.data)==len(self.p)

    def __and__(self, rs_b):
        """AND operation for two residue selections, we aim to preserve the order in a"""
        rs_b=RS(self.p, rs_b)
        out=set(self.data) & set(rs_b.data)
        return RS(self.p, out)

    def __or__(self, rs_b):
        """OR operation for two residue selections, we aim to preserve the order in a, then b"""
        rs_b=RS(self.p, rs_b)
        out=set(self.data) | set(rs_b.data)
        return RS(self.p, out)

    def __iand__(self, rs_b):
        o=self.__and__(rs_b)
        self.data=o.data
        return self

    def __ior__(self, rs_b):
        o=self.__or__(rs_b)
        self.data=o.data
        return self

    @staticmethod
    def _and(*L_rs):
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
    def _or(*L_rs):
        """or operation on a list of RS objects"""
        if len(L_rs)==0: util.error_msg("At least one RS objective should be provided!")
        if not isinstance(L_rs[0], RS): util.error_msg("The first objective must be an RS instance!")
        p=L_rs[0].p
        L_rs=[RS(p, x).data for x in L_rs]
        out=[ x for X in L_rs for x in X ]
        return RS(p, out)

    def _not(self, rs_full=None):
        """NOT operation for a residue selection, if rs_full provided, only pick residues within those chains"""
        rs_full=RS(self.p, rs_full)
        out=[x for x in rs_full.data if x not in self.data]
        return RS(self.p, out)

    def __invert__(self):
        return self._not()

    def __add__(self, rs_b):
        return self.__or__(rs_b)

    def __sub__(self, rs_b):
        """In a but not in b"""
        rs_b=RS(self.p, rs_b)
        return RS(self.p, set(self.data)-set(rs_b.data))

    def str(self, format="CONTIG", rs_name="rs"):
        return self.__str__(format, rs_name=rs_name)

    def __str__(self, format="CONTIG", rs_name="rs"):
        """format: CONTIG, PYMOL
            if PYMOL, rs_name define the name of the pymol selection
        """
        mask=np.zeros_like(self.p.data.chain_index)
        if self.is_empty(): return ""
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
                seg=[(g.residue_index[b] if b==e else f"{g.residue_index[b]}-{g.residue_index[e]}") for b,e in seg]
                if format=="PYMOL":
                    out.append(f"(chain {k} and resi "+"+".join(seg)+")")
                else:
                    out.append(k+",".join(seg))
        if format=="PYMOL":
            return f"select {rs_name}, "+" or ".join(out)
        else:
            return ":".join(out)


if __name__=="__main__":

    fn_ab="/da/NBC/ds/lib/example_files/5cil.pdb"
    if True:
        p=Protein(fn_ab)
        print(p.seq_dict())
        rs=RS(p, "L")
        print(rs)
        print(~rs)
        rs1=RS(p, "H1-3")
        rs2=RS(p, "H3-5:L-3")
        assert len(rs1 & rs2)==1, "rs_and"
        print(rs1 & rs2)
        assert len(rs1 | rs2)==8, "rs_or"
        print(rs1 | rs2)
        assert len(~RS(p, "H:L"))==13, "rs_not"
        print(~RS(p, "H:L"))
        assert len(RS(p, "L-100")._not(rs_in_chains="L"))==11, "rs_notin"
        print(RS(p, "L-100")._not(rs_in_chains="L"))
        assert str(rs1 | rs2)=="L1-3:H1-5", "rs2str"


        exit()
    if True:
        p=Protein(fn_ab)
        print(p.seq())
        p.save("5cil.cif")
        p=Protein("5cil.cif")
        print(p.seq())
        print(p.seq_dict())
        exit()

    if True:
        p=Protein(fn_ab)
        print(p.seq_dict())
        t=p.sasa(in_chains=["L","P"])
        t=t[t.chain=="P"]
        t.display()
        t=p.sasa(in_chains=["P"])
        t=t[t.chain=="P"]
        t.display()
        t=p.sasa()
        t=t[t.chain=="P"]
        t.display()
        exit()

    if True:
        p=Protein(fn_ab)
        #{'L': [0, 110], 'H': [111, 236], 'P': [237, 249]}
        # rs_dist compute pairwise min distance between residues across a and b
        res_a=p.rs("P2-3")
        res_b=p.rs("H56-58")
        print("Min distance between L33 and H113: ")
        p.rs_dist(res_a, res_b).display()

        res_a=p.rs("P")
        # residues in H chain that is close to P within 4A
        res_b, t_dist=p.rs_around(res_a, in_chains=["H"], dist=4)
        t_dist.display()
        exit()

    if True:
        p=Protein(fn_ab)
        #{'L': [0, 110], 'H': [111, 236], 'P': [237, 249]}
        p2=p.split_chains({"L":[10,20], "Y":[32,40], "P":[238,240]}, inplace=False)
        print(p2.seq_dict())

        p.extract("H:L1-90:P3-10", ats="C,CA,N", inplace=True)
        p.save("t.pdb")
        exit()

    if True:
        p=Protein(os.path.dirname(__file__)+"/example_files/gap.pdb")
        print(p.dssp())
        print(p.dssp(simplify=True))
        exit()

    if True:
        p=Protein("/da/NBC/ds/p/4G3/Sep2023/out_4G3/4G3_relaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb")
        #print(p.search_seq("QVQLVQSGAEVKKPGESLKISCKGSGYSFNSYWIGWVRQMPGKGLEWMGIIFPGDSFTTYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARYGGEGGFDSWGQGTLVTVSSASTKGPSVFPLAPS"))
        #print(p.seq_dict()["A"])
        t=Protein("/da/NBC/ds/p/4G3/Sep2023/1muhb.pdb")
        #print(t.search_seq("QVQLVQSGAEVKKPGESLKISCKGSGYSFNSYWIGWVRQMPGKGLEWMGIIFPGDSFTTYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARYGGEGGFDSWGQGTLVTVSSASTKGPSVFPLAPS"))
        #print(t.seq_dict()["H"])
        #p.save("before.pdb")
        #t.save("target.pdb")
        p.align(t, "A1-132", "H1-132", ats="N,CA,C,O")
        p.save("after.pdb")
        exit()

    if False:
        Protein.copy_side_chain("tgt.pdb", "src.pdb", {"H":"C", "L":"A", "P":"B"}, c_seq={"A": "EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTV", "B": "NWFDITNWLWYIK", "C": "VQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCARtevvldpgpekkllartyWGQGTLVTVSS"})

    if True:
        p=Protein.create_from_file("0027.pdb")
        dist=p.peptide_bond_length()
        print("0027")
        for k,v in dist.items():
            # if max(v) is 1, it's the 2nd residue
            if k=='dist': continue
            print(k, v)

        # optionally, we can specify residue_range
        dist=p.peptide_bond_length(residue_range="L:11-68B")
        print("0027 peptide bond")
        for k,v in dist.items():
            print(f"Chain: {k}, Residue {v['max_resi']}, Distance {v['max_dist']}, Mean bond {np.mean(v['dist'])}+/-{np.std(v['dist'])}")

        # optionally, we can specify residue_range
        dist=p.backbone_bond_length(residue_range="G:390-421")
        print("0027 backbone")
        for k,v in dist.items():
            print(f"Chain: {k}, Residue {v['max_resi']}, Distance {v['max_dist']}")
            print(f"Chain: {k}, Residue {v['min_resi']}, Distance {v['min_dist']}")


        # let's compute the average bond length with a good chain
        dist=p.peptide_bond_length("H:1-1000")
        for k,v in dist.items():
            print(f"Chain: {k}, Residue {v['max_resi']}, Distance {v['max_dist']}, Mean bond {np.mean(v['dist'])}+/-{np.std(v['dist'])}")

        exit()

    if False:
        p.from_pdb("0027_v2.pdb")
        c=p.peptide_bond_length()
        print("0027")
        for k,v in c.items():
            # if max(v) is 1, it's the 2nd residue
            print(k, max(v)+1)
        exit()

    if False:
        p=Protein()
        p.from_pdb("3IFN_AB.pdb")
        c_ab=p.seq_dict()
        c_ab2={v:k for k,v in c_ab.items()}
        p.from_pdb("3IFN_AF.pdb")
        c_af=p.seq_dict()
        c={k:c_ab2.get(v, k) for k,v in c_af.items()}
        print(c)
        exit()

    if False:
        p=Protein()
        fn="/da/NBC/ds/zhoyyi1/abdb/NR_LH_Protein_Chothia/1R0A_1.pdb"
        p.from_pdb(fn)
        p.save("test.pdb")
        p, old=p.renumber(renumber="GAP200")
        for a,b in zip(old, p.data.residue_index):
            print(a, b)
        exit()

    if False:
        p=Protein()
        fn="/da/NBC/ds/zhoyyi1/abdb/NR_LH_Protein_Chothia/1R0A_1.pdb"
        p.renumber_residue(fn, "test.pdb")
        p.remove_hetero("test.pdb", "test.pdb")
        exit()

    if False:
        p=Protein()
        p.from_pdb("6a4k.clean.pdb")
        print(p.seq())
        q=Protein.from_atom_positions(p.data.atom_positions)
        q.save("test.pdb")
        exit()

    if False:  #added by zhoubi1

        fn_pdb_input = '/da/NBC/ds/zhoubi1/ides/data/init_guess/transferrin_input/prepare_data/tfnr.pdb'
        fn_pdb_renumbered = 'tfnr_renumbered.pdb'
        fn_pdb_output = 'tfnr_all_atoms.pdb'
        fn_fasta = '/da/NBC/ds/zhoubi1/ides/data/init_guess/transferrin_input/fasta/querys'
        import afpdb
        afpdb.Protein.renumber_residue(fn_pdb_input, fn_pdb_renumbered)
        from protein.mypdb import  MySeq
        p = Protein(util.read_string(fn_pdb_renumbered))
        seq = str(list(MySeq.get_seqs(fn_fasta))[0].seq)
        p.thread_sequence(seq, fn_pdb_output, relax=10)

        p=Protein()
        # p.from_pdb("6a4k.clean.pdb")
        p.from_pdb(fn_pdb_output)
        q=Protein.from_atom_positions(p.data.atom_positions)
        q.save("test.pdb")
        exit()

    if False: # renumber
        Protein.renumber_residue("TfnR_Tfn_Fab_Final_Struc.pdb", "test.pdb")

    if False:
        p=Protein()
        p.from_pdb("6a4k.clean.pdb")
        p.reset_pos()
        print(p.range(np.array([0.,0.,1.]), True))
        p.save("pos.pdb")
        p.spin_to(np.pi/20, 0)
        p.save("pos2.pdb")
        exit()

    if False:
        p=Protein()
        p.fetch_pdb("1crn", remove_file=False)
        p.backbone_only()
        #p.ca_only()
        p.save("my.pdb")
        #p.reverse()
        #p.save("my_rev.pdb")
        p.fill_pos_with_ca(backbone_only=True, cacl_cb=True)
        #p._thread_sequence("L"*len(p.data.aatype))
        #p.save("my_fill.pdb")
        #p.save("my_fill.cif", format="cif")
        p.thread_sequence("L"*len(p.data.aatype), "my_fill.pdb", relax=2)
        fn=p.fetch_pdb("1crn", remove_file=False)
        p.pdb2cif(fn, "1crn.cif")
