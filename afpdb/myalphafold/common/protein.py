# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
import io
from typing import Any, Mapping, Optional
from afpdb.myalphafold.common import residue_constants
from Bio.PDB import PDBParser
import numpy as np
from string import ascii_uppercase,ascii_lowercase
from afpdb.mycolabdesign.protein import MODRES

CHAIN_IDs = ascii_uppercase+ascii_lowercase

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


#YZ: @dataclasses.dataclass(frozen=True)
@dataclasses.dataclass(frozen=False)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # 0-indexed number corresponding to the chain in the protein that this residue
  # belongs to.
  chain_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]

  def __post_init__(self):
    if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
      raise ValueError(
          f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
          'because these cannot be written to PDB format.')

  def clone(self):
    p=Protein(
      atom_positions=np.copy(self.atom_positions),
      atom_mask=np.copy(self.atom_mask),
      aatype=np.copy(self.aatype),
      residue_index=np.copy(self.residue_index),
      chain_index=np.copy(self.chain_index),
      b_factors=np.copy(self.b_factors))
    p.chain_id=np.copy(self.chain_id)
    p.warning(self.warning())
    return p

  def warning(self, data=None):
    if data is not None:
      self._warning=data.copy()
    elif not hasattr(self, '_warning'):
      self._warning={}
    return self._warning

  @staticmethod
  def from_biopython(structure, model=None, chains=None):
    """convert an BioPython structure object to Protein object"""
    # code taken from Bio/PDB/PDBIO.py
    # avoid converting to PDB to support huge strutures
    for m in structure.get_list():
        #  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]
        #  aatype: np.ndarray  # [num_res]
        #  atom_mask: np.ndarray  # [num_res, num_atom_type]
        #  residue_index: np.ndarray  # [num_res]
        #  chain_index: np.ndarray  # [num_res]
        #  b_factors: np.ndarray  # [num_res, num_atom_type]
        chain_ids=[]
        residue_index=[]
        aatype=[]
        atom_positions=[]
        atom_mask=[]
        b_factors=[]
        if model is not None and m!=model: continue
        for chain in m.get_list():
            if chains is not None and chain not in chains: continue
            chain_id=chain.id
            for residue in chain.get_unpacked_list():
                hetfield, resseq, icode = residue.id
                resname = residue.resname
                segid = residue.segid
                resid = str(residue.id[1])+residue.id[2].strip()
                residue_index.append(resid)
                chain_ids.append(chain_id)
                aatype.append(residue_constants.restype_order_with_x.get(residue_constants.restype_3to1.get(resname)))
                atom_pos=np.zeros([residue_constants.atom_type_num,3])
                atom_msk=np.zeros(residue_constants.atom_type_num)
                b_fact=np.zeros(residue_constants.atom_type_num)
                for atom in residue.get_unpacked_list():
                    x, y, z = atom.coord
                    name = atom.fullname.strip()
                    bfactor = atom.bfactor
                    idx=residue_constants.atom_order[name]
                    atom_pos[idx]=[x, y, z]
                    atom_msk[idx]=1
                    b_fact[idx]=bfactor
                atom_positions.append(atom_pos)
                atom_mask.append(atom_msk)
                b_factors.append(b_fact)
        unique_chain_ids=[]
        for x in chain_ids:
            if x not in unique_chain_ids:
                unique_chain_ids.append(x)
        chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
        chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
        p=Protein(
          atom_positions=np.array(atom_positions),
          atom_mask=np.array(atom_mask),
          aatype=np.array(aatype),
          residue_index=np.array(residue_index, dtype=np.dtype("<U6")),
          chain_index=chain_index,
          b_factors=np.array(b_factors))
        p.chain_id=unique_chain_ids
        return p
    return from_pdb_string("MODEL     1\nENDMDL\nEND")


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
  """Takes a PDB string and constructs a Protein object.

  WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

  Args:
    pdb_str: The contents of the pdb file
    chain_id: If chain_id is specified (e.g. A), then only that chain
      is parsed. Otherwise all chains are parsed.

  Returns:
    A new `Protein` parsed from the pdb contents.
  """
  pdb_fh = io.StringIO(pdb_str)
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('none', pdb_fh)
  models = list(structure.get_models())
  if len(models) != 1:
    #raise ValueError(
    #    f'Only single model PDBs are supported. Found {len(models)} models.')
    print(f"WARNING: Only single model PDBs are supported. Found {len(models)} models, use the first one!!!")
  model = models[0]

  atom_positions = []
  aatype = []
  atom_mask = []
  residue_index = []
  chain_ids = []
  b_factors = []

  ##YZ
  chain_index = []
  warning={'renamed_res':[], 'insertion_res':[], 'unknown_res':[]}
  c_chain_id={}
  L_res=[]
  ##

  for chain in model:
    if chain_id is not None and chain.id != chain_id:
      continue
    for res in chain:
      resn=chain.id+str(res.id[1])+res.id[2].strip()
      if res.id[2] != ' ':
        #raise ValueError(
        #print("WARNING:"
        #    f'PDB contains an insertion code at chain {chain.id} and residue '
        #    f'index {res.id[1]}. These are not supported.')
        warning['insertion_res'].append((resn, res.resname))
      # YZ, take care of Modified Residues
      rn=MODRES.get(res.resname, res.resname)
      if rn!=res.resname:
        warning['renamed_res'].append((resn, res.resname))
        print(f"Warning: modified residue converted: {res.resname} to {rn} at {resn}!")
      rn=residue_constants.restype_3to1.get(rn, 'X')
      if rn=='X': # skip unsupported residues
        warning['unknown_res'].append((resn, res.resname))
        print(f"Warning: unrecognized residue ignored: {res.resname} at {resn}!")
        continue
      #

      res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
      restype_idx = residue_constants.restype_order.get(
          res_shortname, residue_constants.restype_num)
      pos = np.zeros((residue_constants.atom_type_num, 3))
      mask = np.zeros((residue_constants.atom_type_num,))
      res_b_factors = np.zeros((residue_constants.atom_type_num,))
      for atom in res:
        if atom.name not in residue_constants.atom_types:
          continue
        pos[residue_constants.atom_order[atom.name]] = atom.coord
        mask[residue_constants.atom_order[atom.name]] = 1.
        res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
      if np.sum(mask) < 0.5:
        # If no known atom positions are reported for the residue then skip it.
        continue

      chain_idx=c_chain_id.get(chain.id, -1)
      if chain_idx==-1:
        c_chain_id[chain.id]=chain_idx=len(c_chain_id)
      chain_index.append(chain_idx)
      L_res.append((chain_idx, int(res.id[1]), res.id[2].strip()))

      aatype.append(restype_idx)
      atom_positions.append(pos)
      atom_mask.append(mask)
      ##YZ
      #residue_index.append(res.id[1])
      residue_index.append(str(res.id[1])+res.id[2].strip())
      ##end
      b_factors.append(res_b_factors)

  # Chain IDs are usually characters so map these to ints.
  #unique_chain_ids = np.unique(chain_ids)
  #YZ keep order
  unique_chain_ids=[x[1] for x in sorted([(v,k) for k,v in c_chain_id.items()])]
  #for x in chain_ids:
  #  if x not in unique_chain_ids:
  #    unique_chain_ids.append(x)
  #
  #chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
  #chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

  atom_positions=np.array(atom_positions)
  atom_mask=np.array(atom_mask)
  aatype=np.array(aatype)
  #YZ: we force it to use 5 chars, as residue may be renamed inplace later
  residue_index=np.array(residue_index, dtype=np.dtype("<U6"))
  chain_index=np.array(chain_index)
  b_factors=np.array(b_factors)

  #YZ: 20240902, deal with PDB where residues are not following the sorted order
  new_idx=np.array(sorted(range(len(L_res)), key=lambda x: L_res[x]))
  if len(new_idx):
    atom_positions=atom_positions[new_idx]
    atom_mask=atom_mask[new_idx]
    aatype=aatype[new_idx]
    residue_index=residue_index[new_idx]
    chain_index=chain_index[new_idx]
    b_factors=b_factors[new_idx]

  p=Protein(
      atom_positions=atom_positions,
      atom_mask=atom_mask,
      aatype=aatype,
      residue_index=residue_index,
      chain_index=chain_index,
      b_factors=b_factors)

  #YZ: inject the original chain id
  p.chain_id=unique_chain_ids
  p.warning(warning)
  if sum([len(v) for k,v in warning.items()]):
      print(f"Warning: {warning}")

  return p

def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')



def to_pdb(prot: Protein) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  restypes = residue_constants.restypes + ['X']
  res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
  atom_types = residue_constants.atom_types

  pdb_lines = []

  # ZHOU Y
  # mkdssp requires the presence of the header line
  from datetime import datetime
  today=datetime.now().strftime("%d-%b-%y")
  pdb_lines.append(f"HEADER    AFPDB PROTEIN                           {today}   XXXX")
  # END

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  ##YZresidue_index = prot.residue_index.astype(np.int32)
  residue_index = np.copy(prot.residue_index)
  ##
  chain_index = prot.chain_index.astype(np.int32)
  b_factors = prot.b_factors

  if np.any(aatype > residue_constants.restype_num):
    raise ValueError('Invalid aatypes.')

  # Construct a mapping from chain integer indices to chain ID strings.
  chain_ids = {}
  #YZ: use chain_id if available
  CHAIN_IDS=prot.chain_id if hasattr(prot, 'chain_id') else PDB_CHAIN_IDS
  ##
  for i in np.unique(chain_index):  # np.unique gives sorted output.
    if i >= PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
    #chain_ids[i] = PDB_CHAIN_IDS[i]
    #YZ: use chain_id if available
    chain_ids[i] = CHAIN_IDS[i]

  pdb_lines.append('MODEL     1')
  atom_index = 1
  last_chain_index = chain_index[0]
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multichain PDB.
    if last_chain_index != chain_index[i]:
      pdb_lines.append(_chain_end(
          atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
          residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.

    res_name_3 = res_1to3(aatype[i])
    for atom_name, pos, mask, b_factor in zip(
        atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      ##YZ, modified to support insertion code
      if (str(residue_index[i])[-1].isalpha()):
        residue_idx, insertion_code=residue_index[i][:-1], residue_index[i][-1]
      else:
        residue_idx, insertion_code=residue_index[i], ''
      ##
      occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                   ##YZ
                   #f'{residue_index[i]:>4}{insertion_code:>1}   '
                   f'{residue_idx:>4}{insertion_code:>1}   '
                   ##
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the final chain.
  pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                              chain_ids[chain_index[-1]], residue_index[-1]))
  pdb_lines.append('ENDMDL')
  pdb_lines.append('END   ')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.

def ideal_atom_mask(prot: Protein) -> np.ndarray:
  """Computes an ideal atom mask.

  `Protein.atom_mask` typically is defined according to the atoms that are
  reported in the PDB. This function computes a mask according to heavy atoms
  that should be present in the given sequence of amino acids.

  Args:
    prot: `Protein` whose fields are `numpy.ndarray` objects.

  Returns:
    An ideal atom mask.
  """
  return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True) -> Protein:
  """Assembles a protein from a prediction.

  Args:
    features: Dictionary holding model inputs.
    result: Dictionary holding model outputs.
    b_factors: (Optional) B-factors to use for the protein.
    remove_leading_feature_dimension: Whether to remove the leading dimension
      of the `features` values.

  Returns:
    A protein instance.
  """
  fold_output = result['structure_module']

  def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
    return arr[0] if remove_leading_feature_dimension else arr

  if 'asym_id' in features:
    chain_index = _maybe_remove_leading_dim(features['asym_id'])
  else:
    chain_index = np.zeros_like(_maybe_remove_leading_dim(features['aatype']))

  if b_factors is None:
    b_factors = np.zeros_like(fold_output['final_atom_mask'])

  return Protein(
      aatype=_maybe_remove_leading_dim(features['aatype']),
      atom_positions=fold_output['final_atom_positions'],
      atom_mask=fold_output['final_atom_mask'],
      residue_index=_maybe_remove_leading_dim(features['residue_index']) + 1,
      chain_index=chain_index,
      b_factors=b_factors)
