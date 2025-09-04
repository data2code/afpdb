# Change Log

## v0.3.0

### Feature

- Support antibody analysis, automatic sequence alignment, and multi-object PyMOL visualization
  Please read contents tagged by "#2025JUL" in the tutorial notebook/document.
- Logic changed in gap detection (no gap if CA-CA distance less than Protein.MIN_GAP_DIST).

### New methods

- RS: split
- Protein: get_pdb_info, seq_case, chain_label2id, Mol3D, PyMOL3D, atom_coordindate,
    backbond_bond_length, align_two, "+" operator.
- Protein: antibody-related methods
    rs_antibody, truncate_antibody, rs_Fv, scFv2mAb, abag_units, seq2antibody, seq2Fv.
- StructureMetrics:
    compute_metrics_pair, compute_iptm, compute_ipsae, compute_pdockq, 
    compute_lis, compute_pae_int, compute_plddt_int.
    
## v0.2.4

### Feature

- Fixed bug in rs_seq, when the sequence contains gaps.
- Added Protein.disulfied_pair(), which returns all S-S atom pairs

## v0.2.3

### Feature

- Added aiparser.LigandMPNNParser
- Added aiparser.ProteinMPNNParser.make_structure
- Improved thread_sequence to support rl_from and rl_to, remove chain_map
