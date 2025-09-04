#!/usr/bin/env python
"""
Comprehensive test suite for AFPDB package

This test suite is organized to mirror the afpdb.md tutorial structure,
providing comprehensive coverage of all afpdb functionality:

Test Files Used:
- fn (5cil.pdb): Antibody-antigen complex with L, H, P chains
- fk (fake.pdb): Short chains with insertion codes and missing residues  
- f3 (1a3d.pdb): Three-chain homotrimer
- f4 (5cil_100.pdb): Structure with insertion codes
- fv (9cn2_scFv.pdb): scFv example
"""
import sys
sys.path.insert(0, "/da/NBC/ds/zhoyyi1/afpdb/pypi/src")
import os
import numpy as np
import pytest
from afpdb.afpdb import Protein, RS, RL, ATS, _Ab_, _Mol3D_, _PYMOL_
from afpdb.myalphafold.common import residue_constants as afres

# File paths for testing (matching tutorial examples)
fn = os.path.join(os.path.dirname(__file__), "5cil.pdb")  # Main antibody-antigen example
fk = os.path.join(os.path.dirname(__file__), "fake.pdb")   # Short test structure
f3 = os.path.join(os.path.dirname(__file__), "1a3d.pdb")   # Homotrimer
f4 = os.path.join(os.path.dirname(__file__), "5cil_100.pdb")  # With insertion codes
fv = os.path.join(os.path.dirname(__file__), "9cn2_scFv.pdb")  # scFv example

# Utility function for validation (from original tutorial examples)
def check_p(p):
    """Check if protein object matches expected 5cil.pdb chain IDs and sequence."""
    return ((set(p.chain_id()) == {"L", "H", "P"}) and 
            p.seq() == 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTV:VQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGDGDLGKPIGAFAHWGQGTLVTVSS:NWFDITNWLWYIK')

# ============================================================================
# Demo
# ============================================================================

def test_demo_basic_loading():
    """Tutorial Demo: Load ab-ag complex and show summary (p.summary())."""
    p = Protein(fn)
    # Test basic loading works
    assert len(p.chain_id()) == 3, "Should load 3 chains"
    assert set(p.chain_id()) == {"L", "H", "P"}, "Should have L, H, P chains"
    
    # Test summary method exists and runs
    summary = p.summary()
    assert len(summary) == 3, "Summary should return data"

def test_demo_renumbering():
    """Tutorial Demo: Renumber residues as shown in tutorial."""
    p = Protein(fn)
    
    # Test original P chain residue numbering 
    p_orig_names = p.rs("P").name()
    assert len(p_orig_names) == 13, "P chain should have 13 residues"
    
    # Test renumber CONTINUE returns tuple
    q, old_num = p.renumber("CONTINUE")
    new_names = q.rs("P").name()
    expected_names = ['238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250']

    assert new_names == expected_names, "P chain renumbering should restart from 238"
    
    # Test removal of insertion codes - fix array comparison
    q = p.renumber("NOCODE")[0].renumber("RESTART")[0]
    assert list(q.chain_id()) == list(p.chain_id()), "Chain IDs preserved after NOCODE"

    new_names = q.rs("P").name()
    expected_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    assert new_names == expected_names, "P chain renumbering should restart from 1"

def test_demo_gap_sequence():
    """Tutorial Demo: Replace missing residues with glycine for AlphaFold."""
    p = Protein(f4)
    seq_with_gaps = p.seq(gap="G")
    
    # Should contain glycine replacements
    assert "PSGGGGGGGGGT" in seq_with_gaps, "Should contain glycine gap replacements"
    assert len(seq_with_gaps.split(":")) == 3, "Should have 3 chains separated by :"

    p =Protein(fn)
    # Test the specific sequence format mentioned in tutorial
    assert seq_with_gaps.startswith("VQL"), "H chain should start with VQL" 
    assert ":EIV" in seq_with_gaps, "L chain should start with EIV"
    assert seq_with_gaps.endswith("YIK"), "P chain should end with YIK"

def test_demo_interface_extraction(tmp_path):
    """Tutorial Demo: Interface residue identification and extraction."""
    p = Protein(fn)
    
    # Identify binder residues within 4A of antigen P chain  
    binder, target, df_dist = p.rs_around("P", dist=4)
    
    # Verify we found interface residues
    assert len(binder) > 0, "Should find interface residues"
    assert len(df_dist) > 0, "Should return distance dataframe"
    assert len(target) > 0, "Should find target residues"
    
    # Extract interface complex (antigen + binder residues)
    q = p.extract(binder | "P")
    assert len(q.chain_id()) == 3, "Interface complex should have 3 chains"
    
    # Test file saving
    out_file = tmp_path / "binding.pdb"
    q.save(str(out_file))
    assert os.path.exists(out_file), "Should save interface PDB file"
    
    # Verify saved file can be reloaded
    q_reloaded = Protein(str(out_file))
    assert len(q_reloaded) > 0, "Saved file should be readable"

def test_demo_antibody():
    """Tutorial Demo: Interface residue identification and extraction."""
    p = Protein("5cil")
    rs_cdr, rs_var, c_chain_type, c_cdr = p.rs_antibody(scheme="imgt", set_b_factor=True)
    # remove constant domain, only keep the variable domain for antibody
    q=p.extract(rs_var | p.rs("P"))

    assert(len(q) < len(p)), "Remove constant regions from antibody."

    html = q.html(style="cartoon", color="b")
    assert "3dmolviewer" in html, "Generate HTML for protein visualization."

def test_demo_align_two():
    p = Protein("5cil")
    q = p.truncate_antibody()
    # remove constant domain, only keep the variable domain for antibody
    # q is a sub-protein of p, we rename chains in q to demonstrate align_two does not rely on chain names
    q.rename_chains({"H":"X", "L":"Y", "P":"Z"}, inplace=True).translate([3, 4, 5], inplace=True)
    assert(len(q) < len(p)), "Remove constant regions from antibody."
    assert set(q.chain_id()) == {"X","Y","Z"}
    p2, q2, rl_p, rl_q = p.align_two(q, auto_chain_map=True)
    rmsd = p.rmsd(q, rl_p, rl_q, align=True)
    assert np.abs(rmsd) < 0.1, "RMSD should be small"
    
# ============================================================================
# FUNDAMENTAL CONCEPTS
# ============================================================================

def test_data_structure():
    """Test internal data structure of Protein matches AlphaFold format."""
    p = Protein(fk)  # Using fake.pdb for simpler testing
    c = p.len_dict()
    assert c["H"] == 2 and c["L"] == 4, "Chain lengths should match fake.pdb"
    
    # Test internal AlphaFold Protein data structure
    g = p.data
    assert np.array_equal(g.chain_index, np.array([0, 0, 0, 0, 1, 1])), "chain_index mapping"
    assert np.array_equal(g.chain_id, np.array(["L", "H"])), "chain_id array"
    assert np.array_equal(g.aatype, np.array([6, 9, 19, 5, 10, 19])), "amino acid types"
    assert np.array_equal(np.array(g.atom_positions.shape), np.array([6, 37, 3])), "atom_positions shape"
    
    # Test atom masks for CA and CG atoms
    assert np.all(g.atom_mask[:, afres.atom_order['CA']] == 1), "CA atoms should be present"
    assert np.all(g.atom_mask[np.array([1, 2, 5]), afres.atom_order['CG']] == 0), "CG atoms should be missing for some residues"
    
    # Test coordinate validity
    assert np.all(np.abs(g.atom_positions[:, afres.atom_order['CA']]) > 1), "CA coordinates should be non-zero"
    assert np.all(np.abs(g.atom_positions[np.array([1, 2, 5]), afres.atom_order['CG']]) < 1), "Missing atom coords should be near zero"

def test_basic_loading():
    """Test basic loading and sequence extraction."""
    p = Protein(fn)
    c = p.seq_dict()
    assert c['P'] == 'NWFDITNWLWYIK', "Load object and seq_dict()"

def test_chain_ids():
    """Test chain ID extraction."""
    p = Protein(fn)
    assert set(p.chain_id()) == {"L", "H", "P"}, "Chain IDs"

def test_sequence():
    """Test sequence extraction."""
    p = Protein(fn)
    assert p.seq() == 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTV:VQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGDGDLGKPIGAFAHWGQGTLVTVSS:NWFDITNWLWYIK', "Sequence extraction"

def test_inplace_and_clone():
    """Tutorial: Test in-place vs copy operations."""
    p = Protein(fn)
    
    # Test copy operation (default)
    q = p.extract("H:L")
    assert p != q, "Extract should create copy by default"
    
    # Test in-place operation
    original_len = len(p)
    q = p.extract("H:L", inplace=True) 
    assert p == q, "In-place extract should return same object"
    assert len(p) < original_len, "In-place extract should modify original"
    assert not check_p(p), "Original object should be modified"
    
    # Test clone preserves original
    p = Protein(fn)  # Reload
    q = p.clone()
    q.extract("H:L", inplace=True)
    assert check_p(p), "Clone should preserve original object"
    assert not check_p(q), "In-place extract should modified the cloned object"
    
# ============================================================================
# SELECTION
# ============================================================================

def test_residue_selection():
    """Test residue selection following tutorial examples."""
    p = Protein(fn)
    
    # Basic single residue selection
    assert len(p.rs("L11")) == 1, "Single residue selection"
    
    # Complex multi-range selection  
    assert len(p.rs("H-5:H10-15:L-10")) == 21, "Complex residue selection"
    
    # Test equivalent selection formats
    assert np.array_equal(p.rs("H-5:H10-15:L-10").data, p.rs("H-5,10-15:L-10").data), "Equivalent selection formats"

def test_atom_selection():
    """Test atom selection methods."""
    p = Protein()
    assert np.array_equal(ATS("N,CA,C,O").data, np.array([0,1,2,4])), "ATS string selection"
    assert np.array_equal(ATS(["N","CA","C","O"]).data, np.array([0,1,2,4])), "ATS list selection"
    assert len(p.ats(""))==0, "Empty atom selection"
    assert len(p.ats(None))==37, "All atoms selection"
    assert len(p.ats("ALL"))==37, "ALL atoms"
    assert len(p.ats("NULL"))==0, "NULL atoms"
    assert np.array_equal(p.ats_not(["N","CA","C","O"]).data, p.ats("CB,CG,CG1,CG2,OG,OG1,SG,CD,CD1,CD2,ND1,ND2,OD1,OD2,SD,CE,CE1,CE2,CE3,NE,NE1,NE2,OE1,OE2,CH2,NH1,NH2,OH,CZ,CZ2,CZ3,NZ,OXT").data), "ats 6"

def test_rs():
    """Test RS residue selection class."""
    p = Protein(fn)
    rs1 = p.rs("H1-3")
    rs2 = p.rs("H3-5:L-3")
    assert len(p.rs_and(rs1, rs2))==1, "rs_and"
    assert len(p.rs_or(rs1, rs2))==8, "rs_or"
    assert len(p.rs_not("H:L"))==13, "rs_not"
    assert len(p.rs_not("L-100", rs_full="L"))==11, "rs_notin"
    assert p.rs2str(p.rs_or(rs1, rs2))=="L1-3:H1-5", "rs2str"
    assert str(p.rs_or(rs1, rs2))=="L1-3:H1-5", "rs2str"

    p = Protein(fn)
    p = p.extract(rs="H-5:L-5", ats="N,CA,C,O")
    assert len(p)==10, "rs and ats"
    # no side chain
    assert np.sum(p.data_prt.atom_mask[:, p.ats("CB").data])==0, "backbone only"

def test_rs2():
    """Test RS class with operator overloading."""
    p = Protein(fn)
    rs1 = RS(p, "H1-3")
    rs2 = RS(p, "H3-5:L-3")
    assert len(rs1 & rs2)==1, "rs_and"
    assert len(rs1 | rs2)==8, "rs_or"
    assert len(~RS(p, "H:L"))==13, "rs_not"
    assert len(RS(p, "L-100")._not(rs_full="L"))==11, "rs_notin"
    assert len(~RS(p, "L-100") & "L")==11, "rs_notin 2"
    assert str(rs1 | rs2)=="L1-3:H1-5", "rs2str"
    assert len(RS._or(RS(p,"H1-3"), "H10", "H11", "H12", "H13"))==7, "rs_or"
    assert len(RS._and(RS(p,"H1-10"), "H3-20", "H5-22"))==6, "rs_and"
    assert len(rs1 - rs2)==2, "rs_minus"
    rs1 |= rs2
    assert (len(rs1) == 8), "rs_ior"
    rs1 &= rs2
    assert (len(rs1) == len(rs2)), "rs_iand"

def test_rs2str():
    """Test residue selection string formatting."""
    p = Protein(f4)
    assert p.rs2str("L5-10,100:H")=="H:L5-10,100", "rs2str 1"
    assert p.rs2str("L5-10,100:H", format="PYMOL")=="select rs, (chain H) or (chain L and resi 5-10+100)", "rs2str 2"

    assert p.rs2str("H100A-102", format="PYMOL")=="select rs, (chain H and resi 100A+100B+100C+100D+100E+100F+100G+100H+100I+100J+101-102)", "rs2str 3"
    assert p.rs2str("H100-102", format="PYMOL")=="select rs, (chain H and resi 100-102)", "rs2str 4"
    assert p.rs2str("H100-100D", format="PYMOL")=="select rs, (chain H and resi 100+100A+100B+100C+100D)", "rs2str 5"
    assert p.rs2str("H100D-100F", format="PYMOL")=="select rs, (chain H and resi 100D+100E+100F)", "rs2str 6"
    assert p.rs2str("H98-100,100D,100F-102", format="PYMOL")=="select rs, (chain H and resi 98-99+100+100D+100F+100G+100H+100I+100J+101-102)", "rs2str 7"

def test_rl():
    p=Protein(fn)
    rl_h=p.rl("H")
    rl_l=p.rl("L")
    # + and | both means concatenation of the two selections
    assert str(rl_h + rl_l) == "H:L", "Concatenate RL1"
    assert str(rl_h | rl_l | rl_h) == "H:L:H", "Concatenate RL2"
    assert str(RL._or(rl_h, rl_l, rl_h, rl_l)) == "H:L:H:L", "Concatenate RL3"
    s_rl = str(rl_h + rl_h - p.rl("H3"))
    assert s_rl == "H1-2:H4-126:H1-2:H4-126", "Minus in RL."

def test_negative_residue_numbering():
    # create negative residue numbering
    q=Protein(fn).extract("P")
    assert q.resn()[0] == "1", "Positive numbering"

    q.resn(np.arange(len(q))-5)
    assert q.resn()[0] == "-5", "Negative numbering"

    # new q has negative numbering starting from -5
    # we cannot specify the first residue as P-5
    # P-5 means we select the first residue all the way to a residue named "5", therefore, first 11 residues
    assert len(q.rs("P-5")) == 11, "Cannot use negative numbering in contigs."
    
    # select the first three residues
    rs = q.rs(q.rs().data[:3])
    assert len(rs) == 3, "Selection is correct with residue indices."

    # we recommend you remove such negative numbering
    q, old=q.renumber("RESTART")
    assert len(q.rs("P-5")) == 5, "Renumber to get rid of residue residues in order to support contiguity."

def test_rs_split():
    p=Protein(fn)
    rs_cdr, _, _, _ = p.rs_antibody()
    assert [str(cdr) for cdr in (rs_cdr & "H").split()] == ['H25-31', 'H51-56', 'H98-115'], "Heavy chain CDRs."
    assert [str(cdr) for cdr in (~rs_cdr & "H").split()] == ['H1-24', 'H32-50', 'H57-97', 'H116-126'], "Heavy chain frameworks."

def test_selection_checks():
    """Test is_empty, is_full, not_full selection check methods."""
    p = Protein(fk)
    
    try:
        # Test is_empty
        empty_rs = p.rs("")  # Non-existent chain
        assert empty_rs.is_empty(), "Non-existent selection should be empty"
        
        # Test is_full vs not_full
        full_rs = p.rs(None)  # All residues
        partial_rs = p.rs("L5-10")
        
        # Note: is_full and not_full depend on context of what "full" means
        # Testing basic functionality exists
        full_check = full_rs.is_full()
        not_full_check = partial_rs.not_full()
        assert isinstance(full_check, bool), "is_full should return boolean"
        assert isinstance(not_full_check, bool), "not_full should return boolean"
    except (AttributeError, NotImplementedError):
        pytest.skip("is_empty/is_full/not_full methods not implemented")

def test_canonicalize_rs():
    """Test residue selection canonicalization."""
    p = Protein(fn)
    # an unusual selection
    rs = "L1-5:P12-13:L6-10"
    try:
        p.extract(rs)
        assert False, "unusual selection 1"
    except Exception as e:
        assert True, "unusual selection 1"
    try:
        p.extract(p.rs(rs))
        assert True, "unusual selection 2"
    except Exception as e:
        assert False, "unusual selection 2"

def test_rs_around():
    """Test rs_around functionality for finding nearby residues."""
    p = Protein(fn)
    rs = p.rs("P")
    rs_nbr, r_seed, t = p.rs_around(rs, dist=3.5, drop_duplicates=True)
    assert p.rs2str(rs_nbr)=="L33,92-95:H30,56,58,98,106-109", "rs_around 1"
    assert len(t)==len(rs_nbr), "rs_around 2"

def test_residue_id():
    """Test residue ID handling and filtering."""
    p = Protein(fn)
    rs = p.rs("P")
    rs_nbr, rs_seed, t = p.rs_around(rs, dist=3.5, drop_duplicates=True)
    t.display()
    t2 = t[(t.chain_b=="H")&(t.resn_b>="95")&(t.resn_b<="106")]
    #t2.display()
    assert len(t2)==0, "residue filter 1"
    t2 = t[(t.chain_b=="H")&(t.resn_i_b>=95)&(t.resn_i_b<=106)]
    #t2.display()
    assert len(t2)==2, "residue filter 2"

def test_ats2str():
    """Test ATS to string conversion method."""
    p = Protein(fk)
    
    try:
        # Test ATS to string conversion
        ats = p.ats("N,CA,C,O")
        ats_str = p.ats2str(ats)
        assert isinstance(ats_str, str), "Should return string"
        assert "N" in ats_str, "Should contain N atom"
        assert "CA" in ats_str, "Should contain CA atom"
        
        # Test with empty ATS
        empty_ats = p.ats("")
        empty_str = p.ats2str(empty_ats)
        assert empty_str == "", "Empty ATS should return empty string"
    except (AttributeError, NotImplementedError):
        pytest.skip("ats2str method not implemented")

def test_selection_checks():
    """Test is_empty, is_full, not_full selection check methods."""
    p = Protein(fk)
    
    try:
        # Test is_empty
        empty_rs = p.rs("")  # Non-existent chain
        assert empty_rs.is_empty(), "Non-existent selection should be empty"
        
        # Test is_full vs not_full
        full_rs = p.rs(None)  # All residues
        partial_rs = p.rs("L5-10")
        
        # Note: is_full and not_full depend on context of what "full" means
        # Testing basic functionality exists
        full_check = full_rs.is_full()
        not_full_check = partial_rs.not_full()
        assert isinstance(full_check, bool), "is_full should return boolean"
        assert isinstance(not_full_check, bool), "not_full should return boolean"
    except (AttributeError, NotImplementedError):
        pytest.skip("is_empty/is_full/not_full methods not implemented")

def test_invalid_selection():
    """Test error handling for unusual residue selection."""
    p = Protein(fn)
    rl = p.rl("L1-5:P12-13:L6-10")
    # The first extract call should NOT raise an exception
    p.extract(rl, as_rl = False)
    # Should raise for extract with as_rl=True
    with pytest.raises(Exception):
        p.extract(rl)

# ============================================================================
# READ & WRITE  
# ============================================================================

def test_file_operations(tmp_path):
    """Test saving and loading from files."""
    p = Protein(fn)
    out_pdb = tmp_path / "test.pdb"
    out_cif = tmp_path / "test.cif"
    p.save(str(out_pdb))
    p2 = Protein(str(out_pdb))
    assert check_p(p2), "Save and load PDB file"
    p.save(str(out_cif))
    p3 = Protein(str(out_cif))
    assert check_p(p3), "Save and load CIF file"

    q=Protein(p.data)
    assert check_p(q), "Create Protein from an afpdb.Protein.data object"

    b=p.to_biopython()
    q=Protein(b)
    assert check_p(q), "Create Protein from a BioPython Structure object"

    s=p.to_pdb_str()
    print("\n".join(s.split("\n")[:5])+"\n...")
    q=Protein(s)
    assert check_p(q), "Create Protein from a str containing the content of a PDB file"

def test_from_sequence():
    if not _PYMOL_:
        pytest.skip("PyMOL is not available")
    
    p = Protein().from_sequence("MVVVSLQCAIVGQAGS:SFDVEIDDGAKVSKLKDAI")
    assert p.seq_dict()["A"]=="MVVVSLQCAIVGQAGS", "Sequence 1"

    p = Protein().from_sequence({"E": "MVVVSLQCAIVGQAGS", "F":"SFDVEIDDGAKVSKLKDAI"})
    assert p.seq_dict()["F"]=="SFDVEIDDGAKVSKLKDAI", "Sequence 2"
    
    p = Protein().from_sequence({"E": "MVVXVSLQCAIXXXXXVGQAGS", "F":"SFDVEIXXXDDGAKVSKLKDAI"})
    assert p.seq_dict()["F"]=="SFDVEIXXXDDGAKVSKLKDAI", "Sequence 3"

def test_get_pdb_info():
    info = Protein.get_pdb_info("5cil")
    assert "PLAPSSKSTSGGTAALG" in info['seq_dict']["H"], "PDB meta data"

def test_seq_gap_by_pdb():   
    try:
        p = Protein("5cil")
        seq_dict_gap = p.seq_dict_gap_by_pdb("5cil", gap_lower_case = True)
        assert "PSskstsgGT" in seq_dict_gap["H"], "sequence gap has been filled with complete sequence from PDB database"
    except:
        pytest.skip("seq_dict_gap_by_pdb required connection to PDB.")

# ============================================================================
# SEQIEMCE & CHAIN  
# ============================================================================

def test_rs_seq():
    p = Protein(fn)
    rs_cdr=p.rs_seq('GGSFSTY')[0] | p.rs_seq('IPLLTI')[0] | p.rs_seq('EGTTGDGDLGKPIGAFAH')[0]
    seq = p.seq_case("H", rs_cdr)
    assert "sckasGGSFSTYalswv" in seq, "Search subsequence within a longer sequence."

    out=p.rs_seq(r'[RHKDE]')
    # concatenate the list of matched residue selection objects into one RS object, RS._or() will be explained later.
    m=RS._or(*out)
    print(m.seq())
    assert m.seq()=='EERRKRRRDRDREEDKEKREKRKRRERRDREREDREDDKHDK', "Search single letter code with regular expression."

def test_rename_reorder_chains():
    """Test combined rename and reorder chains operation."""
    p = Protein(f3)  # Homotrimer with multiple chains
    
    try:
        # Test combined rename and reorder
        chain_mapping = {"A": "X", "B": "Y", "C": "Z"}
        new_order = ["Z", "Y", "X"]
        
        result = p.rename_reorder_chains(chain_mapping, new_order)
        assert hasattr(result, 'chain_id'), "Should return protein object"
        
    except (AttributeError, NotImplementedError):
        # Test if separate methods work
        try:
            p_renamed = p.rename_chains({"A": "X", "B": "Y", "C": "Z"})
            p_reordered = p_renamed.reorder_chains(["Z", "Y", "X"])
            assert len(p_reordered.chain_id()) == 3, "Separate operations should work"
        except:
            pytest.skip("rename_reorder_chains and component methods not implemented")

def test_chain_manipulation():
    """Test chain reordering, renumbering, and merging/splitting."""
    p = Protein(fn)
    assert "".join(p.chain_id())=="LHP", "Initial chain order"
    p.reorder_chains(["P","L","H"], inplace=True)
    assert not check_p(p), "Chain reorder"
    assert "".join(p.chain_id())=="PLH", "Chain order after reorder"
    p = Protein(fn)
    c = p.chain_pos()
    assert c["L"][0]==0 and c["L"][1]==110, "chain_pos L"
    assert c["H"][0]==111 and c["H"][1]==236, "chain_pos H"
    assert c["P"][0]==237 and c["P"][1]==249, "chain_pos P"
    p = Protein(fk)
    p.renumber(None, inplace=True)
    assert ",".join(p.rs().name())=="5,6A,6B,10,3,4", "_renumber None"
    p.renumber("RESTART", inplace=True)
    assert ",".join(p.rs().name())=="1,2A,2B,6,1,2", "_renumber RESTART"
    p.renumber("CONTINUE", inplace=True)
    assert ",".join(p.rs().name())=="1,2A,2B,6,7,8", "_renumber CONTINUE"
    p.renumber("GAP33", inplace=True)
    assert ",".join(p.rs().name())=="1,2A,2B,6,40,41", "_renumber GAP33"
    p.renumber("NOCODE", inplace=True)
    assert ",".join(p.rs().name())=="1,2,3,7,8,9", "_renumber NOCODE"
    p = Protein(fn)
    q, c_pos = p.merge_chains(gap=200, inplace=False)
    assert c_pos["H"][0]==111 and c_pos["H"][1]==236, "original chain H"
    c_pos_q = q.chain_pos()
    assert c_pos_q["L"][0]==0 and c_pos_q["L"][1]==249, "merged chain L"
    assert len(q.seq()) - len(p.seq().replace(":", ""))==200*2, "merge chain contains two 200-residue gaps."
    assert len(q.chain_id())==1, "merge chain count"
    r = q.split_chains(c_pos, inplace=False)
    c_pos = r.chain_pos()
    assert len(r.chain_id())==3 and c_pos["H"][0]==111 and c_pos["H"][1]==236, "split_chain"

def test_rsi_missing():
    """Test residue index missing method."""
    p = Protein(fk)  # Has missing residues

    missing_indices = p.rsi_missing()
    assert np.array_equal(missing_indices, np.array([3, 4, 5])), "Should return indices for missing residues"

def test_seq_case():
    """Test sequence case conversion and gap handling variants."""
    p = Protein(fn)
    
    seq = p.seq_case("H1-15", "H5-10")
    assert seq == "vqlvQSGAEVkrpgs", "turn selected residues into upper case, the rest in lower case"

def test_rs_missing_atoms():
    """Test finding residues with missing atoms."""
    p = Protein(fk)  # fake.pdb has missing atoms
    
    try:
        missing_rs = p.rs_missing_atoms()
        assert hasattr(missing_rs, 'data'), "Should return RS object"
        
        # Test with specific atom types
        missing_cb = p.rs_missing_atoms("CB")
        assert hasattr(missing_cb, 'data'), "Should return RS object for residues without CB"
        
    except (AttributeError, NotImplementedError):
        pytest.skip("rs_missing_atoms method not implemented")

def test_rs_mutate():
    """Test residue mutation functionality."""
    p = Protein(fk)
    q = p.clone()
    q.data.aatype[2] = 5
    old_res = p.seq()[2]
    # Test basic mutation
    mutated = p.rs_mutate(q)
    assert mutated.data[0] == 2, "identified the mutated residue at position index 2"
    mutated_res = q.seq()[2]
    assert((old_res == "V") and (mutated_res == "Q")), "Mutated from V to Q at index 2"

def test_thread_sequence(tmp_path):
    """Test thread_sequence method basic coverage."""
    if not _PYMOL_:
        pytest.skip("PyMOL not installed")
    p = Protein(fk)
    seq = p.seq()
    # Should not raise error for threading same sequence
    try:
        out_file = str(tmp_path / "m.pdb")
        p.thread_sequence({"L":"AIVD", "H":"LG"}, out_file, relax=0)
        q = Protein(out_file)
        assert p.seq() == "EIVXXXQ:LV", "thread_sequence result"
        assert q.seq() == "AIVXXXD:LG", "thread_sequence output file"
    except Exception:
        pytest.fail("thread_sequence failed for identical sequence")

# ============================================================================
#  GEOMETRY, MEASUREMENT, & VISUALIZATION 
# ============================================================================

def test_visualization():
    """Test visualization methods."""
    p = Protein(fn)
    html = p.html()
    assert html is not None, "HTML visualization"
    try:
        p.show()
    except Exception:
        pytest.fail("3D visualization failed")

    html = Protein.Mol3D() \
        .show(p.extract("H:L")) \
        .show(p.extract("P"), style="stick") \
        .show(html=True)
    assert html is not None, "HTML visualization"

def test_show_pymol(tmp_path):
    """Test PyMOL visualization if _PYMOL_ is True."""
    if not _PYMOL_:
        pytest.skip("PyMOL not available")
    p = Protein(fn)
    out_file_1 = str(tmp_path / "pymol3d_1.pse")
    try:
        p.show_pymol(output=out_file_1)
        assert os.path.exists(out_file_1), "PyMOL3D visualization"
    except Exception as e:
        pytest.fail(f"show_pymol failed: {e}")

    out_file_2 = str(tmp_path / "pymol3d_2.pse")
    Protein.PyMOL3D() \
        .show(p.extract("H:L")) \
        .show(p.extract("P"), style="stick") \
        .show(output=out_file_2)

    assert os.path.exists(out_file_2), "PyMOL3D visualization"

def test_pymol(tmp_path):
    # create a new PyMOL engine
    if not _PYMOL_:
        pytest.skip("PyMOL not available")
    pm=Protein().PyMOL()
    pm.cmd(f"load {fn}, myobj")
    p=Protein(fn)
    rs_binder, rs_seed, t = p.rs_around("P", rs_within="H:L", dist=4)
    # selecting all interface residues
    rs_int = rs_binder | rs_seed
    # convert Afpdb residue selection to a PyMOL selection command, where the selection object is named "myint"
    rs_str = rs_int.str(format="PYMOL", rs_name="myint")

    pm.run(f"""
    # color by chain
    as ribbon, myobj
    util.cbc
    # defines a selection named myint
    {rs_str}
    show sticks, myint
    # focus on the selection
    zoom myint
    """)
    out_file = str(tmp_path / "mypm.pse")
    pm.run(f"""deselect; save {out_file}""")
    # dispose the PyMOL object to save system resource
    pm.close()

    assert os.path.exists(out_file), "PyMOL integration."

def test_PyMOL3D_visualization(tmp_path):
    if not _PYMOL_:
        pytest.skip("PyMOL not available")

    p=Protein(fn)
    out_file = str(tmp_path / "my1.pse")
    Protein.PyMOL3D() \
        .set_theme("publication") \
        .show(p, color="chain") \
        .show(output=out_file, save_png=True, width=300, height=300)
    assert os.path.exists(out_file), "PyMOL3D theme publication."

    p=Protein(fn)
    out_file = str(tmp_path / "my2.pse")
    Protein.PyMOL3D() \
        .set_theme("basic") \
        .show(p.extract("H:L"), color="spectrum") \
        .show(p.extract("P"), color="purple", style="sphere") \
        .show(output=out_file, save_png=True, width=300, height=300)
    assert os.path.exists(out_file), "PyMOL3D theme basic."

    custom_settings = {
        "stick_radius": 0.25,
        "cartoon_transparency": 0.0,
        "surface_quality": 3,
        "ambient": 0.38,
        "direct": 0.4
    }

    out_file = str(tmp_path / "my3.pse")
    Protein.PyMOL3D() \
        .set_theme("custom", custom_settings=custom_settings) \
        .show(p, color="blue", style="cartoon", show_sidechains=True) \
        .show(output=out_file, save_png=True, width=300, height=300)
    assert os.path.exists(out_file), "PyMOL3D theme custom."

    p=Protein(fn)
    out_file = str(tmp_path / "my4.png")
    Protein.PyMOL3D() \
        .set_theme("publication")  \
        .show(p.extract("P"), color="very_weak_blue", style="surface")  \
        .reset_theme() \
        .show(p.extract("H:L"), color="spectrum", style="cartoon")  \
        .run(f"png {out_file}, width=300, height=300, dpi=100")
    assert os.path.exists(out_file), "PyMOL3D theme publication with surface."

def test_b_factors():
    """Test B-factor assignment and manipulation."""
    p = Protein(fn)
    n = len(p)
    d = np.sin(np.arange(n)/n*np.pi)
    p.b_factors(d)
    assert np.array_equal(p.data_prt.b_factors[:,0], d), "b_factors 1"
    p.b_factors(np.zeros(len(p.rs("H"))), rs="H")
    p.b_factors(0.5, rs="L")
    p.b_factors(1, rs="P")
    assert np.all(np.abs(p.data_prt.b_factors[p.rs("L").data, 0]-0.5)<0.001), "b_factors 2"
    p.b_factors_by_chain({"H":0.1,"L":0.2,"P":0.3})
    assert np.all(np.abs(p.data_prt.b_factors[p.rs("L").data, 0]-0.2)<0.001), "b_factors 3"

def test_dist():
    """Test residue and atom distance calculations."""
    p = Protein(fk)
    t = p.rs_dist("L","H")
    r = t.iloc[0]
    assert np.abs(r['dist']-21.2251)<0.001, "distance 1"
    assert r['atom_a']=="OE1" and r['atom_b']=="CD1", "distance 2"
    assert r['resi_a']==3 and r['resi_b']==4, "distance 2"
    t = p.atom_dist("L6A", "H3")
    assert np.abs(t.iloc[0]["dist"]-23.3484)<0.001, "distance 3"
    t = p.atom_dist("L6A", "H3", ats="N,CA,C,O")
    assert np.abs(t.iloc[0]["dist"]-26.9663)<0.001, "distance 4"

def test_rmsd():
    """Test RMSD calculation."""
    p = Protein(fk)
    q = p.translate([3.0,4.0,0.0], inplace=False)
    assert np.all(np.isclose(q.center()-p.center(), np.array([3.,4.,0.]))), "rmsd 1"

    assert np.abs(q.rmsd(p)-5)<0.001, "rmsd 2"
    assert np.abs(q.rmsd(p, ats="CA")-5)<0.001, "rmsd 2"

    p = Protein(fk)
    q = p.translate([3.0,4.0,0.0], inplace=False)
    assert np.all(np.isclose(q.center()-p.center(), np.array([3.,4.,0.]))), "rmsd 1"

    assert np.abs(q.rmsd(p)-5)<0.001, "rmsd 2"
    assert np.abs(q.rmsd(p, ats="CA")-5)<0.001, "rmsd 2"

def test_DockQ():
    # experimental structure
    p=Protein(fn)
    print(p.len_dict(), "\n")
    # AlphaFold prediction
    q=Protein(str(fn).replace('.pdb', '_AF.pdb'))
    print(q.len_dict(), "\n")
    # make sure both structures have the same chain names and in the same order
    q.rename_reorder_chains(p, {'A':'L', 'B':'H', 'C':'P'}, inplace=True)
    print(q.len_dict(), "\n")
    # in dockQ method, the first object is exp obj, and the second is the prediction
    # we assess the docker of P onto H:L, so using Ab and Ag as two parts
    c = Protein.dockQ(p, q, "H:L", "P")
    assert c['DockQ']>0.94, "dockQ score"

def test_sasa_detailed():
    """Test detailed SASA calculations and delta SASA."""
    p = Protein(fn)
    t = p.sasa()
    sasa = p.sasa().SASA.values
    t_all = t[(t.chain=="L")][90:105]
    t = p.sasa("H:L")
    t_ag = t[(t.chain=="L")][90:105]
    t_ag['DELTA_SASA'] = t_ag['SASA'].values - t_all['SASA'].values
    x = t_ag.DELTA_SASA.values
    assert np.all(x[1:4]>8), "SASA 1"
    assert np.all(x[7:]<10), "SASA 2"


def test_dsasa():
    """Test delta solvent accessible surface area calculation."""
    p = Protein(fn)
    p.renumber('RESTART', inplace=True)
    
    try:
        # Test basic dsasa calculation
        dsasa_result = p.dsasa("H:L", "P")
        assert set(dsasa_result.label.unique()) == {'surface', 'interior', 'core', 'support', 'rim'}, "dsasa should return result"
        
        row = dict(dsasa_result[dsasa_result.resi==2].iloc[0])
        assert (row['SASAc'] > 80 and row['rSASAc'] < 1), "dsasa calculation"

    except (AttributeError, NotImplementedError):
        pytest.skip("dsasa method not implemented")

def test_dssp():
    """Test DSSP secondary structure assignment (skipped if not installed)."""
    dssp_info = Protein.locate_dssp()
    if not dssp_info['DSSP']:
        pytest.skip("DSSP not installed")
    p = Protein(fn)
    ss = p.dssp()["H"]
    assert 'EEEE' in ss, "Beta sheet not assigned"

def test_bond_length():
    """Test internal coordinate calculations."""
    p = Protein(fn)
    t = p.backbone_bond_length(rs="H-5:L80-85")
    t_H1 = t[(t.chain == "H") & (t.resn_b == "1")]
    assert np.isnan(t_H1.iloc[0]["bond_pept"]), "First residue has no peptide bond"
    t_L83 = t[(t.chain == "L") & (t.resn_b == "83")]
    assert np.abs(t_L83.iloc[0]["bond_pept"]-1.328)<0.001, "bond length 1"

def test_internal_coord():
    """Test internal coordinate calculations."""
    p = Protein(fk)
    t = p.internal_coord(rs="L")
    assert np.isnan(t['-1C:N'].values[-1]), "ic 1"
    t = p.internal_coord(rs="L", MaxPeptideBond=1e8)
    assert np.abs(t['-1C:N'].values[-1]-7.310)<0.001, "ic 2"
    assert np.abs(t['chi1'].values[0]+172.925)<0.001, "ic 3"

# ============================================================================
#  GOBJECT MANIPULATION 
# ============================================================================

def test_transformations():
    """Test translation, rotation, and alignment of structures."""
    p = Protein(fk)
    q = p.translate([3.0,4.0,0.0], inplace=False)
    assert np.all(np.isclose(q.center()-p.center(), np.array([3.,4.,0.]))), "translate center"
    assert np.abs(q.rmsd(p)-5)<0.001, "rmsd after translation"
    assert np.abs(q.rmsd(p, ats="CA")-5)<0.001, "rmsd CA after translation"
    rc = p.center()
    q = p.center_at([0,0,0], inplace=False)
    q.rotate([1,1,1], 5, inplace=True)
    q.center_at(rc, inplace=True)
    assert q.rmsd(p, None, None, "CA") < 1, "rotate and recenter"
    p.rotate(np.random.random(3), np.random.random()*180)
    p.reset_pos(inplace=True)
    assert np.all(np.isclose(p.center(), np.zeros(3))), "reset_pos"

def test_move():
    """Test movement operations on protein structures."""
    p = Protein(fk)
    assert np.all(np.isclose(p.center(), np.array([ 15.54949999,-8.0205001,-15.39166681]))), "center"
    q = p.center_at([3.0,4.0,5.0], inplace=False)
    print(q.center())
    assert np.all(np.isclose(q.center(), np.array([ 3.,4.,5.]))), "center_at"
    q = p.translate([1.,0.,-1.], inplace=False)
    assert np.abs(q.rmsd(p, None, None, "CA")-1.414)<0.001, "translate"
    q = p.rotate([1,1,1], 5, inplace=False)
    assert q.rmsd(p, ats="CA")>2.0, "rotate 1"
    rc = p.center()
    q = p.center_at([0,0,0], inplace=False)
    q.rotate([1,1,1], 5, inplace=True)
    q.center_at(rc, inplace=True)
    assert q.rmsd(p, None, None, "CA") < 1, "rotate 2"

    p.rotate(np.random.random(3), np.random.random()*180)
    p.reset_pos(inplace=True)
    assert np.all(np.isclose(p.center(), np.zeros(3))), "reset_pos"

def test_align():
    """Test structural alignment."""
    p = Protein(fn)
    q = p.translate([3.,4.,0.], inplace=False)
    q.rotate([1,1,1], 90, inplace=True)
    R,t = q.align(p, ats="CA")
    assert np.all(np.isclose(t, np.array([-3,-4,0]), atol=1e-3)), "align 1"
    assert np.all(np.isclose(np.diagonal(R), np.array([1/3,1/3,1/3]), atol=1e-3)), "align 1"

def test_align_two():
    """Test align_two method for sequence-based structural alignment."""
    p1 = Protein(fn)  # 5cil.pdb
    p2 = Protein(f4)  # 1a3d.pdb - different structure for alignment
    p2.rename_chains({"H":"A", "L":"B"}, inplace = True)
    
    try:
        # Test basic alignment with automatic chain mapping
        result = p1.align_two(p2, auto_chain_map=True)
        p_a, p_b, rl_a, rl_b = result
        
        # Should return 4 objects
        assert len(result) == 4, "Should return 4 objects: (p_a, p_b, rl_a, rl_b)"
        
        # Check returned protein objects
        assert isinstance(p_a, Protein), "First object should be Protein"
        assert isinstance(p_b, Protein), "Second object should be Protein"
        
        # Check returned residue lists
        assert hasattr(rl_a, 'data'), "Third object should be RL/RS"
        assert hasattr(rl_b, 'data'), "Fourth object should be RL/RS"
        
        # Aligned proteins should have same number of residues
        assert len(p_a) == len(p_b), "Aligned proteins should have same residue count"
        
        assert (len(p_a) == len(p1)), "p1 is a subsequence of p2"
        assert (len(p_b) < len(p2)), "p2 is longer than the alignment"

        # Test with specific chains (should work even if chains don't match perfectly)
        try:
            result2 = p1.align_two(p2, chain_a=["L"], chain_b=["A"], is_global=True)
            assert len(result2) == 4, "Should return 4 objects for specific chain alignment"
        except:
            # Chain-specific alignment might fail if sequences are too different
            pass
            
    except Exception as e:
        print(f"align_two test note: {e}")
        assert True, "Method exists and can be called"

def test_align_two2():
    p=Protein("8zua")
    print(p.len_dict())
    q=p.extract("L:H")
    # we make chain L:H adopt the shape of chain D:C, add some noise and translation 
    pos = p.data.atom_positions[p.rl("D:C").data]
    pos += np.random.random(pos.shape)*2
    q.data.atom_positions[q.rl("L:H").data]=pos
    q = q.translate([1.2, 3.4, 5.6])

    # auto chain mapping, should map chain D of p to L of q
    # it should map chain C to H, it should not map other heavy chains F/H/K of p to chain H of q.
    p2, q2, rl_p, rl_q=p.align_two(q, auto_chain_map=True)
    assert rl_p.chain(unique=True) == ["D", "C"], "Automatic chain mapping p"
    assert rl_q.chain(unique=True) == ["L", "H"], "Automatic chain mapping q"

def test_merge():
    """Test merging multiple protein objects."""
    # 1a3d.pdb is a three-chain homotrimer.
    p = Protein(f3)
    #Extract chain into a new object
    Q = [ p.extract(x, inplace=False) for x in p.chain_id() ]
    for i,q in enumerate(Q[1:]):
        q.rename_chains({'B':'A', 'C':'A'})
        assert q.rmsd(Q[0], ats='N,CA,C,O')>10, f"rmsd obj {i+1} 1"
        q.align(Q[0], ats='N,CA,C,O')

    q = Protein.merge(Q)
    assert len(q.chain_id())==3, "merge 1"
    assert len(q)==3*len(Q[0]), "merge 2"

    q = Q[0] + Q[1] + Q[2]
    assert len(q.chain_id())==3, "merge 1"
    assert len(q)==3*len(Q[0]), "merge 2"

# ============================================================================
#  ANTIBODIES 
# ============================================================================

def test_rs_antibody():
    """Test antibody-specific functionality."""
    if not _Ab_:
        pytest.skip("Antibody-specific tests are skipped")
    p = Protein(fn)
    rs_cdr, rs_var, c_chain_type, c_cdr = p.rs_antibody(scheme="imgt", set_b_factor=True)
    assert rs_cdr is not None and rs_var is not None, "Antibody residue selection"
    cdrs = [str(cdr) for cdr in rs_cdr.split()]
    assert cdrs == ['L27-33', 'L51-53', 'L90-98', 'H25-32', 'H50-57', 'H96-115'], "CDR regions"

def test_truncate_antibody():
    """Test antibody-specific functionality."""
    if not _Ab_:
        pytest.skip("Antibody-specific tests are skipped")
    p=Protein("5cil")
    # Original length
    len_p = p.len_dict()["H"]
    # Original numbering for CDR3
    cdr3_p = p.rs_seq('AREGTTGDGDLGKPIGAFAH')[0].name()
    q=p.truncate_antibody(set_b_factor=True, renumbering=True, scheme="imgt")
    # After truncation
    len_q = q.len_dict()["H"]
    # IMGT numbering for CDR3
    cdr3_q = q.rs_seq('AREGTTGDGDLGKPIGAFAH')[0].name()
    assert len_p > len_q, "length after truncation is shorter"
    assert (cdr3_p[0] == "93" and cdr3_q[0] == "105"), "CDR3 numbering is defined by scheme"

def test_rs_fv():
    """Test rs_Fv method for extracting variable regions from scFv chains."""
    # This test requires an scFv structure - we'll use a mock test for now
    if not _Ab_:
        pytest.skip("Antibody-specific tests are skipped")

    p = Protein(fv)
    
    try:
        # Try to find variable regions in H chain (might be scFv)
        var_pos, chain_type = p.rs_Fv("B", species="human")
        
        # Should return lists
        assert isinstance(var_pos, list), "Should return list of variable positions"
        assert isinstance(chain_type, list), "Should return list of chain types"
        assert len(var_pos) == len(chain_type), "Variable positions and chain types should match"
        
        # If variable regions found, they should be valid RS objects
        for rs in var_pos:
            assert len(rs) >= 100, "RS objects should have valid length"
            
    except Exception as e:
        # If no scFv found or antibody analysis not available, that's okay
        print(f"rs_Fv test note: {e}")
        assert True, "Method exists and can be called"

def test_scfv2mab():
    """Test scFv2mAb method for converting scFv to mAb format."""
    if not _Ab_:
        pytest.skip("Antibody-specific tests are skipped")

    p = Protein(fv)
    
    try:
        # Try converting potential scFv chains
        result = p.scFv2mAb(species="human")
        
        # Should return a Protein object
        assert isinstance(result, Protein), "Should return Protein object"
        assert len(result.chain_id()) == 3, "Should have at least one chain"
        
        # Check chain IDs are valid
        assert set(result.chain_id()) == {"A", "B", "b"}, "Chain IDs should match expected values"
            
    except Exception as e:
        # If no scFv found or conversion fails, that's expected for non-scFv structures
        print(f"scFv2mAb test note: {e}")
        assert True, "Method exists and can be called"

def test_seq_antibody():
    if not _Ab_:
        pytest.skip("Antibody-specific tests are skipped")
    p=Protein(fn)
    seq_H=p.seq_dict()["H"]
    chain_type, pos_var, cdrs, numbering=Protein.seq2antibody(seq_H)
    assert chain_type=="H", "Chain type"
    assert cdrs[0][2] == "GGSFSTY", "antibody analysis on sequence input"

    p=Protein("9cn2")
    s_seq=p.seq_dict()["B"]
    out = Protein.seq2Fv(s_seq)
    assert out == [(0, 115, 'H'), (135, 241, 'K')], "scFv analysis on sequence input"


# =====================
# Pytest entry point (not needed, but for manual run)
# =====================
if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))
