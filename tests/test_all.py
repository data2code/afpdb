#!/usr/bin/env python
import afpdb.util as util
from afpdb.afpdb import Protein,RS,ATS
import os
import numpy as np
import afpdb.myalphafold.common.residue_constants as afres

fn = os.path.join(os.path.dirname(__file__), "5cil.pdb")
fk = os.path.join(os.path.dirname(__file__), "fake.pdb")
f3 = os.path.join(os.path.dirname(__file__), "1a3d.pdb")
f4 = os.path.join(os.path.dirname(__file__), "5cil_100.pdb")

def check_p(p):
    return ((set(p.chain_id()) == {"L","H","P"}) and p.seq()=='EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTV:VQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGDGDLGKPIGAFAHWGQGTLVTVSS:NWFDITNWLWYIK')

### Demo
def test_demo():
    p=Protein(fn)
    c=p.seq_dict()
    assert c['P']=='NWFDITNWLWYIK', "Load object and seq_dict()"

    rs_binders, rs_seed, df_dist=p.rs_around("P", dist=4, drop_duplicates=True)
    assert str(rs_binders)=="L33,92-95,97:H30,32,46,49-51,54,56-58,98,106-109", "rs_around, rs2str"

    assert(len(df_dist)==21), "df_dist"
    p=p.extract(rs_binders | "P")
    p.save("test.pdb")
    assert(len(p)==34)

### Data structure
def test_data():
    p=Protein(fk)
    c=p.len_dict()
    assert(c["H"]==2 and c["L"]==4)

    g=p.data_prt
    assert(np.array_equal(g.chain_index, np.array([0,0,0,0,1,1]))), "chain_index"
    assert(np.array_equal(g.chain_id, np.array(["L","H"]))), "chain_id"
    assert(np.array_equal(g.aatype, np.array([6,9,19,5,10,19]))), "aatype"
    assert(np.array_equal(np.array(g.atom_positions.shape), np.array([6,37,3]))), "atom_positoins"
    assert(np.all(g.atom_mask[:, afres.atom_order['CA']]==1)), "atom_mask, CA"
    assert(np.all(g.atom_mask[np.array([1,2,5]), afres.atom_order['CG']]==0)), "atom_mask CG"
    assert(np.all(np.abs(g.atom_positions[:, afres.atom_order['CA']])>1)), "atom_positions, CA"
    assert(np.all(np.abs(g.atom_positions[np.array([1,2,5]), afres.atom_order['CG']])<1)), "atom_positions, CG"

### Contig
def test_contig():
    p=Protein(fn)
    assert(len(p.rs("L11"))==1)
    assert(len(p.rs("H-5:H10-15:L-10"))==21)
    assert(np.array_equal(p.rs("H-5:H10-15:L-10").data, p.rs("H-5,10-15:L-10").data))

### inplace & clone
def test_inplace():
    p=Protein(fn)
    q=p.extract("H:L")
    assert (p!=q), "inplace 1"

    q=p.extract("H:L", inplace=True)
    assert p==q, "inplace 2"
    assert not check_p(p)

    p=Protein(fn)
    q=p.clone()
    q.extract("H:L", inplace=True)
    assert check_p(p), "clone"

### Read/Write

def test_local():
    p=Protein(fn)
    assert check_p(p)

def test_save():
    p=Protein(fn)
    p.save("test.pdb")
    p=Protein("test.pdb")
    assert check_p(p), "save to pdb file"
    p.save("test.cif")
    p=Protein("test.cif")
    assert check_p(p), "save to and read from cif file"
    if os.path.exists("test.pdb"): os.remove("test.pdb")
    if os.path.exists("test.cif"): os.remove("test.cif")

def test_pdb():
    p=Protein("1crn")
    assert len(p)==46, f"create from PDB"

def test_embl():
    p=Protein("Q2M403")
    print(len(p))
    assert len(p)==458, f"create from EMBL"

def test_alphafold():
    p=Protein(fn)
    p=p.data_prt
    p=Protein(p)
    assert check_p(p), "create from DeepMind protein object"

def test_biopython():
    p=Protein(fn)
    p=Protein(p.to_biopython())
    assert check_p(p), "create from BioPython object"

def test_pdb_str():
    p=Protein(fn)
    p=Protein(p.to_pdb_str())
    assert check_p(p), "create from PDB string"

def test_missing():
    p=Protein(fn)
    q=p.extract(~p.rs("H10-14"))
    rs_missing=p.rs_insertion(q)
    assert str(rs_missing)=="H10-14"
    q=p.extract("H1-10,15-:L-20,25-")
    assert str(q.rs_next2missing()) == "H10-15:L20-25"

### Sequence, missing residues, insertion code
def test_inplace():
    p=Protein(fn)
    check_p(p), "Sequence 1"

    c=p.seq_dict()
    assert c["P"]=='NWFDITNWLWYIK', "Sequence 2"
    assert c["H"]=='VQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGDGDLGKPIGAFAHWGQGTLVTVSS', "Sequence 3"
    assert c["L"]=='EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTV', "Sequence 4"

    p=Protein(fk)

    assert np.array_equal(p.data_prt.residue_index, np.array(['5','6A','6B','10','3','4'])), "Insertion code"

    assert p.seq()=="EIVXXXQ:LV", "missing X"

    assert p.seq(gap="")=="EIVQ:LV", "gap for missing"

    p=Protein(fn)
    assert len(p)==250, "len"
    assert p.len_dict()["H"]==126, "len_dict"
    out=p.rs_seq('LTI')
    for x in out:
        print(str(x), x.seq())
    assert str(RS._or(* p.rs_seq('LTI')))=='L74-76:H54-56', "seq_search"

### Mutagenesis
def test_thread():
    # currently thread_seq, pymol is not forced to be installed
    pass

### Chain
def test_chain():
    p=Protein(fn)
    assert "".join(p.chain_id())=="LHP", "chain 1"

    p.reorder_chains(["P","L","H"], inplace=True)
    assert not check_p(p), "chain 2"
    assert "".join(p.chain_id())=="PLH", "chain 3"

    p=Protein(fn)
    c=p.chain_pos()
    assert c["L"][0]==0 and c["L"][1]==110, "chain_pos 1"
    assert c["H"][0]==111 and c["H"][1]==236, "chain_pos 1"
    assert c["P"][0]==237 and c["P"][1]==249, "chain_pos 1"

    p=Protein(fk)
    q=p.data_prt
    p.renumber(None, inplace=True)
    assert ",".join(q.residue_index)=="5,6A,6B,10,3,4", "_renumber None"
    p.renumber("RESTART", inplace=True)
    assert ",".join(q.residue_index)=="1,2A,2B,6,1,2", "_renumber RESTART"
    p.renumber("CONTINUE", inplace=True)
    assert ",".join(q.residue_index)=="1,2A,2B,6,7,8", "_renumber CONTINUE"
    p.renumber("GAP33", inplace=True)
    assert ",".join(q.residue_index)=="1,2A,2B,6,40,41", "_renumber GAP33"
    p.renumber("NOCODE", inplace=True)
    assert ",".join(q.residue_index)=="1,2,3,7,8,9", "_renumber NOCODE"

    p=Protein(fn)
    q, c_pos=p.merge_chains(gap=200, inplace=False)
    assert c_pos["H"][0]==111 and c_pos["H"][1]==236, "merge_chain 1"
    assert len(q.chain_id())==1, "merge chain 2"
    r=q.split_chains(c_pos, inplace=False)
    c_pos=r.chain_pos()
    assert len(r.chain_id())==3 and c_pos["H"][0]==111 and c_pos["H"][1]==236, "merge_chain 3"

### Selection
def test_ats():
    p=Protein()
    assert np.array_equal(ATS("N,CA,C,O").data, np.array([0,1,2,4])), "ats 1"
    assert np.array_equal(ATS(["N","CA","C","O"]).data, np.array([0,1,2,4])), "ats 2"
    assert len(p.ats(""))==0, "ats 3"
    assert len(p.ats(None))==37, "ats 4"
    assert len(p.ats("ALL"))==37, "ats 5"
    assert len(p.ats("NULL"))==0, "ats 6"
    assert np.array_equal(p.ats_not(["N","CA","C","O"]).data, p.ats("CB,CG,CG1,CG2,OG,OG1,SG,CD,CD1,CD2,ND1,ND2,OD1,OD2,SD,CE,CE1,CE2,CE3,NE,NE1,NE2,OE1,OE2,CH2,NH1,NH2,OH,CZ,CZ2,CZ3,NZ,OXT").data), "ats 6"

def test_rs():
    p=Protein(fn)
    rs1=p.rs("H1-3")
    rs2=p.rs("H3-5:L-3")
    assert len(p.rs_and(rs1, rs2))==1, "rs_and"
    assert len(p.rs_or(rs1, rs2))==8, "rs_or"
    assert len(p.rs_not("H:L"))==13, "rs_not"
    assert len(p.rs_not("L-100", rs_full="L"))==11, "rs_notin"
    assert p.rs2str(p.rs_or(rs1, rs2))=="L1-3:H1-5", "rs2str"

    p=Protein(fn)
    p=p.extract(rs="H-5:L-5", ats="N,CA,C,O")
    assert len(p)==10, "rs and ats"
    # no side chain
    assert np.sum(p.data_prt.atom_mask[:, p.ats("CB").data])==0, "backbone only"

def test_rs2():
    p=Protein(fn)
    rs1=RS(p, "H1-3")
    rs2=RS(p, "H3-5:L-3")
    assert len(rs1 & rs2)==1, "rs_and"
    assert len(rs1 | rs2)==8, "rs_or"
    assert len(~RS(p, "H:L"))==13, "rs_not"
    assert len(RS(p, "L-100")._not(rs_full="L"))==11, "rs_notin"
    assert str(rs1 | rs2)=="L1-3:H1-5", "rs2str"
    assert len(RS._or(RS(p,"H1-3"), "H10", "H11", "H12", "H13"))==7, "rs_or"
    assert len(RS._and(RS(p,"H1-10"), "H3-20", "H5-22"))==6, "rs_and"
    assert len(rs1 - rs2)==2, "rs_minus"
    rs1 |= rs2
    assert (len(rs1) == 8), "rs_ior"
    rs1 &= rs2
    assert (len(rs1) == len(rs2)), "rs_iand"
def test_rs2str():
    p=Protein(f4)
    assert p.rs2str("L5-10,100:H")=="H:L5-10,100", "rs2str 1"
    assert p.rs2str("L5-10,100:H", format="PYMOL")=="select rs, (chain H) or (chain L and resi 5-10+100)", "rs2str 2"

    assert p.rs2str("H100A-102", format="PYMOL")=="select rs, (chain H and resi 100A+100B+100C+100D+100E+100F+100G+100H+100I+100J+101-102)", "rs2str 3"
    assert p.rs2str("H100-102", format="PYMOL")=="select rs, (chain H and resi 100-102)", "rs2str 4"
    assert p.rs2str("H100-100D", format="PYMOL")=="select rs, (chain H and resi 100+100A+100B+100C+100D)", "rs2str 5"
    assert p.rs2str("H100D-100F", format="PYMOL")=="select rs, (chain H and resi 100D+100E+100F)", "rs2str 6"
    assert p.rs2str("H98-100,100D,100F-102", format="PYMOL")=="select rs, (chain H and resi 98-99+100+100D+100F+100G+100H+100I+100J+101-102)", "rs2str 7"

def test_canonicalize_rs():
    p=Protein(fn)
    # an unusal selection
    rs=p.rs("L1-5:P12-13:L6-10")
    try:
        p.extract(rs)
        assert False, "unusal selection 1"
    except Exception as e:
        assert True, "unusal selection 1"
    try:
        p.extract(p.rs2str(rs))
        assert True, "unusal selection 2"
    except Exception as e:
        assert False, "unusal selection 2"

def test_rs_around():
    p=Protein(fn)
    rs=p.rs("P")
    rs_nbr, r_seed, t=p.rs_around(rs, dist=3.5, drop_duplicates=True)
    assert p.rs2str(rs_nbr)=="L33,92-95:H30,56,58,98,106-109", "rs_around 1"
    assert len(t)==len(rs_nbr), "rs_around 2"

def test_residue_id():
    p=Protein(fn)
    rs=p.rs("P")
    rs_nbr, rs_seed, t=p.rs_around(rs, dist=3.5, drop_duplicates=True)
    t.display()
    t2=t[(t.chain_b=="H")&(t.resn_b>="95")&(t.resn_b<="106")]
    t2.display()
    assert len(t2)==0, "residue filter 1"
    t2=t[(t.chain_b=="H")&(t.resn_i_b>=95)&(t.resn_i_b<=106)]
    t2.display()
    assert len(t2)==2, "residue filter 2"

### Display
def test_html():
    p=Protein(fn)
    s=p.html()
    assert "ATOM      1  N   GLU A   1" in s, "html"

### B-factors
def test_b_factors():
    p=Protein(fn)
    n=len(p)
    d=np.sin(np.arange(n)/n*np.pi)
    p.b_factors(d)
    assert np.array_equal(p.data_prt.b_factors[:,0], d), "b_factors 1"

    p.b_factors(np.zeros(len(p.rs("H"))), rs="H")
    p.b_factors(0.5, rs="L")
    p.b_factors(1, rs="P")
    assert np.all(np.abs(p.data_prt.b_factors[p.rs("L").data, 0]-0.5)<0.001), "b_factors 2"

    p.b_factors_by_chain({"H":0.1,"L":0.2,"P":0.3})
    assert np.all(np.abs(p.data_prt.b_factors[p.rs("L").data, 0]-0.2)<0.001), "b_factors 3"

### Distance
def test_dist():
    p=Protein(fk)
    t=p.rs_dist("L","H")
    r=t.iloc[0]
    assert np.abs(r['dist']-21.2251)<0.001, "distance 1"
    assert r['atom_a']=="OE1" and r['atom_b']=="CD1", "distance 2"
    assert r['resi_a']==3 and r['resi_b']==4, "distance 2"

    t=p.atom_dist("L6A", "H3")
    assert np.abs(t.iloc[0]["dist"]-23.3484)<0.001, "distance 3"

    t=p.atom_dist("L6A", "H3", ats="N,CA,C,O")
    assert np.abs(t.iloc[0]["dist"]-26.9663)<0.001, "distance 4"

### RMSD, translate, rotate
def test_rmsd():
    p=Protein(fk)
    q=p.translate([3.0,4.0,0.0], inplace=False)
    assert np.all(np.isclose(q.center()-p.center(), np.array([3.,4.,0.]))), "rmsd 1"

    assert np.abs(q.rmsd(p)-5)<0.001, "rmsd 2"
    assert np.abs(q.rmsd(p, ats="CA")-5)<0.001, "rmsd 2"

    p=Protein(fk)
    q=p.translate([3.0,4.0,0.0], inplace=False)
    assert np.all(np.isclose(q.center()-p.center(), np.array([3.,4.,0.]))), "rmsd 1"

    assert np.abs(q.rmsd(p)-5)<0.001, "rmsd 2"
    assert np.abs(q.rmsd(p, ats="CA")-5)<0.001, "rmsd 2"

### SASA
def test_sasa():
    p=Protein(fn)
    t=p.sasa()
    sasa=p.sasa().SASA.values
    t_all=t[(t.chain=="L")][90:105]
    t=p.sasa("H:L")
    t_ag=t[(t.chain=="L")][90:105]
    t_ag['DELTA_SASA']=t_ag['SASA'].values-t_all['SASA'].values
    t_ag.display()
    x=t_ag.DELTA_SASA.values
    assert np.all(x[1:4]>8), "SASA 1"
    assert np.all(x[7:]<10), "SASA 2"

### DSSP
def test_dssp():
    # currently DSSP is not forced to be installed
    pass

### Internal Coordinate
def test_ic():
    p=Protein(fk)
    t=p.internal_coord(rs="L")
    assert np.isnan(t['-1C:N'].values[-1]), "ic 1"

    t=p.internal_coord(rs="L", MaxPeptideBond=1e8)
    assert np.abs(t['-1C:N'].values[-1]-7.310)<0.001, "ic 2"

    assert np.abs(t['chi1'].values[0]+172.925)<0.001, "ic 3"

### Move Object
def test_move():
    p=Protein(fk)
    assert np.all(np.isclose(p.center(), np.array([ 15.54949999,-8.0205001,-15.39166681]))), "center"
    q=p.center_at([3.0,4.0,5.0], inplace=False)
    print(q.center())
    assert np.all(np.isclose(q.center(), np.array([ 3.,4.,5.]))), "center_at"
    q=p.translate([1.,0.,-1.], inplace=False)
    assert np.abs(q.rmsd(p, None, None, "CA")-1.414)<0.001, "translate"
    q=p.rotate([1,1,1], 5, inplace=False)
    assert q.rmsd(p, ats="CA")>2.0, "rotate 1"
    rc=p.center()
    q=p.center_at([0,0,0], inplace=False)
    q.rotate([1,1,1], 5, inplace=True)
    q.center_at(rc, inplace=True)
    assert q.rmsd(p, None, None, "CA") < 1, "rotate 2"

    p.rotate(np.random.random(3), np.random.random()*180)
    p.reset_pos(inplace=True)
    assert np.all(np.isclose(p.center(), np.zeros(3))), "reset_pos"

def test_align():
    p=Protein(fn)
    q=p.translate([3.,4.,0.], inplace=False)
    q.rotate([1,1,1], 90, inplace=True)
    R,t=q.align(p, ats="CA")
    assert np.all(np.isclose(t, np.array([-3,-4,0]), atol=1e-3)), "align 1"
    assert np.all(np.isclose(np.diagonal(R), np.array([1/3,1/3,1/3]), atol=1e-3)), "align 1"

### Merge Object
def test_merge():
    # 1a3d.pdb is a three-chain homotrimer.
    p=Protein(f3)
    #Exact chain into a new object
    Q=[ p.extract(x, inplace=False) for x in p.chain_id() ]
    for i,q in enumerate(Q[1:]):
        q.rename_chains({'B':'A', 'C':'A'})
        assert q.rmsd(Q[0], ats='N,CA,C,O')>10, f"rmsd obj {i+1} 1"
        q.align(Q[0], ats='N,CA,C,O')
        assert q.rmsd(Q[0], ats='N,CA,C,O')<0.001, f"rsmd obj {i+1} 1 after align"
    q=Protein.merge(Q)
    assert len(q.chain_id())==3, "merge 1"
    assert len(q)==3*len(Q[0]), "merge 2"

if __name__=="__main__":
    #test_align()
    #test_rs2str()
    test_b_factors()
