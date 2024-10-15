#!/usr/bin/env python
# code taken from
# https://github.com/nrbennet/dl_binder_design/blob/main/mpnn_fr/dl_interface_design.py#L185
import sys
import os, shutil
import re
import tempfile
from Bio.PDB import PDBParser
from .afpdb import Protein,util
import json
import numpy as np
import traceback

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

def remove_files(S_file):
    for x in S_file:
        if os.path.exists(x): os.remove(x)

class ThreadSeq:

    USE_AMBER=True # pyrosetta relax does not seem to work, use amber instead

    def __init__(self, pdb_file):
        self.pdb=pdb_file
        self.p=Protein(pdb_file)
        self.old_seq=self.p.seq_dict()
        self.chains=self.p.chain_list()
        # identify residues with missing atoms
        # these residues need to be mutated regardless
        self.miss_res=self.p.rs_missing_atoms()

    def run(self, out_file, seq, replace_X_with='A', relax=0, seq2bfactor=False, amber_gpu=False, cores=1, side_chain_pdb=None, rl_from=None, rl_to=None):
        """replace_X_with, if there is unrecognized residue in seq, use A or G,
            A is preferred. If the original PDB has CB, G might fail (have not verified)
            otherwise, give an error

            sometimes not all residues can be successfully thethers, so we return the final sequence
            Example: in AbDb, 2J6E_1.pdb contains P and G at the end of chain A & B, however, they miss Ca,
            so these two terminal aa will be dropped in the output .pdb by PyMOL

            seq can be a str for monomer, a json str or a dict for multimer
            {"A": "VAPLHLGKCNIAG", "L":"IVGGTASVRGEWPWQVTLHTT"}
        """

        # parse seq object
        if type(seq) is str:
            if seq=='': # empty sequence means we want to fix residues with missing atoms
                seq=self.old_seq
            elif '{' in seq:
                seq=json.loads(seq)
            else: #multi-chain format
                S=re.split(r'\W', seq)
                if len(self.chains)!=len(S):
                    raise Exception(f"""When using a colon-delimited string, there are {len(S)} chains in your sequence, but {len(self.chains)} chains in PDB!""")
                seq={k:v for k,v in zip(self.chains, S)}
        if len(seq)==0: seq=self.old_seq

        # check seq is sensible
        for k,v in seq.items():
            if k not in self.old_seq:
                print(f"ERROR ThreadSeq> Bad chain id: {k}")
                return {"ok":False}
            self.old_seq[k]=self.old_seq[k].replace('X', '')
            if len(v)!=len(self.old_seq[k]):
                print(f"Sequence length mistach for chain {k}: PDB has {len(self.old_seq[k])} residues, -s contains {len(v)}!")
                print(f"PDB:  {self.old_seq[k]}")
                print(f"-s:   {v}")
                return {"ok":False}
        #print(self.miss_res)

        if side_chain_pdb is not None: # we need to copy side chains first
            tmp=tempfile.NamedTemporaryFile(delete=False, prefix="_THREAD", suffix=".pdb")
            shutil.copyfile(self.pdb, tmp.name)
            self.p=Protein.copy_side_chain(tmp.name, side_chain_pdb, rl_from=rl_from, rl_to=rl_to)
            if self.p is None:
                print(f"ERROR ThreadSeq> Fail to copy side chains: {out_file}, not found!")
                return {"ok":False}
            self.p.save(tmp.name)
            self.pdb=tmp.name
            print("Generated new input PDB: ", tmp.name)
            # these should not change
            self.old_seq=self.p.seq_dict()
            self.chains=self.p.chain_list()
            #
            self.miss_res=self.p.rs_missing_atoms()
            #self.pdb=tmp.name

        # threading
        try:
            mutations=self.mutate_seq_pymol(out_file, seq, replace_X_with=replace_X_with)
        except Exception as e:
            print(traceback.format_exc())
            print("ERROR ThreadSeq> PyMOL failed to mutate the protein. No output is generated.")
            return {"ok":False}

        if not os.path.exists(out_file):
            print(f"ERROR ThreadSeq> expecting mutated output file: {out_file}, not found!")
            return {"ok":False}

        # relaxation
        relax_flag=False
        if relax>0 and len(mutations)>0: # if there is no mutation, we don't need relax
            if ThreadSeq.USE_AMBER:
                relax_flag=self.relax_amber(out_file, amber_gpu, cores)

        # extra steps
        if seq2bfactor:
            try:
                if side_chain_pdb is not None and rl_from is not None and rl_to is not None:
                    self.add_b_factor_rl(out_file, rl_to)
                else:
                    self.add_b_factor(out_file, rl_to, seq)
            except Exception as e:
                print(traceback.format_exc())
                print("Skip b factor!!!!!")
        else:
            self.remove_hydrogen(out_file)

        # return JSON
        data={}
        p=Protein(out_file)
        out_seq=p.seq_dict()
        same=True
        for k,v in seq.items():
            if v.upper()!=out_seq.get(k, ""): same=False
        data["output_pdb"]=out_file
        data["ok"]=True
        data["output_equal_target"] = same
        data["input"]=self.old_seq
        data["output"]=out_seq
        data["target"]=seq
        data["relax"]=relax_flag
        data["residues_with_missing_atom"]=str(self.miss_res)
        data["mutations"]=mutations
        print("###JSON STARTS")
        print(json.dumps(data))
        print("###JSON END")
        return data

    def relax_amber(self, out_file, amber_gpu=False, cores=1):
        if not os.path.exists("/da/NBC/ds/lib/protein/amberrelax.py"): return False
        cuda_env='' if amber_gpu else 'CUDA_VISIBLE_DEVICES= ' # looks like we need to unset CUDA env to prevent it from using GPU
        out_file2=out_file.replace('.pdb', '.relaxed.pdb')
        # we use different names to make sure amber relax indeed runs
        cmd=f"source /da/NBC/ds/bin/envs/colabfold.env && {cuda_env}OPENMM_CPU_THREADS={cores} OMP_NUM_THREADS={cores} MKL_NUM_THREADS={cores} NUMEXPR_MAX_THREADS={cores} python /da/NBC/ds/lib/protein/amberrelax.py {'--gpu' if amber_gpu else ''} {out_file} {out_file2}"
        print(cmd)
        util.unix(cmd)
        if os.path.exists(out_file2):
            print("INFO> Amber relax is successful!")
            relax_flag=True
            os.replace(out_file2, out_file)
            return True
        return False

    def mutate_seq_pymol(self, out_file, seq, replace_X_with='A'):
        #from pymol import cmd
        # make it thread safe, see https://pymolwiki.org/index.php/Launching_From_a_Script
        try:
            import pymol2
        except:
            print("Please install PyMOL with:\nconda install conda-forge::pymol-open-source")
            exit()

        def mutate(session, molecule, chain, resi, target="CYS", mutframe="1"):
            target = target.upper()
            session.cmd.wizard("mutagenesis")
            session.cmd.do("refresh_wizard")
            session.cmd.get_wizard().set_mode("%s" % target)
            selection = "/%s//%s/%s" % (molecule, chain, resi)
            session.cmd.get_wizard().do_select(selection)
            session.cmd.frame(str(mutframe))
            session.cmd.get_wizard().apply()
            # cmd.set_wizard("done")
            session.cmd.set_wizard()

        with pymol2.PyMOL() as session:
            session.cmd.load(self.pdb, "myobj")
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("myobj", self.pdb)
            res_dict = {}
            mutations=[]
            missing=self.miss_res
            for model in structure:
                for chain in model:
                    c=chain.id
                    if c not in seq: continue # this chain is skipped
                    cnt=0
                    #missing=self.miss_res.get(c, [])
                    for residue in chain:
                        resi=residue.id[1]
                        mut_to=seq[c][cnt].upper()
                        cnt+=1
                        if mut_to not in restype_1to3:
                            if replace_X_with in restype_1to3:
                                mut_to=replace_X_with
                            else:
                                raise Exception(f"ERROR PyMOL> Unrecognized residue {mut_to} at position {resi+1}!")
                        name3 = restype_1to3[ mut_to ]
                        # skip if no mutation, use existing sidechain
                        # but force mutation if there were missing atoms in the original PDB
                        if name3==residue.resname and (resi-1) not in missing: continue
                        print("MUTATE PyMOL> Old ", chain.id, resi, residue.resname, ">>> New", name3)
                        mutate(session, "myobj", chain.id, resi=resi, target=name3, mutframe="1")
                        mutations.append((chain.id, resi, residue.resname, name3))
            # pymol reorder the chain by alphabetic order
            session.cmd.save(out_file)
        return mutations

    def remove_hydrogen(self, out_file):
        # Remove hydrogens
        pdb_str=util.read_list(out_file)
        out=[]
        for i,s in enumerate(pdb_str):
            if s.startswith("ATOM ") and s[76:78]==" H":
                continue
            else:
                out.append(s)

        data_str="\n".join(out)
        p = Protein(pdb_str=data_str)
        p.save(out_file)

    def add_b_factor(self, out_file, seq):
        print(out_file)
        p=Protein(out_file)
        x=p.seq_dict()
        C_b={}
        print(seq)
        for k,v in seq.items():
            C_b[k]=np.zeros(len(v))
            for i,res in enumerate(v):
                C_b[k][i]=0.5 if res==res.lower() else 1.0
        p.b_factors_by_chain(C_b)
        p.save(out_file)

    def add_b_factor_rl(self, out_file, rl_to):
        print(out_file)
        p=Protein(out_file)
        p.b_factors(0.5)
        p.b_factors(1, rs=rl_to)
        p.save(out_file)

if __name__=="__main__":
    print("INFO> For multi-chain structures, provide one sequence in the order of how they appear in PDB, chain sequences can be optionally separated by space or colon for the sake of clarity.\n")
    print("INFO> if sequence is not specified, no mutation is done")
    print("Input PDB only needs to contain N, CA, C")
    print('EXAMPLE> ./thread_seq.py -i myexample/mybb.pdb -o my_fill.pdb -s "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLKKKKKKKKKKTTTT:SS"')
    print('EXAMPLE> ./thread_seq.py -i myexample/1crn.pdb1 -o my_fill.pdb -s "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"')
    import argparse as arg
    opt=arg.ArgumentParser(description='Thread a sequence onto a PDB template')
    opt.add_argument('-i', '--input', type=str, default=None, help='input PDB file', required=True)
    opt.add_argument('-o', '--output', type=str, default=None, help='output PDB file', required=True)
    opt.add_argument('-s','--sequence', type=str, default="", help='sequence, empty sequence means we will fix residues with missing atoms')
    opt.add_argument('-r','--relax', type=int, default=100, help='relax cycles')
    opt.add_argument('-x','--replace_X', type=str, default='', help='replace unknown residues in sequence by specied one-letter code (e.g., A or G for Alanine/Glycine)')
    opt.add_argument('-b','--seq2bfactor', action='store_true', default=False, help='Convert upper/lower case in sequence into bfactor 1.0 and 0.5, useful to highlight fixed/hallucinated residues from RFDiffusion output')
    opt.add_argument('-g','--gpu', action='store_true', default=False, help='Use GPU in amber')
    opt.add_argument('-c','--cores', type=int, default=1, help='Number of CPU cores used for amber')
    opt.add_argument('--scpdb', type=str, default=None, help='PDB file provides side chain coordinate')
    # scmap is replaced by contig_scpdb and contig_input, two contigs for residue lists afpdb.RL
    #opt.add_argument('--scmap', type=str, default=None, help='JSON maps chain names, scpdb into input pdb')
    opt.add_argument('--contig_scpdb', type=str, default=None, help='Contig string specifying the fixed residues in scpdb')
    opt.add_argument('--contig_input', type=str, default=None, help='Contig string specifying the fixed residues in input')

    #thread_seq("1cmp.pdb1", "my_fill.pdb", "L"*44+":"+"TK")
    args = opt.parse_args()

    if args.gpu:
        util.warn_msg('GPU for amber does not work most of the time!!!')
    ts=ThreadSeq(args.input)
    ts.run(args.output, args.sequence, relax=args.relax, replace_X_with=args.replace_X, seq2bfactor=args.seq2bfactor, amber_gpu=args.gpu, cores=args.cores, side_chain_pdb=args.scpdb, rl_from=args.contig_scpdb, rl_to=args.contig_input)
    # thread_seq('/da/NBC/ds/zhoubi1/ides/data/init_guess/a.pdb',
    #            '/da/NBC/ds/zhoubi1/ides/data/init_guess/6a4k.pdb',
    #            'VAPLHLGKCNIAGWILGNPECESLSTASSWSYIVETPSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAACPHAGAKSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPSTSADQQSLYQNADAYVFVGSSRYSKKFKPEIAIRPKVRDQEGRMNYYWTLVEPGDKITFEATGNLVVPRYAFAMERNAGSGIIISD:QVQLQESGPGLVKPSETLSLTCTVSGGSVNTGSYYWSWIRQPPGKGLEWIAYSSVSGTSNYNPSLKSRVTLTVDTSKNQFSLSVRSVTAADTAVYFCARLNYDILTGYYFFDFWGQGTLVIVSSASTKGPSVFPLAPSSKSASGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSSGTQTYICNVNHKPSNTKVDKRVEPKSCDKT:QVELTQSPSASASLGTSVKLTCTLSSGHSTYAIAWHQQRPGKGPRYLMNLSSGGRHTRGDGIPDRFSGSSSGADRYLIISSLQSEDEADYYCQTWDAGMVFGGGTKLTVLGQSKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS',
    #            10)


