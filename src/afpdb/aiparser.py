#!/usr/bin/env python
from .afpdb import util
import os,re,glob,json
import numpy as np, pandas as pd,pickle
from Bio import SeqIO
from Bio.Seq import Seq
from .afpdb import Protein

class ESMFoldParser:
    """ESMFold output models in pdb files. To read pLDDT, use:
        p=Protein(pdb_file_name)
        p.b_factors()
    """

    def __init__(self, folder, relaxed=True):
        self.fd=folder
        self.data=self.parse()

    def parse(self):
        S=glob.glob(f"{self.fd}/*.pdb")
        out=[]
        for fn in S:
            out.append({'path':fn, 'name':re.sub(r'\.pdb$', '', os.path.basename(fn))})
        t=pd.DataFrame(out)
        #test_130a1/ptm0.374_r3_default.pae.txt
        S=glob.glob(f"{self.fd}/*.pae.txt")
        out=[]
        for fn in S:
            out.append({'pae_path':fn, 'name':re.sub(r'\.pae\.txt$', '', os.path.basename(fn))})
        if len(out):
            t2=pd.DataFrame(out)
            t=t.merge(t2, on=['name'], how='left')
        return t

    def get_pdb(self, idx=0):
        if len(self.data):
            return self.data.loc[idx, 'path']
        return None

    def get_pae(self, idx=0):
        ## the following was tested with https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb
        if len(self.data) and 'pae_path' in self.data.header():
            fn=self.data.loc[idx, 'pae_path']
            t=pd.read_table(fn, sep=' ', header=None)
            return t.values
        return None

    def get_plddt(self, idx=0):
        fn=self.get_pdb(idx=idx)
        return Protein(fn).b_factors()

class ColabFoldParser:

    def __init__(self, folder):
        self.fd=folder
        self.data=self.parse()

    def parse(self, relaxed=True):
        S=glob.glob(f"{self.fd}/*.pdb")
        out=[]
        for fn in S:
            one={'path':fn, 'name':re.sub(r'\.pdb$', '', os.path.basename(fn))}
            one['relaxed']='_relaxed_' in fn
            if not relaxed and one['relaxed']: continue
            m = re.search(r'_model_(?P<model>\d+)', fn)
            if m is None: continue
            one['model']=int(m.group('model'))
            m = re.search(r'_rank_(?P<rank>\d+)', fn)
            one['ranking']=int(m.group('rank')) if m is not None else -1
            m = re.search(r'_seed_(?P<seed>\d+)', fn)
            one['seed']=int(m.group('seed')) if m is not None else -1
            out.append(one)
        t=pd.DataFrame(out)

        S=glob.glob(f"{self.fd}/*.json")
        out=[]
        for fn in S:
            one={'json_path':fn}
            m = re.search(r'_model_(?P<model>\d+)', fn)
            if m is None: continue
            one['model']=int(m.group('model'))
            m = re.search(r'_seed_(?P<seed>\d+)', fn)
            one['seed']=int(m.group('seed')) if m is not None else -1
            x=json.loads(util.read_string(fn))
            one['ptm']=x.get('ptm', None)
            one['iptm']=x.get('iptm', None)
            out.append(one)
        if len(out):
            t2=pd.DataFrame(out)
            t=t.merge(t2, on=['model','seed'], how='left')
        t.sort_values(['relaxed','ranking'], ascending=[False, True], inplace=True)
        t.index=range(len(t))
        return t

    def get_pdb(self, idx=0):
        if len(self.data):
            return self.data.loc[idx, 'path']
        return None

    def get_ptm(self, idx=0):
        if len(self.data):
            return self.data.loc[idx, 'ptm']
        return None

    def get_iptm(self, idx=0):
        if len(self.data):
            return self.data.loc[idx, 'iptm']
        return None

    def get_pae(self, idx=0):
        if len(self.data) and 'json_path' in self.data.header():
            fn=self.data.loc[idx, 'json_path']
            data=np.array(json.loads(util.read_string(fn)).get('pae'), None)
            return data
        return None

    def get_plddt(self, idx=0):
        if len(self.data) and 'json_path' in self.data.header():
            fn=self.data.loc[idx, 'json_path']
            data=np.array(json.loads(util.read_string(fn)).get('plddt'), None)
            return data
        return None

#https://stackoverflow.com/questions/46857615/how-to-replace-objects-causing-import-errors-with-none-during-pickle-load
#https://github.com/google-deepmind/alphafold/issues/629
class Dummy:

    def __init__(*args):
        pass

class MyUnpickler(pickle._Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception as e:
            return Dummy

class AlphaFoldParser(ColabFoldParser):

    def parse(self, relaxed=True):
        S=glob.glob(f"{self.fd}/*.pdb")
        out=[]
        for fn in S:
            #relaxed_model_5_multimer_v3_pred_1.pdb
            one={'path':fn, 'name':re.sub(r'\.pdb$', '', os.path.basename(fn))}
            one['relaxed']=one['name'].startswith('relaxed_')
            if not relaxed and one['relaxed']: continue
            m = re.search(r'_model_(?P<model>\d+)', fn)
            if m is None: continue
            one['model']=int(m.group('model'))
            m = re.search(r'_pred_(?P<seed>\d+)', fn)
            one['seed']=int(m.group('seed')) if m is not None else -1
            out.append(one)
        t=pd.DataFrame(out)

        S=glob.glob(f"{self.fd}/*.pkl")
        out=[]
        for fn in S:
            #result_model_1_multimer_v3_pred_2.pkl
            if not os.path.basename(fn).startswith('result_'): continue
            one={'pkl_path':fn}
            m = re.search(r'_model_(?P<model>\d+)', fn)
            if m is None: continue
            one['model']=int(m.group('model'))
            m = re.search(r'_pred_(?P<seed>\d+)', fn)
            one['seed']=int(m.group('seed')) if m is not None else -1
            with open(fn, 'rb') as f:
                x=MyUnpickler(f).load()
                one['ptm']=x.get('ptm').item()
                one['iptm']=x.get('iptm').item()
            out.append(one)
        t2=pd.DataFrame(out)
        t=t.merge(t2, on=['model','seed'], how='left')
        t.sort_values(['relaxed','iptm','ptm'], ascending=[False, False, False], inplace=True)
        t.index=range(len(t))
        return t

    def get_pae(self, idx=0):
        if len(self.data) and 'pkl_path' in self.data.header():
            fn=self.data.loc[idx, 'pkl_path']
            with open(fn, 'rb') as f:
                x=MyUnpickler(f).load()
                pae=x.get('predicted_aligned_error', None)
            if pae is not None: return np.array(pae)
        return None

    def get_plddt(self, idx=0):
        if len(self.data) and 'pkl_path' in self.data.header():
            fn=self.data.loc[idx, 'pkl_path']
            with open(fn, 'rb') as f:
                x=MyUnpickler(f).load()
                plddt=x.get('plddt', None)
            if plddt is not None: return np.array(plddt)
        return None

class ProteinMPNNParser:

    def __init__(self, folder):
        self.fd=folder
        self.data=self.parse()

    def parse(self):
        S=glob.glob(f"{self.fd}/seqs/*.fa")
        out=[]
        for fn in S:
            #>1crn, score=1.7228, global_score=1.7573, fixed_chains=['E'], designed_chains=['A'], model_name=v_48_020, git_hash=8907e6671bfbfc92303b5f79c4b5e6ce47cdef57, seed=931
            #TTCCPSIXXRSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDY
            #>T=0.1, sample=1, score=1.1982, global_score=1.2612, seq_recovery=0.5000
            name=os.path.basename(fn)
            for r in SeqIO.parse(fn, "fasta"):
                x={s.split("=")[0]:s.split("=")[1] for s in re.split(r',\s*', r.description) if '=' in s}
                one={'path':fn, 'name':name}
                if 'sample' not in x: continue
                for k in ['sample','score','global_score','T','seq_recovery']:
                    one[k]=x[k]
                one['seq']=str(r.seq)
                out.append(one)
        t=pd.DataFrame(out)
        t.sort_values('score', ascending=True, inplace=True)
        t.index=range(len(t))
        return t

    def make_structure(self, pdb, output_folder, side_chain_pdb=None, rl_from=None, rl_to=None):
        p=Protein(pdb)
        os.makedirs(output_folder, exist_ok=True)
        pg=util.Progress(len(self.data))
        for i,r in self.data.iterrows():
            seq=r['seq']
            p.thread_sequence(seq, f"{output_folder}/sample{r['sample']}.pdb", relax=0, seq2bfactor=False, side_chain_pdb=side_chain_pdb, rl_from=rl_from, rl_to=rl_to)
            pg.check(i+1)

class LigandMPNNParser:

    def __init__(self, folder):
        self.fd=folder
        self.data=self.parse()

    def parse(self):
        S=glob.glob(f"{self.fd}/seqs/*.fa")
        out=[]
        for fn in S:
            #>1BC8, T=0.1, seed=111, num_res=93, num_ligand_res=93, use_ligand_context=True, ligand_cutoff_distance=8.0, batch_size=1, number_of_batches=1, model_path=./model_params/proteinmpnn_v_48_020.pt
            #>1BC8, id=1, T=0.1, seed=111, overall_confidence=0.3848, ligand_confidence=0.3848, seq_rec=0.4946
            name=os.path.basename(fn)
            for r in SeqIO.parse(fn, "fasta"):
                x={s.split("=")[0]:s.split("=")[1] for s in re.split(r',\s*', r.description) if '=' in s}
                one={'path':fn, 'name':name}
                if 'id' not in x: continue
                #>1BC8, id=1, T=0.1, seed=111, overall_confidence=0.3848, ligand_confidence=0.3848, seq_rec=0.4946
                for k in ['id','T','seed','overall_confidence','ligand_confidence','seq_rec']:
                    one[k]=x[k]
                one['seq']=str(r.seq)
                out.append(one)
        t=pd.DataFrame(out)
        t.sort_values(['overall_confidence','ligand_confidence'], ascending=[False, False], inplace=True)
        t.index=range(len(t))
        return t

