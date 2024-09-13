#!/usr/bin/env python
import pandas as pd,os,json,glob,shutil
from afpdb import util
from maps import map

OUTPUT="output"

def timeit(cmd, fn_out):
    cmd=f"mprof run --output {fn_out} ./ram.py {cmd}"
    if os.path.exists(fn_out): os.remove(fn_out)
    util.unix(cmd, l_print=False)
    S=util.read_list(fn_out)
    cost=max([float(x.split(" ")[1]) for x in S if x.startswith('MEM')])
    #os.remove(fn)
    return cost

def mem_usage(s_pdb):
    fn_out=OUTPUT+"/"+s_pdb+".dat"
    s_pdb=s_pdb+".pdb"
    af_pdb=f"AFDB/{s_pdb}"
    exp_pdb=f"AbDb/{s_pdb}"

    one={'pdb': s_pdb}

    cmd=f"./ram.py afpdb_lib {af_pdb}"
    one['afpdb_lib']=float(util.unix(cmd, l_print=False))
    cmd=f"./ram.py biopython_lib {af_pdb}"
    one['biopython_lib']=float(util.unix(cmd, l_print=False))

    cmd=f"./ram.py afpdb_io {af_pdb} {exp_pdb}"
    one['afpdb_io']=float(util.unix(cmd, l_print=False))
    cmd=f"./ram.py biopython_io {af_pdb} {exp_pdb}"
    one['biopython_io']=float(util.unix(cmd, l_print=False))

    cmd=f"afpdb_int {af_pdb}"
    one['afpdb_int']=timeit(cmd, fn_out)-one['afpdb_lib']
    cmd=f"biopython_int {af_pdb}"
    one['biopython_int']=timeit(cmd, fn_out)-one['biopython_lib']

    cmd=f"afpdb_align {af_pdb} {exp_pdb}"
    one['afpdb_align']=timeit(cmd, fn_out)-one['afpdb_lib']
    cmd=f"biopython_align {af_pdb} {exp_pdb}"
    one['biopython_align']=timeit(cmd, fn_out)-one['biopython_lib']

    cmd=f"afpdb_rmsd {af_pdb} {exp_pdb}"
    one['afpdb_rmsd']=timeit(cmd, fn_out)-one['afpdb_lib']
    cmd=f"biopython_rmsd {af_pdb} {exp_pdb}"
    one['biopython_rmsd']=timeit(cmd, fn_out)-one['biopython_lib']

    cmd=f"afpdb_b {af_pdb}"
    one['afpdb_b_factor']=timeit(cmd, fn_out)-one['afpdb_lib']
    cmd=f"biopython_b {af_pdb}"
    one['biopython_b_factor']=timeit(cmd, fn_out)-one['biopython_lib']
    return one

if __name__ == "__main__":
    if not os.path.exists('AFDB') or not os.path.exists('AbDb'):
        util.unix('tar xvfz pdb.tar.gz')

    out=util.unix("mprof -h |grep Available |wc -l", l_print=False)
    if out.strip()!="1":
        print("Please install Memory Profiler as described on:")
        print("   https://github.com/pythonprofilers/memory_profiler/blob/master/README.rst")
        exit()

    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    else:
        os.makedirs(OUTPUT, exist_ok=True)

    t=pd.read_csv('benchmark.csv')
    tasks = t.pdb.tolist()
    out=map(mem_usage, tasks, n_CPU=30)
    t=pd.DataFrame(out)
    util.df2sdf(t, s_format="%.4g").to_csv("afpdb_mem.csv", index=False)
    shutil.rmtree(OUTPUT)

