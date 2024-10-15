import os
# from ColabDesign
# https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/rf/examples/diffusion.ipynb#scrollTo=pZQnHLuDCsZm
def get_pdb(pdb_code=None, assembly1=False):
  if os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    #fn=f"{pdb_code}.pdb1" if not assembly1 else f"{pdb_code}-assembly1.cif"
    fn=f"{pdb_code}.cif" if not assembly1 else f"{pdb_code}-assembly1.cif"
    if not os.path.isfile(fn):
      cmd=f"wget -nc --no-check-certificate https://files.rcsb.org/download/{fn}.gz && gunzip {fn}.gz"
      os.system(cmd)
      if not os.path.exists(fn):
        print(f"Fail to download PDB file: {pdb_code}.pdb1!")
        print(cmd)
        return None
    return fn
  else:
    os.system(f"wget -nc --no-check-certificate https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
    return f"AF-{pdb_code}-F1-model_v3.pdb"

if __name__=="__main__":
    print(get_pdb("1crn"))

