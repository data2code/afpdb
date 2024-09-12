import os
# from ColabDesign
# https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/rf/examples/diffusion.ipynb#scrollTo=pZQnHLuDCsZm
def get_pdb(pdb_code=None):
  if os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    if not os.path.isfile(f"{pdb_code}.pdb1"):
      cmd=f"wget -nc --no-check-certificate https://files.rcsb.org/download/{pdb_code}.pdb1.gz && gunzip {pdb_code}.pdb1.gz"
      os.system(cmd)
      if not os.path.exists(f"{pdb_code}.pdb1"):
        print(f"Fail to download PDB file: {pdb_code}.pdb1!")
        print(cmd)
        return None
    return f"{pdb_code}.pdb1"
  else:
    os.system(f"wget -nc --no-check-certificate https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
    return f"AF-{pdb_code}-F1-model_v3.pdb"

if __name__=="__main__":
    print(get_pdb("1crn"))

