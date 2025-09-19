import os
import requests
import gzip
import shutil

def _download_file(url, out_fn):
    """Downloads a file from a URL, decompressing if it's a .gz file."""
    is_gz = url.endswith(".gz")
    download_fn = out_fn + ".gz" if is_gz else out_fn

    if os.path.exists(out_fn):
        return out_fn

    try:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        with open(download_fn, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if is_gz:
            with gzip.open(download_fn, 'rb') as f_in:
                with open(out_fn, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(download_fn)
        
        return out_fn

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(download_fn):
            os.remove(download_fn)
        return None

# from ColabDesign
# https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/rf/examples/diffusion.ipynb#scrollTo=pZQnHLuDCsZm
def get_pdb(pdb_code=None, assembly1=False):
  if os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    #fn=f"{pdb_code}.pdb1" if not assembly1 else f"{pdb_code}-assembly1.cif"
    fn=f"{pdb_code}.cif" if not assembly1 else f"{pdb_code}-assembly1.cif"
    if not os.path.isfile(fn):
      url = f"https://files.rcsb.org/download/{fn}.gz"
      if _download_file(url, fn) is None:
        print(f"Fail to download PDB file: {pdb_code}!")
        print(f"From URL: {url}")
        return None
    return fn
  else:
    out_fn = f"AF-{pdb_code}-F1-model_v3.pdb"
    url = f"https://alphafold.ebi.ac.uk/files/{out_fn}"
    return _download_file(url, out_fn)

if __name__=="__main__":
    print(get_pdb("1crn"))

