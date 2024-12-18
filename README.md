# Afpdb - An Efficient Protein Structure Manipulation Tool

<a href="https://pypi.org/project/afpdb" rel="nofollow">
<img alt="PyPI - Version" src="https://img.shields.io/pypi/v/afpdb?logo=pypi">
</a>
<a href="https://anaconda.org/bioconda/afpdb" rel="nofollow">
<img alt="Conda Version" src="https://img.shields.io/conda/vn/bioconda/afpdb">
</a>

The advent of AlphaFold and other protein AI models has transformed protein design, necessitating efficient handling of large-scale data and complex workflows. Traditional programming packages, developed before these AI advancements, often lead to inefficiencies in coding and slow execution. To bridge this gap, we introduce Afpdb, a high-performance Python module built on AlphaFold’s NumPy architecture. Afpdb leverages RFDiffusion's contig syntax to streamline residue and atom selection, making coding simpler and more readable. By integrating PyMOL’s visualization capabilities, Afpdb enables automatic visual quality control, enhancing productivity in structural biology. With over 180 methods commonly used in protein AI design, Afpdb supports the development of concise, high-performance code, addressing the limitations of existing tools like Biopython. Afpdb is designed to complement powerful AI models such as AlphaFold and ProteinMPNN, providing the additional utility needed to effectively manipulate protein structures and drive innovation in protein design.

Please read our short artical published in <a href="https://doi.org/10.1093/bioinformatics/btae654">Bioinformatics</a>.

<img src="https://github.com/data2code/afpdb/blob/main/tutorial/img/afpdb.png?raw=true">

## Tutorial

The tutorial book is availabe in <a href="tutorial/Afpdb_Tutorial.pdf">PDF</a>.

The best way to learn and practice Afpdb is to open [Tutorial Notebook](https://colab.research.google.com/github/data2code/afpdb/blob/main/tutorial/afpdb.ipynb) in Google Colab.

Table of Content

1. Demo
2. Fundamental Concepts
   - Internal Data Structure
   - Contig 
3. Selection
   - Atom Selection
   - Residue Selection
   - Residue List
4. Read/Write
5. Sequence & Chain
6. Geometry, Measurement, & Visualization
   - Select Neighboring Residues
   - Display
   - B-factors
   - PyMOL Interface
   - RMSD
   - Solvent-Accessible Surface Area (SASA)
   - Secondary Structures - DSSP
   - Internal Coordinates
7. Object Manipulation
   - Move Objects
   - Align
   - Split & Merge Objects
8. Parsers for AI Models

## AI Use Cases

Interested in applying Afpdb to AI protein design? Open [AI Use Case Notebook](https://colab.research.google.com/github/data2code/afpdb/blob/main/tutorial/AI.ipynb) in Google Colab.

Table of Content

- Example AI Protein Design Use Cases
   - Handle Missing Residues in AlphaFold Prediction
   - Structure Prediction with ESMFold
   - Create Side Chains for de novo Designed Proteins
   - Compute Binding Scores in EvoPro

## Developer's Note

Open [Developer Notebook](https://colab.research.google.com/github/data2code/afpdb/blob/main/tutorial/Developer.ipynb) in Google Colab.

## Install
Stable version:
```
pip install afpdb
```
or
```
conda install bioconda::afpdb
```
Development version:
```
pip install git+https://github.com/data2code/afpdb.git
```
or
```
git clone https://github.com/data2code/afpdb.git
cd afpdb
pip install .
```
To import the package use:
```
from afpdb.afpdb import Protein,RS,RL,ATS
```
## Demo

### Structure Read & Summary
```
# load the ab-ag complex structure 5CIL from PDB
p=Protein("5cil")
# show key statistics summary of the structure
p.summary().display()
```
Output
```
    Chain    Sequence                    Length    #Missing Residues    #Insertion Code    First Residue Name    Last Residue Name
--  -------  ---------------------------------------------------------------------------------------------------------------------
 0  H        VQLVQSGAEVKRPGSSVTVS...        220                   20                 14                     2                  227
 1  L        EIVLTQSPGTQSLSPGERAT...        212                    0                  1                     1                  211
 2  P        NWFDITNWLWYIK                   13                    0                  0                   671                  683
```
### Residue Relabeling

```
print("Old P chain residue numbering:", p.rs("P").name(), "\n")

Output:
Old P chain residue numbering: ['671', '672', '673', '674', '675', '676', '677', '678', '679', '680', '681', '682', '683'] 

p.renumber("RESTART", inplace=True)
print("New P chain residue numbering:", p.rs("P").name(), "\n")

Output:
New P chain residue numbering: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'] 

p.summary()
```
Output

```
    Chain    Sequence                    Length    #Missing Residues    #Insertion Code    First Residue Name    Last Residue Name
--  -------  ---------------------------------------------------------------------------------------------------------------------
 0  H        VQLVQSGAEVKRPGSSVTVS...        220                   20                 14                     1                  226
 1  L        EIVLTQSPGTQSLSPGERAT...        212                    0                  1                     1                  211
 2  P        NWFDITNWLWYIK                   13                    0                  0                     1                   13
 ```
### Replace Missing Residues for AI Prediction
```
print("Sequence for AlphaFold modeling, with missing residues replaced by Glycine:")
print(">5cil\n"+p.seq(gap="G")+"\n")
```
Output
```
Sequence for AlphaFold modeling, with missing residues replaced by Glycine:
>5cil
VQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGDGDLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSGGGGGGGGGTAALGCLVKDYFPEPVTVGSWGGGGNSGALTSGGVHTFPAVLQSGSGLYSLSSVVTVPSSSLGTGGQGTYICNVNHKPSNTKVDKKGGVEP:EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNR:NWFDITNWLWYIK
```
### Interface Computing
```
# identify H,L chain residues within 4A to antigen P chain
rs_binder, rs_seed, df_dist=p.rs_around("P", dist=4)

# show the distance of binder residues to antigen P chain
df_dist[:5].display()
```
Output
```
     chain_a      resn_a    resn_i_a    resi_a  res_a    chain_b    resn_b      resn_i_b    resi_b  res_b       dist  atom_a    atom_b
---  ---------  --------  ----------  --------  -------  ---------  --------  ----------  --------  -------  -------  --------  --------
408  P                 6           6       437  T        H          94                94        97  E        2.63625  OG1       OE2
640  P                 4           4       435  D        L          32                32       252  K        2.81482  OD1       NZ
807  P                 2           2       433  W        L          94                94       314  S        2.91194  N         OG
767  P                 1           1       432  N        L          91                91       311  Y        2.9295   ND2       O
526  P                 7           7       438  N        H          99E               99       107  K        3.03857  ND2       CE
```
### Residue Selection & Boolean Operations
```
# create a new PDB file only containing the antigen and binder residues
p=p.extract(rs_binder | "P")
```
### Structure I/O
```
# save the new structure into a local PDB file
p.save("binders.pdb")
```
### Structure Display within Jupyter Notebook
```
# display the PDB struture, default is show ribbon and color by chains.
p.show(show_sidechains=True)
```
Output (It will be 3D interactive within Jupyter Notebook)<br>
<img src="https://github.com/data2code/afpdb/blob/main/tutorial/img/demo.png?raw=true">
### PyMOL Integration
```
# convert the selection into a PyMOL selection command
rs = (rs_binder | P).str(format="PYMOL", rs_name="myint")
cmd=f'''fetch 5cil, myobj; util.cbc;
{rs}
show sticks, myint; zoom myint; deselect;
save binders.pse
# generate a PyMOL session file binders.pse
Protein.PyMOL().run(cmd)
```
