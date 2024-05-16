#!/usr/bin/env python
#@title Display 3D structure {run: "auto"}
import py3Dmol
import re
pymol_color_list = ["#33ff33","#00ffff","#ff33cc","#ffff00","#ff9999","#e5e5e5","#7f7fff","#ff7f00",
                    "#7fff7f","#199999","#ff007f","#ffdd5e","#8c3f99","#b2b2b2","#007fff","#c4b200",
                    "#8cb266","#00bfbf","#b27f7f","#fcd1a5","#ff7f7f","#ffbfdd","#7fffff","#ffff7f",
                    "#00ff7f","#337fcc","#d8337f","#bfff3f","#ff7fff","#d8d8ff","#3fffbf","#b78c4c",
                    "#339933","#66b2b2","#ba8c84","#84bf00","#b24c66","#7f7f7f","#3f3fa5","#a5512b"]

#pymol_cmap = matplotlib.colors.ListedColormap(pymol_color_list)
from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

def my_style(self, color, style, chains=1, model_id=0):
  if color=="lDDT":
    self.setStyle({'model': model_id}, {style: {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
  elif color=='b':
    self.setStyle({'model': model_id}, {style: {'colorscheme': {'prop':'b','gradient': 'roygb','min':0,'max':1}}})
  elif color in ("rainbow","spectrum"):
    self.setStyle({'model': model_id}, {style: {'color':'spectrum'}})
  elif color == "chain":
    #chains = len(queries[0][1]) + 1 if is_complex else 1
    for n,chain,color in zip(range(chains),alphabet_list,pymol_color_list):
       #print({'model': model_id}, {'chain':chain},{style: {'color':color}})
       # color by chain is not compatible with model_id, see
       # https://github.com/3dmol/3Dmol.js/issues/671
       #self.setStyle({'model': model_id}, {'chain':chain},{style: {'color':color}})
       self.setStyle({'model': model_id,'chain':chain},{style: {'color':color}})
  elif color == "ss":
    #https://github.com/3dmol/3Dmol.js/issues/668
    self.setStyle({'model': model_id}, {style: {'colorscheme':{'prop':'ss','map':"@REPLACE@$3Dmol.ssColors.Jmol@REPLACE@"}}})
  else:
    self.setStyle({'model': model_id}, {style: {'color':color}})

def my_html(self):
  self.zoomTo()
  out=self._make_html()
  out=re.sub(r'"?@REPLACE@"?', '', out)
  return out

py3Dmol.view.my_style=my_style
py3Dmol.view.my_html=my_html

class Mol3D:

    URL='https://3dmol.org/build/3Dmol.js'

    def add_model(self, pdb_file):
        if pdb_file is None or pdb_file=="_model_":
            # multiple models, model already preloaded
            self.chains=1
        else:
            from afpdb.afpdb import Protein
            p=Protein(pdb_file)
            # rename chain to A,B,C, etc
            p.data_prt.chain_id[:]=alphabet_list[:len(p.chain_id())]
            self.chains=len(p.chain_id())
            p._renumber('RESTART')
            data=p.to_pdb_str()
            self.view.addModel(data,'pdb')

    def show(self, pdb_file=None, show_sidechains=False, show_mainchains=False, color="lDDT", style="cartoon", width=480, height=480, model_id=0):
        """
        color: lDDT/b, spectrum/rainbow, chain, ss
        style: cartoon, stick, line, sphere, cross
        Not all combinations work, see https://3dmol.org/viewer.html?pdb=4KW4&select=all&style=cartoon:color~spectrum;stick:radius~0.25,colorscheme~greenCarbon&select=bonds:0&style=sphere:radius~0.5
        """
        self.view = py3Dmol.view(js=Mol3D.URL, width=width, height=height)
        self.add_model(pdb_file)
        self.view.my_style(color, style, model_id=model_id, chains=self.chains)

        if show_sidechains:
            BB = ['C','O','N']
            self.view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                                {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
            self.view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                                {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
            self.view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                                {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        if show_mainchains:
            BB = ['C','O','N','CA']
            self.view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

        return self.view.my_html()

    def cartoon_b(self, pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="lDDT", style="cartoon", width=width, height=height, model_id=model_id)

    def cartoon_spectrum(self, pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="spectrum", style="cartoon", width=width, height=height, model_id=model_id)

    def cartoon_chain(self, pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="chain", style="cartoon", width=width, height=height, model_id=model_id)

    def cartoon_ss(self, pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="ss", style="cartoon", width=width, height=height, model_id=model_id)

    def stick_b(self, pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="lDDT", style="stick", width=width, height=height, model_id=model_id)

    def stick_chain(self, pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="chain", style="stick", width=width, height=height, model_id=model_id)

    #spectrum does not supported for stick
    #def show_stick_spectrum(pdb_file, width=480, height=480):
    #  return show_pdb(pdb_file, color="spectrum", style="stick", width=width, height=height)

    def stick_ss(self, pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="ss", style="stick", width=width, height=height, model_id=model_id)

    # currently color by chain only works if chains are named as A, B, C ...
    # otherwise, you need to rename chains in pymol first
    def show_stick_chain(pdb_file=None, width=480, height=480, model_id=0):
        return self.show(pdb_file, color="chain", style="stick", width=width, height=height, model_id=model_id)

    def show_many(self, S_pdb, S_style, S_color, S_chains=None, width=480, height=480):
        self.view = py3Dmol.view(width=width, height=height)
        if S_chains is None: S_chains=1
        if type(S_pdb) is str: S_pdb=[S_pdb]
        n=len(S_pdb)
        if type(S_style) not in (list, tuple): S_style=[S_style]*n
        if type(S_color) not in (list, tuple): S_color=[S_color]*n
        if type(S_chains) not in (list, tuple): S_chains=[S_chains]*n
        model_id=0
        for pdb,style,color,chains in zip(S_pdb, S_style, S_color, S_chains):
            self.view.addModel(open(pdb,'r').read(), 'pdb')
            self.view.my_style(model_id=model_id, color=color, style=style, chains=self.chains)
            model_id+=1
        return self.view.my_html()

if __name__=="__main__":
    x=Mol3D()
    out=["<table><tr><td>"]
    out.append(x.cartoon_b("6bgn.pdb"))
    out.append("</td><td>")
    out.append(x.cartoon_spectrum("6bgn.pdb"))
    out.append("</td><td>")
    out.append(x.cartoon_ss("6bgn.pdb"))
    out.append("</td><td>")
    out.append(x.cartoon_chain("7x95.pdb"))
    out.append("</td></tr><tr><td>")
    out.append(x.stick_b("6bgn.pdb"))
    out.append("</td><td>")
    out.append(x.show_many(["1crn.pdb","6bgn.pdb"], ["cartoon", "cartoon"], ["#bcbddc", "ss"]))
    out.append("</td><td>")
    out.append(x.stick_ss("6bgn.pdb"))
    out.append("</td><td>")
    out.append(x.stick_chain("7x95.pdb"))
    out.append("</td></tr></table>")
    print("\n".join(out))
