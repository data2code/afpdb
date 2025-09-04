from anarci import anarci
import pandas as pd
import numpy as np,re
#from .afpdb import util

class Antibody:

    def __init__(self, sequence, scheme='chothia', chain_type=None, species="human"):
        """species: a sublist of ['human','mouse','rat','rabbit','rhesus','pig','alpaca']
        chain_type can be ["H", "L", "K"], if you just want light chain, use ["L","K"], instead of "L"
        scheme must be one of chothia, imgt, kabat, chothia_consensus
            chothia_consensus is a revised version, with shorter CDRs
            See http://www.bioinf.org.uk/abs/info.html
            Note: see note on the page:
            # This page previously described the Chothia CDRs using a wider defintion (taken from these papers), but I have updated that interpretation to use a consensus.
            The previous set was: CDR-L1:L24-34; CDR-L2:L50-56; CDR-L3:L89-97; CDR-H1:H26-32; CDR-H2:H52-56; CDR-H3:H95-102.
        """
        self.sequence = sequence
        self.scheme = scheme.lower()
        if species is None:
            species = ['human','mouse','rat','rabbit','rhesus','pig','alpaca']
        if type(species) is str:
            species = [species]
        if chain_type is None:
            chain_type = ["H", "L", "K"]
        if type(chain_type) is str:
            chain_type=[chain_type]
        out = self._guess_chain_type(species=species, allow=chain_type)
        if out[0] is not None:
            self.allow = [out[1]]
            self.species = [out[0]]
            self.chain_type = out[1]
            self.hits=out[2]
        else:
            self.allow = chain_type
            self.species = species
            self.chain_type = None
            self.hits=pd.DataFrame([], columns=["id","description","evalue","bitscore","bias","query_start","query_end"])
        self.numbering = self._get_numbering()
        self.cdrs = self.get_cdrs()
        # The start and end position of the variable domain, the rest are constant domain or linker
        if self.chain_type is None:
            self.start, self.end = 0, len(self.sequence)-1
        else:
            pos=self.numbering.pos.tolist()
            self.start, self.end = pos[0], pos[-1]

    def __str__(self):
        out=f"scheme={self.scheme}, species={self.species}, chain={self.chain_type}"
        out+="\n"+str(self.get_cdrs())
        out+="\nOriginal: "+self.sequence
        out+="\nAnnotated: "+self.seq()
        return out

    def seq(self):
        if self.chain_type is None:
            return self.sequence.lower()
        seq=np.array(list(self.sequence[self.start : self.end+1].lower()))
        for (b,e,s) in self.cdrs:
            if b is not None:
                for x in range(b, e+1):
                    seq[x]=seq[x].upper()
        return "".join(seq)

    def first_scheme(self):
        """When scheme is a combination, e.g., "chothia,imgt", we extract the first
        scheme to pass it to anarci call.
        Scheme does not affect numbering, it only affects the CDR boundaries.
        """
        return re.split(r'\W+', self.scheme)[0]

    def _guess_chain_type(self, species="human", allow=None):
        """
        Guess whether the sequence is a heavy or light chain based on ANARCI output.
        """
        scheme=self.first_scheme()
        numbering, alignment, hit_tables = anarci([("query", self.sequence)], scheme=scheme, allowed_species=species, allow=allow)
        if numbering[0] is None:
            return (None, None, None)
        # [[{'id': 'human_H', 'description': '', 'evalue': 5.2e-51, 'bitscore': 163.2, 'bias': 0.1, 'query_start': 118, 'query_end': 244, 'species': 'human', 'chain_type': 'H', 'scheme': 'chothia', 'query_name': 'query'}]]
        r = alignment[0][0]
        species, chain_type = r["species"], r["chain_type"]
        # hit_tables actually contains everything, not filtered by allow, we need to filter that
        #print(hit_tables)
        t = pd.DataFrame(hit_tables[0][1:], columns=hit_tables[0][0])
        t['species']=t.id.apply(lambda x: x.split("_")[0])
        t['chain_type']=t.id.apply(lambda x: x.split("_")[1])
        t = t[t.chain_type.isin(allow)].copy()
        return (species, chain_type, t)  # 'H' for heavy, 'K' or 'L' for light

    def _get_numbering(self):
        """
        Get the full ANARCI numbering output.
        """
        scheme=self.first_scheme()
        if re.search(r'\W', self.scheme) is not None:
            print(f"Warning: numbering is done with {scheme} only.")
        numbering, alignment, hit_tables = anarci([("query", self.sequence)], scheme=scheme, allowed_species=self.species, allow=self.allow)
        out=[]
        if numbering[0] is not None:
            pos=numbering[0][0][0]
            cnt=numbering[0][0][1]
            for i,x in enumerate(pos):
                if x[1]=='-': continue # a gap, otherwise pos is wrong for 4G3 VL
                out.append((cnt, int(x[0][0]), str(x[0][1]).strip(), x[1]))
                cnt+=1
        t=pd.DataFrame(out, columns=["pos", "num_int", "num_letter", "residue"])
        return t

    def _get_cdrs(self, scheme):
        """
        Extract CDR regions and their positions based on the numbering scheme.
        """

        #http://www.bioinf.org.uk/abs/info.html
        # see note on the page regarding consensus version of Chothia

        cdr_ranges = {
            'imgt': {
                'H': {'CDR1': (27,"",38,"ZZ"), 'CDR2': (56,"",65,"ZZ"), 'CDR3': (105,"",117,"ZZ")},
                'K': {'CDR1': (27,"",38,"ZZ"), 'CDR2': (56,"",65,"ZZ"), 'CDR3': (105,"",117,"ZZ")},
                'L': {'CDR1': (27,"",38,"ZZ"), 'CDR2': (56,"",65,"ZZ"), 'CDR3': (105,"",117,"ZZ")},
            },
            'kabat': {
                'H': {'CDR1': (31,"",35,"B"), 'CDR2': (50,"",65,"ZZ"), 'CDR3': (95,"",102,"ZZ")},
                'K': {'CDR1': (24,"",34,"ZZ"), 'CDR2': (50,"",56,"ZZ"), 'CDR3': (89,"",97,"ZZ")},
                'L': {'CDR1': (24,"",34,"ZZ"), 'CDR2': (50,"",56,"ZZ"), 'CDR3': (89,"",97,"ZZ")},
            },
            'chothia': {
                'H': {'CDR1': (26,"",32,"ZZ"), 'CDR2': (52,"",56,"ZZ"), 'CDR3': (95,"",102,"ZZ")},
                'K': {'CDR1': (24,"",34,"ZZ"), 'CDR2': (50,"",56,"ZZ"), 'CDR3': (89,"",97,"ZZ")},
                'L': {'CDR1': (24,"",34,"ZZ"), 'CDR2': (50,"",56,"ZZ"), 'CDR3': (89,"",97,"ZZ")},
            },
            'chothia_consensus': {
                'H': {'CDR1': (26,"",32,"ZZ"), 'CDR2': (50,"",52,"ZZ"), 'CDR3': (96,"",101,"ZZ")},
                'K': {'CDR1': (26,"",32,"ZZ"), 'CDR2': (50,"",52,"ZZ"), 'CDR3': (91,"",96,"ZZ")},
                'L': {'CDR1': (26,"",32,"ZZ"), 'CDR2': (50,"",52,"ZZ"), 'CDR3': (91,"",96,"ZZ")},
            },
        }

        chain = self.chain_type
        cdrs = []
        t=self.numbering

        for cdr, (bi,bs,ei,es) in cdr_ranges[scheme][chain].items():
            t2=t[((t.num_int>bi)&(t.num_int<ei))|((t.num_int==bi) & (t.num_letter>=bs))|((t.num_int==ei)&(t.num_letter<=es))]
            #print(bi,bs,ei,es, scheme)
            #t2.display()
            if len(t2)>0:
                b,e=t2.pos.iloc[0], t2.pos.iloc[-1]
                cdrs.append((b,e, "".join(t2.residue)))
            else:
                cdrs.append((None, None, None))
        return cdrs

    def get_cdrs(self):
        chain = self.chain_type
        if chain not in ("H","L","K"):
            return ((None, None, None),)*3

        S_scheme=re.split(r'\W+', self.scheme)
        if len(S_scheme)==1:
            return self._get_cdrs(S_scheme[0])
        out=[]
        for x in S_scheme:
            p=Antibody(self.sequence, scheme=x, chain_type=self.chain_type, species=self.species)
            out.append(p._get_cdrs(x))
        # merge CDRs
        out2=[]
        for i in range(3):
            b=[X[i][0] for X in out if X[i][0] is not None]
            e=[X[i][1] for X in out if X[i][1] is not None]
            if len(b)==0 or len(e)==0:
                out2.append((None, None, None))
            else:
                b, e=min(b), max(e)
                s=self.sequence[b:e+1]
                out2.append((b,e,s))
        return out2

if __name__=="__main__":
    seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARGGYYYAMDYWGQGTLVTVSS"
    ab = Antibody(seq, scheme='chothia', species="human")
    print(ab)

