from .afpdb import Protein,ATS,RS
import numpy as np
import warnings

# ipSAE
# script for calculating the ipSAE score for scoring pairwise protein-protein interactions in AlphaFold2 and AlphaFold3 models
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

# pDockQ: Bryant, Pozotti, and Eloffson. https://www.nature.com/articles/s41467-022-28865-w
# pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
# LIS: Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

# Afpdb metrics
# PAE_int: PAE-based score for interface residues, normalized to [0, 1]. 1 means perfect.
# pLDDT_int: pLDDT-based score for interface residues, normalized to [0, 1]. 1 means perfect.

class StructureMetrics:

    def __init__(self, pdb, plddt = None, pae = None):
        """
        Initialize with a predicted structure (PDB file or string), pLDDT array, and pAE array.
        """
        self.protein = Protein(pdb)
        self.plddt = plddt
        self.pae = pae
        self.chains = self.protein.chain_id()
        self.results = {}

    def compute_pae_int_pair(self, chain1, chain2, cutoff=5.0):
        """
        Compute average PAE score for interface residues between chain1 and chain2.
        Interface residues are identified using rs_around.
        Normalized to [0, 1], where 1 means accurate prediction
        """
        if self.pae is None:
            return np.nan
        rs2, rs1, dist = self.protein.rs_around(chain1, dist=cutoff, rs_within=chain2)
        if len(rs1) == 0 or len(rs2) == 0:
            return np.nan
        pae_sub = [(self.pae[i, j] + self.pae[j, i]) / 2 for i, j in zip(dist.resi_a, dist.resi_b)]
        return 1-np.mean(pae_sub)/35.0

    def compute_pae_int(self, cutoff=5.0):
        """
        Compute average PAE score for all chain pairs (interface residues).
        Returns a dict of results.
        """
        results = {}
        chains = self.protein.chain_id()
        for chain1 in chains:
            for chain2 in chains:
                if chain1 <= chain2:
                    continue
                results[(chain1, chain2)] = self.compute_pae_int_pair(chain1, chain2, cutoff=cutoff)
                results[(chain2, chain1)] = results[(chain1, chain2)]
        return results

    def compute_plddt_int_pair(self, chain1, chain2, cutoff=5.0):
        """
        Compute average pLDDT score for interface residues between chain1 and chain2.
        Interface residues are identified using rs_around.
        Normalized to [0, 1]
        """
        if self.plddt is None:
            return np.nan
        rs2, rs1, _ = self.protein.rs_around(chain1, dist=cutoff, rs_within=chain2)
        int_idx1 = rs1.data
        int_idx2 = rs2.data
        all_int_idx = np.unique(np.concatenate([int_idx1, int_idx2]))
        if len(all_int_idx) == 0:
            return np.nan
        return np.mean(self.plddt[all_int_idx])/100

    def compute_plddt_int(self, cutoff=5.0):
        """
        Compute average pLDDT score for all chain pairs (interface residues).
        Returns a dict of results.
        """
        results = {}
        chains = self.protein.chain_id()
        for chain1 in chains:
            for chain2 in chains:
                if chain1 <= chain2:
                    continue
                results[(chain1, chain2)] = self.compute_plddt_int_pair(chain1, chain2, cutoff=cutoff)
                results[(chain2, chain1)] = results[(chain1, chain2)]
        return results

    def compute_iptm_pair(self, chain1, chain2):
        """
        Compute PTM (Predicted TM-score) for a given chain pair.
        """
        if self.pae is None:
            return np.nan
        _, score = self.ptm_matrix_pair(chain1, chain2, L_mode="all")
        return score

    def compute_iptm(self):
        """
        Compute PTM (Predicted TM-score) for all chain pairs and overall complex.
        Returns a dict of results.
        """
        results = {}
        chains = self.protein.chain_id()
        for chain1 in chains:
            for chain2 in chains:
                if chain1 == chain2:
                    continue
                results[(chain1, chain2)] = self.compute_iptm_pair(chain1, chain2)
        # compute the overall iPTM
        for chain1 in chains:
            chain2 = ":".join([chain for chain in chains if chain != chain1])
            results[chain1] = self.ptm_matrix_pair(chain1, chain2, L_mode="all")[1]
        results['complex'] = max([results[chain] for chain in chains])
        return results

    def compute_ipsae_pair(self, chain1, chain2, pae_cutoff=10.0):
        """
        Compute ipSAE (interface predicted aligned error) for a given chain pair.
        """
        if self.pae is None:
            return np.nan
        _, score = self.ptm_matrix_pair(chain1, chain2, pae_cutoff=pae_cutoff, L_mode="valid")
        return score

    def compute_ipsae(self, pae_cutoff=10.0):
        """
        Compute ipSAE (interface predicted aligned error) for all chain pairs and overall complex.
        Returns a dict of results.
        """
        results = {}
        chains = self.protein.chain_id()
        for chain1 in chains:
            for chain2 in chains:
                if chain1 == chain2:
                    continue
                results[(chain1, chain2)] = self.compute_ipsae_pair(chain1, chain2, pae_cutoff=pae_cutoff)
        results['complex'] = np.mean(list(results.values())) if results else 0.0
        return results

    def compute_pdockq_pair(self, chain1, chain2, dist_cutoff=8.0):
        """
        Compute pDockQ and pDockQ2 for a given chain pair.
        Returns (pDockQ, pDockQ2)
        """
        if self.plddt is None:
            return np.nan, np.nan
        idx1 = self.protein.rs(chain1).data
        coordinates1 = self.protein.atom_coordinate(chain1, "CB")
        idx2 = self.protein.rs(chain2).data
        coordinates2 = self.protein.atom_coordinate(chain2, "CB")
        cb_plddt = self.plddt
        # Compute pairwise distances
        dist_matrix = np.linalg.norm(coordinates1[:, None] - coordinates2[None, :], axis=2)
        valid_pairs = dist_matrix <= dist_cutoff
        npairs = np.sum(valid_pairs)
        if npairs > 0:
            valid1 = np.any(valid_pairs, axis=1)
            valid2 = np.any(valid_pairs, axis=0)
            idx = np.concatenate((idx1[valid1], idx2[valid2]))
            mean_plddt = cb_plddt[idx].mean()
            x1 = mean_plddt * np.log10(npairs)
            score1 = 0.724 / (1 + np.exp(-0.052 * (x1 - 152.611))) + 0.018

            if self.pae is None:
                return score1, np.nan
            pae_sub = self.pae[np.ix_(idx1, idx2)]
            ptm = self.ptm_func(pae_sub, 10.0)
            mean_ptm = np.nanmean(ptm[valid_pairs])
            x2 = mean_plddt * mean_ptm
            score2 = 1.31 / (1 + np.exp(-0.075 * (x2 - 84.733))) + 0.005
        else:
            score1 = score2 = 0.0
        return score1, score2

    def compute_pdockq(self, dist_cutoff=8.0):
        """
        Compute pDockQ and pDockQ2 for all chain pairs and overall complex.
        Returns two dicts: pDockQ and pDockQ2.
        """
        results1 = {}
        results2 = {}
        chains = self.protein.chain_id()
        for chain1 in chains:
            for chain2 in chains:
                if chain1 == chain2:
                    continue
                score1, score2 = self.compute_pdockq_pair(chain1, chain2, dist_cutoff=dist_cutoff)
                results1[(chain1, chain2)] = score1
                results2[(chain1, chain2)] = score2
        results1['complex'] = np.mean(list(results1.values())) if results1 else 0.0
        results2['complex'] = np.mean(list(results2.values())) if results2 else 0.0
        return results1, results2



    def ptm_matrix_pair(self, chain1, chain2, pae_cutoff=None, L_mode="all"):
        """
        Utility: Return PTM matrix for a chain pair, optionally masked by PAE cutoff.
        L_mode: 'all' uses all residues, 'valid' uses only unique residues in valid pairs for normalization.
        """
        idx1 = self.protein.rs(chain1).data
        idx2 = self.protein.rs(chain2).data
        pae_sub = self.pae[np.ix_(idx1, idx2)]
        if pae_cutoff is not None:
            mask = pae_sub < pae_cutoff
            if L_mode == "valid":
                # Count unique residues in chain1 and chain2 involved in any valid pair
                valid1 = np.any(mask, axis=1)
                valid2 = np.any(mask, axis=0)
                L = np.sum(valid1) + np.sum(valid2)
                if L < 1:
                    L = 1.0
            else:
                L = len(idx1) + len(idx2)
            d0 = self.calc_d0(L)
            ptm_mat = self.ptm_func(pae_sub, d0)
            ptm_mat = np.where(mask, ptm_mat, np.nan)
        else:
            L = len(idx1) + len(idx2)
            d0 = self.calc_d0(L)
            ptm_mat = self.ptm_func(pae_sub, d0)
        if ptm_mat.size > 0:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                score = np.nanmax(np.nanmean(ptm_mat, axis=1), axis=0)
        else:
            score = 0.0
        return (ptm_mat, score)

    def ptm_func(self, x, d0):
        """PTM scoring function."""
        return 1.0 / (1 + (x / d0) ** 2.0)

    def calc_d0(self, L):
        """Calculate d0 for PTM/ipSAE scoring (protein chains only)."""
        L = float(L)
        if L < 27:
            L = 27
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
        return max(1.0, d0)

    def compute_lis_pair(self, chain1, chain2, pae_threshold=12.0):
        """
        Compute LIS (Local Interface Score) for a given chain pair.
        Uses PAE threshold.
        """
        if self.pae is None:
            return np.nan
        idx1 = self.protein.rs(chain1).data
        idx2 = self.protein.rs(chain2).data
        pae_sub = self.pae[np.ix_(idx1, idx2)]
        mask = pae_sub <= pae_threshold
        valid_pae = pae_sub[mask]
        if valid_pae.size > 0:
            scores = (pae_threshold - valid_pae) / pae_threshold
            avg_score = np.mean(scores)
        else:
            avg_score = 0.0
        return avg_score

    def compute_lis(self, pae_threshold=12.0):
        """
        Compute LIS (Local Interface Score) for all chain pairs and overall complex.
        Returns a dict of results.
        """
        results = {}
        chains = self.protein.chain_id()
        for chain1 in chains:
            for chain2 in chains:
                if chain1 == chain2:
                    continue
                results[(chain1, chain2)] = self.compute_lis_pair(chain1, chain2, pae_threshold=pae_threshold)
        results['complex'] = np.mean(list(results.values())) if results else 0.0
        return results

    def compute_metrics_pair(self, chain1, chain2, ipsae_pae_cutoff=10.0,
                             pdockq_dist_cutoff=8.0, lis_pae_cutoff=12.0, pae_plddt_int_cutoff=5.0):
        # compute all metrics for the given chain1 and chain2
        # return max if the score is assymetric
        iptm = max(self.compute_iptm_pair(chain1, chain2),
                   self.compute_iptm_pair(chain2, chain1))
        ipsae = max(self.compute_ipsae_pair(chain1, chain2, pae_cutoff=ipsae_pae_cutoff),
                    self.compute_ipsae_pair(chain2, chain1, pae_cutoff=ipsae_pae_cutoff))
        pdockq_1, pdockq2_1 = self.compute_pdockq_pair(chain1, chain2, dist_cutoff=pdockq_dist_cutoff)
        pdockq_2, pdockq2_2 = self.compute_pdockq_pair(chain2, chain1, dist_cutoff=pdockq_dist_cutoff)
        pdockq = max(pdockq_1, pdockq_2)  # two scores should be identical
        pdockq2 = max(pdockq2_1, pdockq2_2)
        lis = max(self.compute_lis_pair(chain1, chain2, pae_threshold=lis_pae_cutoff),
                  self.compute_lis_pair(chain2, chain1, pae_threshold=lis_pae_cutoff))
        pae_int = self.compute_pae_int_pair(chain1, chain2, cutoff=pae_plddt_int_cutoff)
        plddt_int = self.compute_plddt_int_pair(chain1, chain2, cutoff=pae_plddt_int_cutoff)

        return {
            "PTM": iptm,
            "ipSAE": ipsae,
            "pDockQ": pdockq,
            "pDockQ2": pdockq2,
            "LIS": lis,
            "PAE_int": pae_int,
            "pLDDT_int": plddt_int
        }

# Example usage for direct comparison with ipsae.py

if __name__ == "__main__":
    import json
    import numpy as np
    # Example file paths (update as needed)
    pae_file = "IPSAE/Example/RAF1_KSR1_MEK1_9f755_scores_alphafold2_multimer_v3_model_1_seed_000.json"
    pdb_file = "IPSAE/Example/RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000.pdb"

    # Load PAE and pLDDT from JSON
    with open(pae_file, "r") as f:
        data = json.load(f)
    pae = np.array(data["pae"])
    plddt = np.array(data["plddt"])

    # Initialize StructureMetrics
    metrics = StructureMetrics(pdb_file, plddt, pae)

    print(metrics.compute_metrics_pair("A", "B"))
    print(metrics.compute_metrics_pair("A", "C"))
    print(metrics.compute_metrics_pair("B", "C"))
    print(metrics.compute_metrics_pair("A", "C:B"))
    exit()

    # Calculate metrics
    iptm = metrics.compute_iptm()
    ipsae = metrics.compute_ipsae(pae_cutoff=10.0)
    pdockq, pdockq2 = metrics.compute_pdockq(dist_cutoff=8.0)
    lis = metrics.compute_lis(pae_threshold=12.0)
    pae_int = metrics.compute_pae_int('A', 'B', cutoff=5.0)
    plddt_int = metrics.compute_plddt_int('A', 'B', cutoff=5.0)

    # Output results for comparison
    print("iPTM:", iptm)
    print("ipSAE:", ipsae)
    print("pDockQ:", pdockq)
    print("pDockQ2:", pdockq2)
    print("LIS:", lis)
