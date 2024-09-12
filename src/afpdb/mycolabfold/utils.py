import re
from Bio.PDB import MMCIFIO
from Bio.PDB.Polypeptide import standard_aa_names
#YZ
from collections import defaultdict
#

CIF_REVISION_DATE = """loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 1971-01-01
#\n"""

### begin section copied from Bio.PDB
mmcif_order = {
    "_atom_site": [
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "pdbx_formal_charge",
        "auth_seq_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_atom_id",
        "pdbx_PDB_model_num",
    ]
}

## YZ
# if True: keep the original chain name in PDB
#   use https://github.com/sokrypton/ColabFold/issues/449
# if False: rename chain to A, B, C, but add a mapping section in cif to fix AF error
#   use https://github.com/speleo3/ColabFold/commit/68e7090c0c2401257e4a9392370f66b28ac82543
_KEEP_CHAIN=False
##

class CFMMCIFIO(MMCIFIO):
    def _save_dict(self, out_file):
        ## YZ
        # alternatively, use fix provided by
        #  https://github.com/speleo3/ColabFold/commit/68e7090c0c2401257e4a9392370f66b28ac82543
        if not _KEEP_CHAIN:
            asym_id_auth_to_label = dict(
                zip(self.dic.get("_atom_site.auth_asym_id", ()),
                    self.dic.get("_atom_site.label_asym_id", ())))
        ##
        # Form dictionary where key is first part of mmCIF key and value is list
        # of corresponding second parts
        key_lists = {}
        for key in self.dic:
            if key == "data_":
                data_val = self.dic[key]
            else:
                s = re.split(r"\.", key)
                if len(s) == 2:
                    if s[0] in key_lists:
                        key_lists[s[0]].append(s[1])
                    else:
                        key_lists[s[0]] = [s[1]]
                else:
                    raise ValueError("Invalid key in mmCIF dictionary: " + key)

        # Re-order lists if an order has been specified
        # Not all elements from the specified order are necessarily present
        for key, key_list in key_lists.items():
            if key in mmcif_order:
                inds = []
                for i in key_list:
                    try:
                        inds.append(mmcif_order[key].index(i))
                    # Unrecognised key - add at end
                    except ValueError:
                        inds.append(len(mmcif_order[key]))
                key_lists[key] = [k for _, k in sorted(zip(inds, key_list))]

        # Write out top data_ line
        if data_val:
            out_file.write("data_" + data_val + "\n#\n")
            ### end section copied from Bio.PDB
            # Add poly_seq as default MMCIFIO doesn't handle this
            out_file.write(
                """loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
#\n"""
            )
            poly_seq = []
            chain_idx = 1
            for model in self.structure:
                for chain in model:
                    res_idx = 1
                    for residue in chain:
                        poly_seq.append(
                            (chain_idx, res_idx, residue.get_resname(), "n")
                        )
                        res_idx += 1
                    chain_idx += 1
            for seq in poly_seq:
                out_file.write(f"{seq[0]} {seq[1]} {seq[2]}  {seq[3]}\n")
            out_file.write("#\n")
            out_file.write(
                """loop_
_chem_comp.id
_chem_comp.type
#\n"""
            )
            for three in standard_aa_names:
                out_file.write(f'{three} "peptide linking"\n')
            out_file.write("#\n")
            out_file.write(
                """loop_
_struct_asym.id
_struct_asym.entity_id
#\n"""
            )
            chain_idx = 1
            for model in self.structure:
                for chain in model:
                    ## YZ
                    if _KEEP_CHAIN:
                        out_file.write(f"{chain.get_id()} {chain_idx}\n")
                    else:
                        label_asym_id = asym_id_auth_to_label[chain.get_id()]
                        out_file.write(f"{label_asym_id} {chain_idx}\n")
                    ####
                    chain_idx += 1
            out_file.write("#\n")

        ### begin section copied from Bio.PDB
        for key, key_list in key_lists.items():
            # Pick a sample mmCIF value, which can be a list or a single value
            sample_val = self.dic[key + "." + key_list[0]]
            n_vals = len(sample_val)
            # Check the mmCIF dictionary has consistent list sizes
            for i in key_list:
                val = self.dic[key + "." + i]
                if (
                    isinstance(sample_val, list)
                    and (isinstance(val, str) or len(val) != n_vals)
                ) or (isinstance(sample_val, str) and isinstance(val, list)):
                    raise ValueError(
                        "Inconsistent list sizes in mmCIF dictionary: " + key + "." + i
                    )
            # If the value is a single value, write as key-value pairs
            if isinstance(sample_val, str) or (
                isinstance(sample_val, list) and len(sample_val) == 1
            ):
                m = 0
                # Find the maximum key length
                for i in key_list:
                    if len(i) > m:
                        m = len(i)
                for i in key_list:
                    # If the value is a single item list, just take the value
                    if isinstance(sample_val, str):
                        value_no_list = self.dic[key + "." + i]
                    else:
                        value_no_list = self.dic[key + "." + i][0]
                    out_file.write(
                        "{k: <{width}}".format(k=key + "." + i, width=len(key) + m + 4)
                        + self._format_mmcif_col(value_no_list, len(value_no_list))
                        + "\n"
                    )
            # If the value is more than one value, write as keys then a value table
            elif isinstance(sample_val, list):
                out_file.write("loop_\n")
                col_widths = {}
                # Write keys and find max widths for each set of values
                for i in key_list:
                    out_file.write(key + "." + i + "\n")
                    col_widths[i] = 0
                    for val in self.dic[key + "." + i]:
                        len_val = len(val)
                        # If the value requires quoting it will add 2 characters
                        if self._requires_quote(val) and not self._requires_newline(
                            val
                        ):
                            len_val += 2
                        if len_val > col_widths[i]:
                            col_widths[i] = len_val
                # Technically the max of the sum of the column widths is 2048

                # Write the values as rows
                for i in range(n_vals):
                    for col in key_list:
                        out_file.write(
                            self._format_mmcif_col(
                                self.dic[key + "." + col][i], col_widths[col] + 1
                            )
                        )
                    out_file.write("\n")
            else:
                raise ValueError(
                    "Invalid type in mmCIF dictionary: " + str(type(sample_val))
                )
            out_file.write("#\n")
            ### end section copied from Bio.PDB
            out_file.write(CIF_REVISION_DATE)

    # Preserve chain_id
    # https://github.com/sokrypton/ColabFold/issues/449
    def _save_structure(self, out_file, select, preserve_atom_numbering):
        atom_dict = defaultdict(list)

        for model in self.structure.get_list():
            if not select.accept_model(model):
                continue
            # mmCIF files with a single model have it specified as model 1
            if model.serial_num == 0:
                model_n = "1"
            else:
                model_n = str(model.serial_num)
            # This is used to write label_entity_id and label_asym_id and
            # increments from 1, changing with each molecule
            entity_id = 0
            if not preserve_atom_numbering:
                atom_number = 1
            for chain in model.get_list():
                if not select.accept_chain(chain):
                    continue
                chain_id = chain.get_id()
                if chain_id == " ":
                    chain_id = "."
                # This is used to write label_seq_id and increments from 1,
                # remaining blank for hetero residues
                residue_number = 1
                prev_residue_type = ""
                prev_resname = ""
                for residue in chain.get_unpacked_list():
                    if not select.accept_residue(residue):
                        continue
                    hetfield, resseq, icode = residue.get_id()
                    if hetfield == " ":
                        residue_type = "ATOM"
                        label_seq_id = str(residue_number)
                        residue_number += 1
                    else:
                        residue_type = "HETATM"
                        label_seq_id = "."
                    resseq = str(resseq)
                    if icode == " ":
                        icode = "?"
                    resname = residue.get_resname()
                    # Check if the molecule changes within the chain
                    # This will always increment for the first residue in a
                    # chain due to the starting values above
                    if residue_type != prev_residue_type or (
                        residue_type == "HETATM" and resname != prev_resname
                    ):
                        entity_id += 1
                    prev_residue_type = residue_type
                    prev_resname = resname
                    label_asym_id = self._get_label_asym_id(entity_id)
                    ##YZ
                    if _KEEP_CHAIN:
                        label_asym_id = chain.id
                    ##YZ END
                    for atom in residue.get_unpacked_list():
                        if select.accept_atom(atom):
                            atom_dict["_atom_site.group_PDB"].append(residue_type)
                            if preserve_atom_numbering:
                                atom_number = atom.get_serial_number()
                            atom_dict["_atom_site.id"].append(str(atom_number))
                            if not preserve_atom_numbering:
                                atom_number += 1
                            element = atom.element.strip()
                            if element == "":
                                element = "?"
                            atom_dict["_atom_site.type_symbol"].append(element)
                            atom_dict["_atom_site.label_atom_id"].append(
                                atom.get_name().strip()
                            )
                            altloc = atom.get_altloc()
                            if altloc == " ":
                                altloc = "."
                            atom_dict["_atom_site.label_alt_id"].append(altloc)
                            atom_dict["_atom_site.label_comp_id"].append(
                                resname.strip()
                            )
                            atom_dict["_atom_site.label_asym_id"].append(label_asym_id)
                            # The entity ID should be the same for similar chains
                            # However this is non-trivial to calculate so we write "?"
                            atom_dict["_atom_site.label_entity_id"].append("?")
                            atom_dict["_atom_site.label_seq_id"].append(label_seq_id)
                            atom_dict["_atom_site.pdbx_PDB_ins_code"].append(icode)
                            coord = atom.get_coord()
                            atom_dict["_atom_site.Cartn_x"].append(f"{coord[0]:.3f}")
                            atom_dict["_atom_site.Cartn_y"].append(f"{coord[1]:.3f}")
                            atom_dict["_atom_site.Cartn_z"].append(f"{coord[2]:.3f}")
                            atom_dict["_atom_site.occupancy"].append(
                                str(atom.get_occupancy())
                            )
                            atom_dict["_atom_site.B_iso_or_equiv"].append(
                                str(atom.get_bfactor())
                            )
                            atom_dict["_atom_site.auth_seq_id"].append(resseq)
                            atom_dict["_atom_site.auth_asym_id"].append(chain_id)
                            atom_dict["_atom_site.pdbx_PDB_model_num"].append(model_n)

        # Data block name is the structure ID with special characters removed
        structure_id = self.structure.id
        for c in ["#", "$", "'", '"', "[", "]", " ", "\t", "\n"]:
            structure_id = structure_id.replace(c, "")
        atom_dict["data_"] = structure_id

        # Set the dictionary and write out using the generic dictionary method
        self.dic = atom_dict
        self._save_dict(out_file)

