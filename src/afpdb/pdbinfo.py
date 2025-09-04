#pip install rcsb-api

def get_pdb_info(pdb_id):
    from rcsbapi.data import DataQuery
    # Create a DataQuery instance
    query = DataQuery(
        input_type="entry",
        input_ids=[pdb_id],
        return_data_list=[
            "rcsb_accession_info.initial_release_date",
            "rcsb_accession_info.deposit_date",
            "rcsb_accession_info.revision_date",
            "rcsb_entry_info.resolution_combined",
            "polymer_entities.rcsb_entity_source_organism.taxonomy_lineage.name",
            "polymer_entities.rcsb_entity_host_organism.taxonomy_lineage.name",
            "exptl.method",
            "polymer_entities.entity_poly.pdbx_seq_one_letter_code_can",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.auth_asym_ids",
        ]
    )

    def get_val(x, key, default):
        """Return default, if it does not exist, or exist as a None value"""
        if key in x and x[key] is not None:
            return x[key]
        return default

    # Execute the query
    results = query.exec()

    release_date=deposit_date=revision_date=None
    resolution=source_organism=expression_system=method=None
    seq_dict={}

    try:
        # Extract the required information from the results
        entry_data = results['data']['entries'][0]
        release_date = get_val(get_val(entry_data, "rcsb_accession_info", {}), "initial_release_date", None)
        deposit_date = get_val(get_val(entry_data, "rcsb_accession_info", {}), "deposit_date", None)
        revision_date = get_val(get_val(entry_data, "rcsb_accession_info", {}), "revision_date", None)
        resolution = get_val(get_val(entry_data, "rcsb_entry_info", {}), "resolution_combined", [None])[0]

        for entity in get_val(entry_data, "polymer_entities", [{}]):
            sequence = entity.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
            chains = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
            for chain_id in chains:
                seq_dict[chain_id] = sequence

        polymer = get_val(entry_data, "polymer_entities", [{}])[0]
        organism = get_val(polymer, "rcsb_entity_source_organism", [{}])[-1]
        #organism=", ".join([v for x in organism for y in get_val(x, "taxonomy_lineage", [{}]) for k,v in y.items() if k=='name'])
        organism = get_val(organism, "taxonomy_lineage", [{}])[-1]
        organism = get_val(organism, "name", None)
        expression_system = get_val(polymer, "rcsb_entity_host_organism", [{}])[-1]
        expression_system = get_val(expression_system, "taxonomy_lineage", [{}])[-1]
        expression_system = get_val(expression_system, "name", None)
        method = get_val(get_val(entry_data, "exptl", [{}])[0], "method", None)

    except Exception as e:
        print(e)

    return {
        'pdb_id':pdb_id.lower(),
        'release_date':release_date,
        'deposit_date':deposit_date,
        'revision_date':deposit_date,
        'resolution':resolution,
        'source_organism':organism,
        'expression_system':expression_system,
        'method':method,
        'seq_dict': seq_dict
        }


if __name__=="__main__":
    print(get_pdb_info("8ytk"))
