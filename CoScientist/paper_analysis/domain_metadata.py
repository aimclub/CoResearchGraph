import ast

def simplify_extracted_reactions(data: list) -> list:
    """Simplifies the extracted reactions data."""
    simplified = []

    for item in data:
        for reaction in item.get("reactions", []):

            reactants = [
                r["smiles"]
                for r in reaction.get("reactants", [])
                if r.get("smiles")
            ]

            conditions = []

            for c in reaction.get("conditions", []):

                if c.get("smiles"):
                    conditions.append(c["smiles"])

                if c.get("text"):
                    if isinstance(c["text"], list):
                        conditions.extend(c["text"])
                    else:
                        conditions.append(c["text"])

            products = [
                p["smiles"]
                for p in reaction.get("products", [])
                if p.get("smiles")
            ]

            simplified.append({
                "reactants": reactants,
                "conditions": conditions,
                "products": products,
            })

    return simplified

def add_domain_metadata_to_img_info(domain: str, img_meta: dict, img_info: dict) -> dict:
    """Adds domain-specific metadata to the image information dictionary."""
    if domain == "Chemistry":
        if 'domain_metadata.molecules' in img_meta:
            mols = ast.literal_eval(img_meta['domain_metadata.molecules'])
            img_info['Molecules'] = [m["smiles"] for m in mols[0]['bboxes'] if "smiles" in m]

        if 'domain_metadata.reactions' in img_meta:
            reactions = ast.literal_eval(img_meta['domain_metadata.reactions'])
            img_info['Reactions'] = simplify_extracted_reactions(reactions)
    return img_info


def format_domain_metadata(domain: str, img_info_list: list) -> str:
    """Formats domain-specific metadata for text context."""
    domain_metadata = ""
    if domain == "Chemistry":
        for img_info in img_info_list:
            if 'Molecules' in img_info:
                domain_metadata += f"Molecules found in {img_info['Paper']}:\n{img_info['Molecules']}\n\n"
            if 'Reactions' in img_info:
                domain_metadata += f"Reactions found in {img_info['Paper']}:\n{img_info['Reactions']}\n\n"
    return domain_metadata