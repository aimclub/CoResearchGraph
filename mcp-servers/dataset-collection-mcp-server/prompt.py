extract_mol_properties_prompt = (
    "You will receive a USER QUESTION asking you to extract a dataset of molecules and their properties from PDF documents. "
    "From the USER QUESTION, identify which molecular properties are required (e.g., MIC, IC50, LD50, solubility, permeability, etc.). "
    "Then, output a CSV table with the following columns:\n"
    "id, property, units, value\n"
    "— 'id' is the molecule identifier as reported in the paper (e.g., 1a, 5, 28, etc.).\n"
    "— 'property' is the name of the reported property (e.g., pMIC and MIC are different).\n"
    "— 'units' are the measurement units of the property, as stated in the document.\n"
    "— 'value' is the numerical value of the property.\n"
    "Include all molecules for which the required properties are reported — do not omit any.\n"
    "Do not include molecules where the required property is not mentioned."
    "Output only the CSV data (no explanations, no markdown, no additional text)."
)