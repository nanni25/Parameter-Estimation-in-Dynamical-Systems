import xml.etree.ElementTree as ET

def modify_sbml(input_file, output_file):
    print(f"Loading raw SBML from: {input_file}")
    
    # Register SBML namespace to avoid annoying ns0: prefixes
    ns_url = "http://www.sbml.org/sbml/level3/version1/core"
    ET.register_namespace('', ns_url)
    ns = {'sbml': ns_url}
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    model = root.find('sbml:model', ns)

    # 1. IDENTIFY TOPOLOGY
    all_species = [s.get('id') for s in model.findall('.//sbml:species', ns)]
    
    reactants = set(sr.get('species') for sr in model.findall('.//sbml:listOfReactants/sbml:speciesReference', ns))
    products = set(sr.get('species') for sr in model.findall('.//sbml:listOfProducts/sbml:speciesReference', ns))
    modifiers = set(sr.get('species') for sr in model.findall('.//sbml:listOfModifiers/sbml:modifierSpeciesReference', ns))
    
    # Inputs: Species that are never produced
    input_species = [s for s in all_species if s not in products]
    # Outputs: Species that are produced, but never consumed or used as a catalyst
    output_species = [s for s in all_species if s not in reactants and s not in modifiers]
    
    # 2. INJECT PARAMETERS
    list_of_params = model.find('sbml:listOfParameters', ns)
    if list_of_params is None:
        list_of_params = ET.SubElement(model, 'listOfParameters')
        
    for rxn in model.findall('.//sbml:reaction', ns):
        ET.SubElement(list_of_params, 'parameter', {'id': f"lambda_{rxn.get('id')}", 'value': '10.0', 'constant': 'true'})
        
    for s in input_species:
        ET.SubElement(list_of_params, 'parameter', {'id': f"K_in_{s.replace('species_', '')}", 'value': '10.0', 'constant': 'true'})
        
    for s in output_species:
        ET.SubElement(list_of_params, 'parameter', {'id': f"K_out_{s.replace('species_', '')}", 'value': '1.0', 'constant': 'true'})

    for s in all_species:
        id_num = s.replace('species_', '')
        ET.SubElement(list_of_params, 'parameter', {'id': f"y_{id_num}", 'value': '0.0', 'constant': 'false'})
        ET.SubElement(list_of_params, 'parameter', {'id': f"y2_{id_num}", 'value': '0.0', 'constant': 'false'})

    # 3. INJECT RATE RULES (Calculators)
    list_of_rules = model.find('sbml:listOfRules', ns)
    if list_of_rules is None:
        list_of_rules = ET.SubElement(model, 'listOfRules')

    math_ns = 'http://www.w3.org/1998/Math/MathML'
    for s in all_species:
        id_num = s.replace('species_', '')
        
        # Mean Rule (y)
        y_rule = ET.SubElement(list_of_rules, 'rateRule', {'variable': f"y_{id_num}"})
        y_math = f'<math xmlns="{math_ns}"><apply><divide/><apply><minus/><ci>{s}</ci><ci>y_{id_num}</ci></apply><apply><plus/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol><cn type="e-notation">1<sep/>-6</cn></apply></apply></math>'
        y_rule.append(ET.fromstring(y_math))
        
        # Second Moment Rule (y2)
        y2_rule = ET.SubElement(list_of_rules, 'rateRule', {'variable': f"y2_{id_num}"})
        y2_math = f'<math xmlns="{math_ns}"><apply><divide/><apply><minus/><apply><power/><ci>{s}</ci><cn>2</cn></apply><ci>y2_{id_num}</ci></apply><apply><plus/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol><cn type="e-notation">1<sep/>-6</cn></apply></apply></math>'
        y2_rule.append(ET.fromstring(y2_math))

    # 4. REWRITE BIOLOGICAL REACTIONS (Hill Functions)
    for rxn in model.findall('.//sbml:reaction', ns):
        rxn_id = rxn.get('id')
        subs = [sr.get('species') for sr in rxn.findall('.//sbml:listOfReactants/sbml:speciesReference', ns)]
        
        # Get both the species ID and its SBO term to determine activation vs inactivation
        mods = []
        for sr in rxn.findall('.//sbml:listOfModifiers/sbml:modifierSpeciesReference', ns):
            mods.append({
                'species': sr.get('species'),
                'sboTerm': sr.get('sboTerm')
            })
        
        math_str = f'<math xmlns="{math_ns}"><apply><times/><ci>lambda_{rxn_id}</ci>'
        
        for mod in mods:
            mod_id = mod['species']
            sbo = mod['sboTerm']
            
            # If it is an inhibitor (SBO:0000020), apply H-(x)
            if sbo == "SBO:0000020":
                math_str += f'<apply><divide/><apply><power/><cn>0.5</cn><cn>10</cn></apply><apply><plus/><apply><power/><cn>0.5</cn><cn>10</cn></apply><apply><power/><ci>{mod_id}</ci><cn>10</cn></apply></apply></apply>'
            # Otherwise (catalyst/activator), apply H+(x)
            else:
                math_str += f'<apply><divide/><apply><power/><ci>{mod_id}</ci><cn>10</cn></apply><apply><plus/><apply><power/><cn>0.5</cn><cn>10</cn></apply><apply><power/><ci>{mod_id}</ci><cn>10</cn></apply></apply></apply>'
            
        for sub in subs:
            math_str += f'<ci>{sub}</ci>'
            
        math_str += '</apply></math>'
        
        kinetic_law = ET.SubElement(rxn, 'kineticLaw')
        kinetic_law.append(ET.fromstring(math_str))

    # 5. INJECT DUMMY REACTIONS (Open System)
    list_of_rxns = model.find('sbml:listOfReactions', ns)
    
    for s in input_species:
        id_num = s.replace('species_', '')
        rxn = ET.SubElement(list_of_rxns, 'reaction', {'id': f"source_{id_num}", 'reversible': 'false', 'fast': 'false'})
        prods = ET.SubElement(rxn, 'listOfProducts')
        ET.SubElement(prods, 'speciesReference', {'species': s, 'stoichiometry': '1', 'constant': 'true'})
        kl = ET.SubElement(rxn, 'kineticLaw')
        kl.append(ET.fromstring(f'<math xmlns="{math_ns}"><ci>K_in_{id_num}</ci></math>'))

    for s in output_species:
        id_num = s.replace('species_', '')
        rxn = ET.SubElement(list_of_rxns, 'reaction', {'id': f"sink_{id_num}", 'reversible': 'false', 'fast': 'false'})
        reacts = ET.SubElement(rxn, 'listOfReactants')
        ET.SubElement(reacts, 'speciesReference', {'species': s, 'stoichiometry': '1', 'constant': 'true'})
        kl = ET.SubElement(rxn, 'kineticLaw')
        kl.append(ET.fromstring(f'<math xmlns="{math_ns}"><ci>K_out_{id_num}</ci></math>'))

    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Successfully cured SBML and saved to: {output_file}")

if __name__ == "__main__":
    modify_sbml("Models/R-HSA-5653890_Original.sbml", "Models/R-HSA-5653890.sbml")