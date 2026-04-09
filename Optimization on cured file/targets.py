import xml.etree.ElementTree as ET
import json
from groq import Groq
import os
# --- CONFIGURATION ---
API_KEY = os.environ.get("GROQ_API_KEY") # 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"
SBML_FILE_PATH = "Models/R-HSA-5653890.sbml"
OUTPUT_JSON_PATH = "targets.json"

def extract_species_from_sbml(file_path):

    print(f"Extracting biological data from {file_path}...")
    
    ns = {'sbml': 'http://www.sbml.org/sbml/level3/version1/core'}
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing SBML file: {e}")
        return []

    species_data = []
    
    for species in root.findall('.//sbml:species', ns):
        s_id = species.get('id')
        s_name = species.get('name')
        
        if s_id and s_name:
            species_data.append({"id": s_id, "name": s_name})
            
    print(f"Found {len(species_data)} species.")
    return species_data

def query_groq_for_targets(species_list, api_key):
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert computational systems biologist. I am building a kinetic optimizer 
    and need realistic target steady-state mean concentrations for the following biological species 
    extracted from a Reactome SBML model.
    
    Species Data:
    {json.dumps(species_list, indent=2)}
    
    Task:
    Estimate a realistic steady-state mean concentration for each species. Provide the raw values
    without any unit (e.g. 1).
    
    Output format:
    You MUST return ONLY a raw JSON object. The keys must be the exact 'id' from the list 
    (e.g., "species_964746") and the values must be your float estimates.
    """
    
    print("Sending prompt to Groq (Llama 3 70B)...")
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        response_format={"type": "json_object"} 
    )
    
    return chat_completion.choices[0].message.content

def main():

    # 1. Extract data from the SBML file
    species_list = extract_species_from_sbml(SBML_FILE_PATH)
    
    if not species_list:
        print("No species found. Exiting.")
        return

    # 2. Query the LLM
    try:
        json_response = query_groq_for_targets(species_list, API_KEY)
        
        # 3. Parse and save the results
        targets = json.loads(json_response)
        
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(targets, f, indent=4)
            
        print(f"\nSuccess! Saved target estimates to {OUTPUT_JSON_PATH}")
        print(json.dumps(targets, indent=2))
        
    except Exception as e:
        print(f"An error occurred while querying the LLM: {e}")

if __name__ == "__main__":
    main()