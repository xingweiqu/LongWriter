from inference_model import load_model, infer
import json

def extract_instructions(json_data):
    documents = []
    for document in json_data:
        doc_instructions = {}
        for instruction in document.get('section_instructions', []):
            prompt = instruction['instruction']
            section_number = instruction['section_number']
            section_title = instruction['section_title']
            
            word_count = int(prompt.split('Approximate word count:')[1].split('words')[0].strip())
            max_tokens = word_count * 2 + 100  # Assuming average word length of 2 tokens, plus 100 extra tokens

            doc_instructions[section_number] = {
                'prompt': prompt,
                'max_tokens': max_tokens,
                'section_title': section_title
            }
        documents.append(doc_instructions)
    return documents

# Load the JSON data
with open('/gpfs/public/research/gezhang/git/LongWriter/data/paper/processed_data.json', 'r') as f:
    data = json.load(f)

# Extract instructions for all documents
extracted_documents = extract_instructions(data)

# Prepare for inference
model_args = {
    'model_path_or_name': '/gpfs/public/01/models/hf_models/Meta-Llama-3.1-8B-Instruct',
    'model_type': 'local',
    'tp': 8
}

model_components = load_model("Meta-Llama-3.1-8B-Instruct", model_args, use_accel=True)

# Process each document
results = []
for doc_index, document in enumerate(extracted_documents):
    doc_results = {}
    for section_number, instruction in document.items():
        prompt = instruction['prompt']
        max_tokens = instruction['max_tokens']
        section_title = instruction['section_title']
        
        responses = infer([prompt], None, max_tokens=max_tokens, **model_components)
        
        # Assuming only one response per prompt
        response = responses[0] if responses else ""
        
        doc_results[section_number] = {
            "section_title": section_title,
            "prompt": prompt,
            "response": response
        }
    
    results.append(doc_results)

# Save all results to a single JSON file
output_file = "all_documents_sections.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"All documents processed and saved to {output_file}")