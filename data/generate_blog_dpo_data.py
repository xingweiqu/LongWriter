import json
from openai import OpenAI
from vllm import LLM, SamplingParams

# from inference_model import load_model, infer


dataset_name = 'blog'

# Initialize OpenAI client
# base_url = ""
# api_key = ""
# client = OpenAI(base_url=base_url, api_key=api_key)

# Example input JSON file (replace this with the actual file path)
# input_file = "/map-vepfs/shuyue/xw_huggingface/LongWriter/data/generated_dpo_500.json"
# input_file = "/map-vepfs/shuyue/xw_huggingface/LongWriter/data/generated_blog_dpo_500.json"
# input_file = "/map-vepfs/shuyue/xw_huggingface/LongWriter/data/blog_detailed_prompts.json"
# input_file = "/map-vepfs/shuyue/xw_huggingface/LongWriter/data/blog_planned_prompts.json"
input_file = "/map-vepfs/shuyue/xw_huggingface/LongWriter/data/blog_title_prompts_all.json"

# Output file
# output_file = "dpo_blog_data_negative_long_writer_positive_gt.json"
# output_file = "dpo_blog_detailed_negative_long_writer_positive_gt.json"
# output_file = "dpo_blog_planned_negative_long_writer_positive_gt.json"
output_file = "dpo_blog_title_negative_long_writer_positive_gt.json"
# Extract instructions for all documents

# Prepare for inference
# model_args = {
#     "model_path_or_name": "/map-vepfs/shuyue/xw_huggingface/LongWriter-llama3.1-8b",
#     "model_type": "local",
#     "tp": 8,
# }


# model_components = load_model("Meta-Llama-3.1-8B-Instruct", model_args, use_accel=True)
model = LLM(
    model="/map-vepfs/shuyue/xw_huggingface/LongWriter-llama3.1-8b",
    dtype="auto",
    trust_remote_code=True,
    tensor_parallel_size=2,
    max_model_len=32768,
    gpu_memory_utilization=1,
)
tokenizer = model.get_tokenizer()
generation_params = SamplingParams(
    temperature=0.5,
    top_p=0.8,
    top_k=50,
    max_tokens=32768,
    repetition_penalty=1,
)


def infer(query):
    # query = "Write a 10000-word China travel guide"
    prompt = f"[INST]{query}[/INST]"
    input_ids = (
        tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0].tolist()
    )
    outputs = model.generate(
        sampling_params=generation_params,
        prompt_token_ids=[input_ids],
    )
    output = outputs[0]
    return output.outputs[0].text


# Process each document
# results = []
# for doc_index, document in enumerate(extracted_documents):
#     doc_results = {}
#     for section_number, instruction in document.items():
#         prompt = instruction['prompt']
#         max_tokens = instruction['max_tokens']
#         section_title = instruction['section_title']

#         responses = infer([prompt], None, max_tokens=max_tokens, **model_components)

#         # Assuming only one response per prompt
#         response = responses[0] if responses else ""

#         doc_results[section_number] = {
#             "section_title": section_title,
#             "prompt": prompt,
#             "response": response
#         }

#     results.append(doc_results)

# Save all results to a single JSON file
# output_file = "all_documents_sections.json"
# with open(output_file, "w") as f:
#     json.dump(results, f, indent=2)

# print(f"All documents processed and saved to {output_file}")


# Function to generate summary
# def generate_summary(prompt):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-2024-08-06",
#             messages=[{"role": "user", "content": prompt}],
#             stream=False,
#             max_tokens=10000,
#             temperature=0.6,
#             top_p=0.9,
#             n=1,
#             stop=None,
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"Error generating summary: {e}")
#         return "Error occurred while generating."


# Generate a single paper in Markdown format
def generate_single_paper(input_sections):
    paper_md = "# Complete Paper\n\n"
    sections_data = []
    input_sections = input_sections[0]

    # print(input_sections)

    # # print(input_sections.keys())
    # input()

    # section = input_sections["title_prompt"]
    section = input_sections 

    


    paper_prompt = f"You are an AI researcher who is reviewing a paper that was submitted to a prestigious conference or journal. "
    blog_prompt = f"You are an AI researcher who is reviewing a blog that was submitted to a website. "
    wiki_prompt = f"You are an AI researcher who is reviewing a wikipedia that was submitted to a website. "

    if dataset_name == 'blog':
        final_prompt = blog_prompt
    elif dataset_name == 'paper':
        final_prompt = paper_prompt
    else:
        final_prompt = wiki_prompt


    section_prompt = (
        f"{final_prompt}" ## 改一下
        f"Write the {section['title']} section of the paper. "
        f"Focus on {section['summary']} , You should write this paper in about {section['target_length']} words. "
        f"Use technical language suitable for a submission to a top-tier venue."
    )

    print(f"Generating content for: {section['title']}")
    print(section_prompt)
    # content = infer(
    #     [section_prompt],
    #     None,
    #     max_tokens=section["target_length"] + 1000,
    #     **model_components,
    # )
    content = infer(section_prompt)
    print(content)
    # content = generate_summary(section_prompt)

    # Add section to Markdown document
    paper_md += f"## {section['title']}\n\n"
    paper_md += content + "\n\n"

    # # Add section data to the structure
    # sections_data.append(
    #     {
    #         # "section_number": section["section_number"],
    #         "title": section["title"],
    #         "content": content,
    #     }
    # )

    return paper_md, sections_data


# Generate multiple papers and save them
def generate_all_papers(input_data):
    all_papers = []

    for idx, paper_sections in enumerate(input_data):
        print(f"Processing paper {idx + 1}...")
        paper_md, sections_data = generate_single_paper(paper_sections)

        # Add paper data to the overall structure
        all_papers.append(
            {
                "paper_id": idx + 1,
                "markdown": paper_md,
            }
        )

    return all_papers


# Load input JSON
with open(input_file, "r", encoding='utf-8') as file:
    input_data = json.load(file)

# Generate all papers
all_papers_data = generate_all_papers(input_data)

# Save all papers to a JSON file
with open(output_file, "w") as file:
    json.dump(all_papers_data, file, indent=4)

print(f"All papers saved to {output_file}.")
