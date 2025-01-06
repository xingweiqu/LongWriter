import json
import re
from datasets import load_dataset
from openai import OpenAI
import random
import os
import tqdm
import numpy as np
import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 文件为 Python 字典或列表

    return data

# Initialize OpenAI client
base_url = "http://180.184.175.69:3000/v1"
api_key = "sk-8l67qB1S9wZEZOo6A03eAb11191f4d22B99301F5A017D6D4"
client = OpenAI(base_url=base_url, api_key=api_key)

# dataset_path = "/map-vepfs/shuyue/xw_huggingface/parsed_arxiv/data"
# ds = load_dataset(dataset_path, revision="2023-08-20")
dataset_path  = './hf_blog.json'
data = load_json(dataset_path)
data_filter = []

for item in tqdm.tqdm(data):
    text = item['text']

    text_split = text.replace('\n',' ')
    text_split= text_split.split(' ')
    text_split = [t for t in text_split if t != '']
    data_filter.append(item)

    # if len(text_split) > 200:
    #     if '\n\u00a7 introduction' in text.lower()  :
    #         data_filter.append(item)
ds = data_filter

# Helper functions
def extract_abstract(text):
    abstract_pattern = r"(?i)(?<=\nAbstract\n)(.+?)(?=\n\n|\u00a7|\Z)"
    abstract_match = re.search(abstract_pattern, text, re.DOTALL)
    if abstract_match:
        return abstract_match.group(1).strip()
    return ""


def split_text_into_hierarchy(text):
    heading_pattern = re.compile(r"^(\u00a7(?:\.\u00a7)*)\s*(.*)$")
    lines = text.strip().split("\n")

    hierarchy = []
    stack = []
    current_content = ""

    abstract = extract_abstract(text)
    if abstract:
        hierarchy.append({"title": "Abstract", "content": abstract, "subsections": []})

    for line in lines:
        line = line.rstrip()
        match = heading_pattern.match(line)
        if match:
            if current_content:
                if stack:
                    stack[-1]["content"] += current_content.strip() + "\n"
                current_content = ""

            marker = match.group(1)
            title = match.group(2).strip()
            level = marker.count("\u00a7")

            node = {"title": title, "content": "", "subsections": []}

            while stack and len(stack) >= level:
                stack.pop()
            if stack:
                stack[-1]["subsections"].append(node)
            else:
                hierarchy.append(node)
            stack.append(node)
        else:
            current_content += line + "\n"

    if current_content and stack:
        stack[-1]["content"] += current_content.strip() + "\n"

    return hierarchy


def merge_hierarchy_to_markdown(hierarchy, level=1):
    merged_text = ""
    for node in hierarchy:
        heading_marker = "#" * level
        merged_text += f"{heading_marker} {node['title']}\n\n"
        if node["content"].strip():
            merged_text += node["content"].strip() + "\n\n"
        if node["subsections"]:
            merged_text += merge_hierarchy_to_markdown(node["subsections"], level + 1)
    return merged_text


def generate_summary(prompt):
    try:
        response = client.chat.completions.create(
            model="claude-3-5-sonnet-20240620",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=500,
            temperature=0.6,
            top_p=0.9,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error occurred while generating."


def calculate_word_count(text, acc=100):
    return round(len(text.split()) / acc) * acc


def generate_prompts(hierarchy, paper_title, total_word_count):
    whole_text = merge_hierarchy_to_markdown(hierarchy)
    print('phase1')
    title_prompt = (
        f"Paper Title: {paper_title}\n"
        f"Context: {whole_text}\n"
        f"Generate a command-based prompt that can guide a large language model (LLM) to write the content of a paper. This generated prompt should be based on the provided full content of the paper and should summarize the main purpose or goal of the paper in one or two sentence, avoiding any explicit templates, lists, or overly structured phrases. The goal is to produce a prompt that helps the LLM recreate or expand the paper content coherently and accurately."
    )
    title_summary = generate_summary(title_prompt)
    title_content = {
        "title": paper_title,
        "summary": title_summary,
        "target_length": calculate_word_count(whole_text),
        "origin_content": whole_text,
    }

    return {"title_prompt": title_content}


# Resume from saved file
output_file = "./generated_blog_dpo_500.json"
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        processed_data = json.load(f)
else:
    processed_data = []
# processed_data = []

# Get indices of already processed items
processed_indices = {item["title_prompt"]["title"] for item in processed_data}

# for item in tqdm.tqdm(ds):
#     print(item.get("text", ""))
#     input()

# Split text lengths into 50 bins
text_lengths = [calculate_word_count(item.get("text", "")) for item in ds]
print(text_lengths)
bins = np.linspace(min(text_lengths), max(text_lengths), 100)
digitized = np.digitize(text_lengths, bins) - 1

# Resume progress tracking
resume_state_file = "./resume_state_blog.json"
if os.path.exists(resume_state_file):
    with open(resume_state_file, "r", encoding="utf-8") as f:
        resume_state = json.load(f)
else:
    resume_state = {"bin_idx": 0, "sample_idx": 0}

# Process each bin
for bin_idx in tqdm.tqdm(range(resume_state["bin_idx"], 100)):
    bin_indices = [i for i, x in enumerate(digitized) if x == bin_idx]
    sampled_indices = random.sample(bin_indices, min(10, len(bin_indices)))

    for sample_idx in tqdm.tqdm(
        range(resume_state["sample_idx"], len(sampled_indices))
    ):
        idx = sampled_indices[sample_idx]
        item = ds[idx]
        text = item.get("text", "")
        title = item.get("title", "Untitled Paper")

        if not text.strip() or title in processed_indices:
            continue

        total_word_count = calculate_word_count(text)
        hierarchy = split_text_into_hierarchy(text)
        prompts = generate_prompts(hierarchy, title, total_word_count)

        # Append new data
        processed_data.append(prompts)

        # Save progress incrementally
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

        # Update and save resume state
        resume_state = {"bin_idx": bin_idx, "sample_idx": sample_idx + 1}
        with open(resume_state_file, "w", encoding="utf-8") as f:
            json.dump(resume_state, f, ensure_ascii=False)

    # Reset sample_idx when moving to the next bin
    resume_state["sample_idx"] = 0
    with open(resume_state_file, "w", encoding="utf-8") as f:
        json.dump(resume_state, f, ensure_ascii=False)

print(f"Prompts have been saved incrementally to: {output_file}")
