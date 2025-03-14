import json
import re
from datasets import load_dataset
from openai import OpenAI
import random
import json
import tqdm

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 文件为 Python 字典或列表

    return data

dataset_name = 'blog'

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

# Helper function: Extract abstract
def extract_abstract(text):
    # Define pattern to capture abstract
    abstract_pattern = r"(?i)(?<=\nAbstract\n)(.+?)(?=\n\n|\u00a7|\Z)"
    abstract_match = re.search(abstract_pattern, text, re.DOTALL)
    if abstract_match:
        return abstract_match.group(1).strip()
    return ""


# Helper function: Split text into a hierarchy of sections
def split_text_into_hierarchy(text):
    heading_pattern = re.compile(r"^(\u00a7(?:\.\u00a7)*)\s*(.*)$")
    lines = text.strip().split("\n")

    hierarchy = []
    stack = []
    current_content = ""

    # Extract abstract and prepend it as a special section
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
        # Add the heading marker based on the level
        heading_marker = "#" * level
        merged_text += f"{heading_marker} {node['title']}\n\n"

        # Add the content of the current node
        if node["content"].strip():
            merged_text += node["content"].strip() + "\n\n"

        # Recursively add subsections
        if node["subsections"]:
            merged_text += merge_hierarchy_to_markdown(node["subsections"], level + 1)

    return merged_text


# Function to generate GPT-4-O output
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


# Generate prompts
def generate_prompts(hierarchy, paper_title, total_word_count):
    # Meta and Title-based prompt
    meta_prompt = "This task involves analyzing and structuring blog into a clear and rigorous format for academic publishing."
    # title_prompt = f"Write a title or headline for the paper titled '{paper_title}' that captures its core contribution."
    whole_text = merge_hierarchy_to_markdown(hierarchy)
    title_prompt = (
        f"Paper Title: {paper_title}\n"
        f"Context: {whole_text}\n"
        f"Generate a command-based prompt that can guide a large language model (LLM) to write the content of a paper. This generated prompt should be based on the provided full content of the paper and should summarize the main purpose or goal of the paper in a single sentence, avoiding any explicit templates, lists, or overly structured phrases. The goal is to produce a prompt that helps the LLM recreate or expand the paper content coherently and accurately."
    )
    # print(title_prompt)
    title_summary = generate_summary(title_prompt)
    # print(title_summary)
    title_content = [
        {
            "title": paper_title,
            "summary": title_summary,
            "target_length": calculate_word_count(whole_text),
            "origin_content": whole_text,
        }
    ]

    # Section-based planned content
    planned_content = []
    for idx, section in enumerate(hierarchy, 1):
        section_prompt = (
            f"Paper Title: {paper_title}\n"
            f"Section Title: {section['title']}\n"
            f"Context: {section['content']}\n"
            f"Generate a concise and effective prompt that can guide a large language model (LLM) to write the content of a specific section of a paper. This generated prompt should be based on the provided full content of the paper and should summarize the main purpose or goal of the section in a single sentence, avoiding any explicit templates, lists, or overly structured phrases. The goal is to produce a prompt that helps the LLM recreate or expand the section content coherently and accurately."
        )
        summary = generate_summary(section_prompt)
        planned_content.append(
            {
                "section_number": idx,
                "title": section["title"],
                "summary": summary,
                "target_length": calculate_word_count(section["content"]),
                "origin_content": section["content"],
            }
        )

    # Section-based detailed content
    detailed_content = []
    for idx, section in enumerate(hierarchy, 1):
        detailed_prompt = (
            f"Paper Title: {paper_title}\n"
            f"Section Title: {section['title']}\n"
            f"Context: {section['content']}\n"
            f"Write a command-style prompt to guide the user in creating the content for this section. The generated prompt should direct the user to provide multiple technically detailed points addressing the key aspects of the section. The prompt must include a clear structure with steps or targeted questions to ensure the user generates content that is specific, technically accurate, and aligned with the context of the paper. Use the provided full section text as a reference for generating this prompt."
        )
        summary = generate_summary(detailed_prompt)
        detailed_content.append(
            {
                "section_number": idx,
                "title": section["title"],
                "summary": summary,
                "target_length": calculate_word_count(section["content"]),
                "origin_content": section["content"],
            }
        )

    # Combine outputs
    prompts = {
        "meta_prompt": meta_prompt,
        "title_prompt": title_content,
        "planned_prompt": planned_content,
        "detailed_prompt": detailed_content,
    }
    return prompts


# # Process dataset items
# def process_item(item):
#     text = item.get("text", "")
#     title = item.get("title", "Untitled Paper")

#     if not text.strip():
#         return {"title": title, "prompts": {}, "original_content": text}

#     hierarchy = split_text_into_hierarchy(text)
#     total_word_count = len(text.split())
#     prompts = generate_prompts(hierarchy, title, total_word_count)

#     return {"title": title, "prompts": prompts, "original_content": text}

import numpy as np

# Calculate text lengths
# text_lengths = [calculate_word_count(item.get("text", "")) for item in ds["train"]]
text_lengths = [calculate_word_count(item.get("text", "")) for item in ds]

# Split text lengths into 50 bins
bins = np.linspace(min(text_lengths), max(text_lengths), 51)
digitized = np.digitize(text_lengths, bins) - 1

# Sample 10 items from each bin
import tqdm

processed_data = []
for bin_idx in tqdm.tqdm(range(300)):
    bin_indices = [i for i, x in enumerate(digitized) if x == bin_idx]
    sampled_indices = random.sample(bin_indices, min(10, len(bin_indices)))
    for idx in tqdm.tqdm(sampled_indices):
        item = ds[idx]
        text = item.get("text", "")
        title = item.get("title", "Untitled Paper")
        total_word_count = calculate_word_count(text)
        hierarchy = split_text_into_hierarchy(text)
        prompts = generate_prompts(hierarchy, title, total_word_count)
        processed_data.append(prompts)

# Ensure total samples collected is 500 or less depending on available data
processed_data = processed_data[:1000]

# processed_data = []
# for item in ds["train"].select(range(10)):  # Adjust range to process more papers
#     text = item.get("text", "")
#     title = item.get("title", "Untitled Paper")
#     total_word_count = calculate_word_count(text)
#     hierarchy = split_text_into_hierarchy(text)

#     prompts = generate_prompts(hierarchy, title, total_word_count)
#     processed_data.append(prompts)

# Separate prompts for saving
title_prompts = [data["title_prompt"] for data in processed_data]
planned_prompts = [data["planned_prompt"] for data in processed_data]
detailed_prompts = [data["detailed_prompt"] for data in processed_data]

# Save prompts to separate files
with open(f"./{dataset_name}_title_prompts_all.json", "w", encoding="utf-8") as f:
    json.dump(title_prompts, f, ensure_ascii=False, indent=4)

with open(f"./{dataset_name}_planned_prompts.json", "w", encoding="utf-8") as f:
    json.dump(planned_prompts, f, ensure_ascii=False, indent=4)

with open(f"./{dataset_name}_detailed_prompts.json", "w", encoding="utf-8") as f:
    json.dump(detailed_prompts, f, ensure_ascii=False, indent=4)

print("Title, Planned, and Detailed Prompts saved successfully.")
