import openai
from openai import OpenAI
from datasets import load_dataset, Dataset, Features, Value, Sequence, Array2D, Array3D

base_url=""
api_key=""

import datasets

dataset_path = '/path/arxiv_latest'

ds = datasets.load_dataset('/path/arxiv_latest', revision = '2023-08-20')

def split_text_into_hierarchy(text):
    import re

    lines = text.strip().split('\n')
    hierarchy = []
    stack = []

    heading_pattern = re.compile(r'^(§(?:\.\§)*)\s*(.*)$')

    current_content = ''
    for line in lines:
        line = line.rstrip()
        heading_match = heading_pattern.match(line)
        if heading_match:
            if current_content:
                if stack:
                    stack[-1]['content'] += current_content.strip() + '\n'
                current_content = ''

            marker = heading_match.group(1)
            title = heading_match.group(2).strip()

            level = marker.count('§')

            node = {
                'title': title,
                'content': '',
                'subsections': []
            }

            while stack and len(stack) >= level:
                stack.pop()
            if stack:
                stack[-1]['subsections'].append(node)
            else:
                hierarchy.append(node)
            stack.append(node)
        else:
            current_content += line + '\n'

    if current_content:
        if stack:
            stack[-1]['content'] += current_content.strip() + '\n'

    return hierarchy

def process_sections(hierarchy):
    previous_summaries = []
    all_instructions = []

    outputs = {
        'section_summaries': [],
        'section_instructions': [],
        'general_summary': ''
    }

    for idx, section in enumerate(hierarchy, 1):
        section_title = section.get('title', '')
        section_content = section.get('content', '')
        subsections = section.get('subsections', [])
        num_subsections = len(subsections)

        section_summary = generate_section_summary(
            section_title=section_title,
            section_content=section_content,
        )
        previous_summaries.append(f"Summary of Section {idx} - {section_title}:\n{section_summary}\n")
        outputs['section_summaries'].append({
            'section_number': idx,
            'section_title': section_title,
            'summary': section_summary
        })

        instruction = generate_instruction(
            section_title=section_title,
            section_content=section_content,
            summaries=previous_summaries,
            num_subsections=num_subsections,
            section_number=idx
        )
        outputs['section_instructions'].append({
            'section_number': idx,
            'section_title': section_title,
            'instruction': instruction
        })

    general_summary = generate_general_summary(previous_summaries )
    outputs['general_summary'] = general_summary

    return outputs

def generate_instruction(section_title, section_content, summaries, num_subsections, section_number):
    import math

    word_count = len(section_content.split())
    rounded_word_count = int(math.ceil(word_count / 100.0)) * 100
    current_summary = summaries[-1] 
    previous_summaries = summaries[:-1]

    instruction = f"""Instruction for Section {section_number}:

- Generate a section with the instruction {current_summary} for a scientific paper.
- Approximate word count: {rounded_word_count} words.
"""
    if num_subsections > 0:
        instruction += f"- This section contains {num_subsections} subsections.\n"
    else:
        instruction += "- This section does not contain subsections.\n"

    instruction += """- Follow these paper writing rules:
  - Use formal academic language.
  - Ensure clarity and conciseness.
  - Include relevant citations and references where appropriate.
  - Follow the standard structure for scientific papers.
  - Avoid plagiarism by writing in your own words.

"""

    if previous_summaries:
        instruction += "- Summaries of previous sections to consider:\n"
        instruction += "\n".join(previous_summaries) + "\n"

    instruction += "Please generate the section accordingly."

    return instruction

def generate_section_summary(section_title, section_content, ):
    prompt = f"""
You are tasked with providing a formal and concise summary of the following section from a scientific paper. The description must be clear, objective, and in third-person perspective, adhering to academic writing standards.

Requirements:
1. Use formal academic language.
2. Be concise but ensure that key points are captured.
3. Write in third-person perspective, as if describing the content for someone else.
4. The summary should not include any personal opinions or subjective language.

Section Title: {section_title}

Section Content:
{section_content}

Summary:
"""
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[ {"role": "user",  "content": prompt} ],
            stream=False,
            max_tokens=150,
            temperature=0.6,
            top_p=0.9,
            n=1,
            stop=None,
        )
        summary = response.choices[0].message.content.strip() 
        return summary
    except openai.error.OpenAIError as e:
        print(f"An error occurred while generating section summary: {e}")
        return ""

def generate_general_summary(previous_summaries ):
    combined_summaries = "\n".join(previous_summaries)

    prompt = f"""
You are asked to write a general descriptions of a scientific paper based on the descriptions of its sections provided below. The summary should be approximately 400 words and should capture the main objectives, methods, results, and conclusions of the paper.

Section Summaries:
{combined_summaries}

General Summary:
"""
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[ {"role": "user",  "content": prompt} ],
            stream=False,
            temperature=0.6,
            top_p=0.9,
            max_tokens=600,
            n=1,
            stop=None,
        )
        summary = response.choices[0].message.content.strip() 
        return summary
    except openai.error.OpenAIError as e:
        print(f"An error occurred while generating section summary: {e}")
        return ""

def split_into_sections(text):
    sections = {}
    raw_sections = [sec.strip() for sec in text.strip().split('§') if sec.strip()]
    for raw_section in raw_sections:
        section_lines = raw_section.split('\n', 1)
        section_title = section_lines[0].strip()
        section_content = section_lines[1].strip() if len(section_lines) > 1 else ''
        sections[section_title] = {'': section_content}
    return sections

def process_item(item):
    text = item.get('text', '')

    section_summaries = []
    section_instructions = []
    general_summary = ""

    if text.strip():
        hierarchy = split_text_into_hierarchy(text)

        outputs = process_sections(hierarchy)

        section_summaries = outputs.get('section_summaries', [])
        section_instructions = outputs.get('section_instructions', [])
        general_summary = outputs.get('general_summary', "")

    return {
        'text': text,
        'section_summaries': section_summaries,
        'section_instructions': section_instructions,
        'general_summary': general_summary
    }

data = ds['train'].select(range(2))

processed_data = data.map(
    process_item,
    batched=False,
    desc="Processing dataset"
)

import json

data_list = [dict(item) for item in processed_data]
print(data_list[0]['section_summaries'])

output_dataset_path = './single_round/'

output_json_path = './processed_data.json'

with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)

data.save_to_disk(output_dataset_path)
