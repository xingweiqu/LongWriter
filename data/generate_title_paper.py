import json
from openai import OpenAI

# Initialize OpenAI client
base_url = ""
api_key = ""
client = OpenAI(base_url=base_url, api_key=api_key)

# Example input JSON file (replace this with the actual file path)
input_file = "/map-vepfs/shuyue/xw_huggingface/LongWriter/data/title_prompts.json"

# Output file
output_file = "complete_title_paper.json"


# Function to generate summary
def generate_summary(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=10000,
            temperature=0.6,
            top_p=0.9,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error occurred while generating."


# Generate a single paper in Markdown format
def generate_single_paper(input_sections):
    paper_md = "# Complete Paper\n\n"
    sections_data = []

    for section in input_sections:
        # Build the section-specific prompt with meta-prompt
        section_prompt = (
            f"You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. "
            f"Write the {section['title']} section of the paper. "
            f"Focus on {section['summary']} in about {section['target_length']} words. "
            f"Use technical language suitable for a submission to a top-tier venue."
        )

        print(f"Generating content for: {section['title']}")
        print(section_prompt)
        content = generate_summary(section_prompt)

        # Add section to Markdown document
        paper_md += f"## {section['title']}\n\n"
        paper_md += content + "\n\n"

        # Add section data to the structure
        sections_data.append(
            {
                # "section_number": section["section_number"],
                "title": section["title"],
                "content": content,
            }
        )

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
with open(input_file, "r") as file:
    input_data = json.load(file)

# Generate all papers
all_papers_data = generate_all_papers(input_data)

# Save all papers to a JSON file
with open(output_file, "w") as file:
    json.dump(all_papers_data, file, indent=4)

print(f"All papers saved to {output_file}.")
