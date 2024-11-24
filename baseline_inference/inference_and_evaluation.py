from utils import load_model, load_json, json_save
import argparse, json
from tqdm import tqdm
import os

def load_response_json(response):
    results = response
    # print(results)
    # results = results.split("```json")[1]

    results = results.replace('json', '')
    results = results.replace('\n', '')
    results = results.replace("```", '')

    if '##' in results:
        results = results.split('#')[0]
    elif '**' in results:
        results = results.split('*')[0]


    results = results.strip()

    # print(results)
    # input()

    step_data = json.loads(results)
    return step_data

def generate_response(model, tokenizer, prompt, system_prompt = None):
    # prompt = "Give me a short introduction to large language model."
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
            # {"role": "user", "content": f"Give you a text: {text} \n\n please help me generate the outline (no more than 1 sentences) of this paragraph."}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# def load_data(dataset_name):
#     if dataset_name == 'paper':
#         data = load_json('./data/paper.json')
#     elif dataset_name == 'blog':
#         data = load_json('./data/blog.json')

#     headlines = data['headlines']
#     length_requirement = data['length_requirement']
#     text = data['text']
#     title = data['title']

#     experiment_resules = data['experiment_resules']

#     results_recording = []
#     for key in experiment_resules:
#         r = experiment_resules[key]
#         results_recording.append(f' {key}: {r} ')
#     results_recording = '\n\n'.join(results_recording)

#     return data, headlines, length_requirement, text, title, results_recording

def generate_long_text(model, tokenizer, headlines, length_requirement):
    context = []
    length_b = []

    # print(len(headlines))

    for h, l in tqdm(zip(headlines, length_requirement)):

        if context:
            if dataset_name == 'paper' and 'Introduction' in h:
                prompt = f'Title: {title} \n\n  Context: {context} \n\n please generate a Introduction that is more than {l} words based on the given bullet point: {h}, the title, and the context.' #You can also use the content in the table1-3 which is related with the bullet point: {h}
            else:
                prompt = f'Title: {title} \n\n  Context: {context} \n\n please continually generate a section that is more than {l} words based on the given bullet point: {h}, the title, and the context.' #You can also use the content in the table1-3 which is related with the bullet point: {h}
        else:
            if dataset_name == 'paper':
                prompt = f'Title: {title} \n\n  please generate a abstract that is more than {l} words based on the given bullet point: {h}, and the title.' # You can also use the content in the table1-3 which is related with the bullet point: {h}
            else:
                prompt = f'Title: {title} \n\n  please generate a section that is more than {l} words based on the given bullet point: {h}, and the title.' # You can also use the content in the table1-3 which is related with the bullet point: {h}
        p = generate_response(model, tokenizer, prompt)
        context.append(p)

        # length_b.append(len(p.split(' ')) / len(l))
        length_b.append(len(p.split(' ')) / l )
    
    return context, length_b

def evaluate_response(model, tokenizer, title, context, dimensions, sections, headlines):
    context_str = '\n\n'.join(context)
    score_responses = {}
    for level in dimensions:
        if level == 'passage-level':
            for d in dimensions[level]:
                system_prompt = """You are an expert AI assistant and you need to generate a score of the paper from a given dimension, and you just need to give me score and do not need to generate explanation.

                                Example of a valid JSON response:
                                ```json
                                {
                                    "score": your_score
                                }```
                                """
        
                d_text = dimensions[level][d]
                if d == 'Instruction-following-sec':
                    evaluate_prompt = f"Give you the title of the paper: {title} \n\n Text: {context_str}. \n\n Instruciton: {headlines} \n\n {d_text}"
                else:
                    evaluate_prompt = f"Give you the title of the paper: {title} \n\n Text: {context_str}. \n\n {d_text}"
                score_response = generate_response(model, tokenizer, evaluate_prompt, system_prompt)
                score_response = load_response_json(score_response)
                score_responses[d] = score_response['score']

        elif level == 'section-level':
            for i, d in enumerate(dimensions[level]):
                system_prompt = """You are an expert AI assistant and you need to generate a score of the section from a given dimension, and you just need to give me score and do not need to generate explanation.

                                Example of a valid JSON response:
                                ```json
                                {
                                    "score": your_score
                                }```
                                """
                d_text = dimensions[level][d]


                if dataset_name == 'paper_hf':
                    Abstract = context[0]
                    Introduction = context[1]
                    rest_context = '\n\n'.join(context[2:])
                else:
                    Introduction = context[0]
                    rest_context = '\n\n'.join(context[1:])

                if d == 'Instruction-following-sec':
                    evaluate_prompt = f"Give you the title of the paper: {title}, Introduction: {headlines[i]}, Main Body: {rest_context} \n\n {d_text}"
                    score_response = generate_response(model, tokenizer, evaluate_prompt, system_prompt)
                    score_response = load_response_json(score_response)
                    score_responses[d] = score_response['score']

                elif d == 'Abstract-Introduction':
                    evaluate_prompt = f"Give you the title of the paper: {title}, Introduction: {Introduction}, Abstract: {Abstract} \n\n {d_text}"
                    score_response = generate_response(model, tokenizer, evaluate_prompt, system_prompt)
                    score_response = load_response_json(score_response)
                    score_responses[d] = score_response['score']

                elif d == 'Introduction-Passage':
                    instrct_follo_s = []
                    for s, h in zip(sections, headlines):
                        evaluate_prompt = f"Give you the title of the paper: {title}, Headline: {h}, Section: {s} \n\n {d_text}"
                        score_response = generate_response(model, tokenizer, evaluate_prompt, system_prompt)
                        score_response = load_response_json(score_response)
                        instrct_follo_s.append(score_response['score'])
                    score_responses[d] = sum(instrct_follo_s) / len(instrct_follo_s)
                elif d ==  'Ref-Compare':
                    instrct_follo_s = []
                    for s, h in zip(sections, headlines):
                        evaluate_prompt = f"Give you the title of the paper: {title}, Headline: {h}, Section: {s} \n\n {d_text}"
                        score_response = generate_response(model, tokenizer, evaluate_prompt, system_prompt)
                        score_response = load_response_json(score_response)

                        if type(score_response) == type(1):
                            score_response = score_response
                        else:
                            score_response = score_response['score']

                        instrct_follo_s.append(score_response)
                    score_responses[d] = sum(instrct_follo_s) / len(instrct_follo_s)
                

                # : "Evaluate whether the original paper and the paper written based on the model are similar. Grade the paper (1-10 points).",
    
    return score_responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for long-text evaluation")

    parser.add_argument("--dataset_name" , type = str , default = 'paper_hf')
    parser.add_argument("--model_name" , type = str , default = 'Qwen2.5-7B')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name

    data_dir = f'./data/{dataset_name}/'

    fns = os.listdir(data_dir)
    model, tokenizer = load_model(model_name)
    for f in fns:
        data = load_json(f'{data_dir}{f}')

        print(data.keys())
        input()
        
        title = data['title']
        bullet_points = data['bullet_points']
        headlines = data['headlines']
        text = data['text']
        sections = data['sections']
        length_requirement = data['length_requirement']



        context, length_b = generate_long_text(model, tokenizer, headlines, length_requirement)

        data['result'] = '\n\n'.join(context)
        data['length_b'] = length_b
        # data['length_requirement'] = length_requirement

        if dataset_name == 'paper_hf':
            # dimensions = {
            #     'Facility': f'Evaluate whether there are factual descriptive errors in the paper, and grade it based on the extent of the errors (1-10 points).',
            #     'Comprehensive Analysis': 'Assess whether the description and analysis of the experiment in the paper are comprehensive. Grade the paper (1-10 points).',
            #     'Novelty': 'Evaluate whether the motivation of the paper is novel. Grade the paper (1-10 points).',
            #     'Semantic Coherence': 'Evaluate whether the paragraphs in the paper are semantically coherent. Grade the paper (1-10 points).',
            #     'Writing Style': 'Evaluate whether the writing style of the paper aligns with that of a research paper. Grade the paper (1-10 points).',
            #     'Reasonability of Structure': 'Evaluate whether the overall results of the paper are reasonable. Grade the paper (1-10 points).',
            #     'Explaining the principle clearly': 'Please help me determine if this text clearly explains the principles it aims to explain',
            #     'Readability': 'Help me determine if the language in this article is relatively simple and suitable for science popularization',
            # }
            dimensions = {
                'section-level': {
                    'Instruction-following-sec': 'Evaluate whether the content of this section meets the key points required in the headline, and grade it based on the extent of the errors (1-10 points).',
                    'Abstract-Introduction': 'Evaluate whether the content in the Introduction corresponds to the content in the Abstract. Grade the paper (1-10 points).',
                    'Introduction-Passage': 'Evaluate whether the content in the Introduction corresponds to the rest content in the paper. Grade the paper (1-10 points).',
                    'Ref-Compare': "Evaluate whether the original paper and the paper written based on the model are similar. Grade the paper (1-10 points).",
                },
                'passage-level': {
                    # 'Novelty': 'Evaluate whether the motivation of the paper is novel. Grade the paper (1-10 points).',
                    'Motivation': 'Evaluate whether the explanation in the Motivation section of the paper is sufficient to persuade readers to recognize the value of this work. Grade the paper (1-10 points).',
                    "Metiond": 'Evaluate whether the proposed method in the paper is novel and whether it intuitively addresses the problem effectively. and grade it based on the extent of the errors (1-10 points).',
                    'Writing Style': 'Evaluate whether the writing style of the paper aligns with that of a research paper. Grade the paper (1-10 points).',
                    # 'Comprehensive Analysis': 'Assess whether the description and analysis of the experiment in the paper are comprehensive. Grade the paper (1-10 points).',
                    'Semantic Coherence': 'Evaluate whether the paragraphs in the paper are semantically coherent. Grade the paper (1-10 points).',
                    # 'Facility': 'Evaluate whether there are factual descriptive errors in the paper, and grade it based on the extent of the errors (1-10 points).',
                    'Reasonability of Structure': 'Evaluate whether the overall results of the paper are reasonable. Grade the paper (1-10 points).',
                    # 'Explaining the principle clearly': 'Please help me determine if this text clearly explains the principles it aims to explain',
                    # 'Readability': 'Help me determine if the language in this article is relatively simple and suitable for science popularization',
                    'Redundancy': 'Evaluate whether the paper has repetitive content. The more repetitive the content, the lower the score. Grade the paper (1-10 points).',
                    'Instruction-following-passage': 'Evaluate the overall correspondence between the generated text and the instruction. Grade the paper (1-10 points).'
                },
            }
        else:
            dimensions = {

            }

        score_responses = evaluate_response(model, tokenizer, title, context, dimensions, sections, headlines)
        golden_scores = evaluate_response(model, tokenizer, title, sections, dimensions, sections, headlines)

        data['score_responses'] = score_responses
        data['golden_scores'] = golden_scores

        data = json_save(data, f'{data_dir}{f}')

        # print(score_responses)
        # print(golden_scores)
        # input()
