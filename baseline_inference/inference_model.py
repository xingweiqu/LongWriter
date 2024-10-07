from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper

def load_model(model_name, model_args, use_accel=False, max_tokens=1024):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if 'DeepSeek-V2' in model_name:
            model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, max_model_len=8192, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        else:
            model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        model_components['model_name'] = model_name
    else:
        model_components['use_accel'] = False
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        model_components['model_name'] = model_name
    return model_components

def infer(prompts, historys, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    max_tokens = kwargs.get('max_tokens', 1024)
    

    # if isinstance(prompts[0], str):
    #     messages = [build_conversation(history, prompt) for history, prompt in zip(historys, prompts)]
    # else:
    #     raise ValueError("Invalid prompts format")
    # messages =  
    if not historys:
        historys = [{}] * len(prompts)
        messages = [build_conversation(history, prompt) for history, prompt in zip(historys, prompts)]
    
    max_tokens = 1024
    
    if use_accel:
        if model_name == 'yi-large':
            prompt_token_ids = [[tokenizer.convert_tokens_to_ids("<|startoftext|>")] + tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
        else:
            prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
        stop_token_ids=[tokenizer.eos_token_id]
        if 'Meta-Llama-3' in model_name:
            stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        sampling_params = SamplingParams(max_tokens=max_tokens, stop_token_ids=stop_token_ids)
        outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)
    else:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, truncation=True, return_dict=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        responses = []
        for i, prompt in enumerate(prompts):
            response = tokenizer.decode(outputs[i, len(inputs['input_ids'][i]):], skip_special_tokens=True)
            responses.append(response)

    return responses


if __name__ == '__main__':
    # messages = [{'role': 'user', 'content': "中国水利水电第十三工程局有限公司单位隶属是什么？ 还有单位性质是什么？单位经济类型？"}]
    
    prompts = [
        '''中国水利水电第十三工程局有限公司单位隶属是什么？ 还有单位性质是什么？单位经济类型？''',
        '''中国水利水电第十三工程局有限公司单位隶属是什么？ 还有单位性质是什么？单位经济类型？''',
    ]
    model_args = {
        'model_path_or_name': '/gpfs/public/01/models/hf_models/Meta-Llama-3.1-8B-Instruct',
        'model_type': 'local',
        'tp': 8
    }
    model_components = load_model("Meta-Llama-3.1-8B-Instruct", model_args, use_accel=True, max_tokens=1024)
    # model_components = {"model": None, "chat_template": get_chat_template_from_config('')}
    responses = infer(prompts, None, **model_components)
    for response in responses:
        print(response)
