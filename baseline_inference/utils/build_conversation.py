

def build_conversation(history={}, prompt='', system_prompt=''):
    conversation = []
    if system_prompt:
        conversation.append({'role': 'system', 'content': system_prompt})
    for idx, message in history.items():
        conversation.append({'role': 'user', 'content': message['prompt']})
        conversation.append({'role': 'assistant', 'content': message['response']})
    conversation.append({'role': 'user', 'content': prompt})
    return conversation

# Example usage
if __name__ == '__main__':
    history = {}
    system_prompt = 'You are a helpful assistant'
    # history[0] = {'prompt': 'What is the capital of France?', 'response': 'Paris'}
    # history[1] = {'prompt': 'What is the capital of Germany?', 'response': 'Berlin'}
    # history[2] = {'prompt': 'What is the capital of Italy?', 'response': 'Rome'}
    # history[3] = {'prompt': 'What is the capital of Spain?', 'response': 'Madrid'}
    # history[4] = {'prompt': 'What is the capital of Portugal?', 'response': 'Lisbon'}
    # history[5] = {'prompt': 'What is the capital of Switzerland?', 'response': 'Bern'}
    # history[6] = {'prompt': 'What is the capital of Austria?', 'response': 'Vienna'}
    # history[7] = {'prompt': 'What is the capital of Belgium?', 'response': 'Brussels'}
    print(build_conversation(history, 'What is the capital of Belgium?', system_prompt))
    
