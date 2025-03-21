import pandas as pd
import torch
import codecs
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LlamaGuard 3 locally
model_name = "meta-llama/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

# Load HarmBench prompts
prompts_df = pd.read_csv('HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv')

prompts_df = prompts_df[:10]

# Function to test a prompt against LlamaGuard 3 locally
def test_llamaguard(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0,
        pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return reply

# Testing each prompt
results = []
for i, prompt in enumerate(prompts_df['Behavior']):
    print(f"Testing prompt {i+1}/{len(prompts_df)} (normal)")
    normal_reply = test_llamaguard(f"User: {prompt}\n\n")
    
    rot13_prompt = codecs.encode(prompt, 'rot_13')
    print(f"Testing prompt {i+1}/{len(prompts_df)} (rot13)")
    rot13_reply = test_llamaguard(f"User: {rot13_prompt}\n\n")

    normal_safe = True
    rot13_safe = True
    if 'unsafe' in normal_reply: normal_safe = False
    if 'unsafe' in rot13_reply: rot13_safe = False

    print({
        'prompt': prompt.encode('unicode_escape').decode('utf-8'),
        'normal_response': normal_reply.encode('unicode_escape').decode('utf-8'),
        'rot13_prompt': rot13_prompt.encode('unicode_escape').decode('utf-8'),
        'rot13_response': rot13_reply.encode('unicode_escape').decode('utf-8'),
    })
    results.append({
        'prompt': prompt.encode('unicode_escape').decode('utf-8'),
        'normal_response': normal_reply.encode('unicode_escape').decode('utf-8'),
        'rot13_response': rot13_reply.encode('unicode_escape').decode('utf-8'),
        'normal_safe': normal_safe,
        'rot13_safe': rot13_safe
    })


# Convert to DataFrame and save results
results_df = pd.DataFrame(results)
results_df.to_csv('llamaguard3_harmbench_results.csv', index=False)

print("Testing complete. Results saved to llamaguard3_harmbench_results.csv")
