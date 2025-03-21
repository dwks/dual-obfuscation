import pandas as pd
import torch
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
    print(f"Testing prompt {i+1}/{len(prompts_df)}")
    reply = test_llamaguard(f"User: {prompt}")
    results.append({'prompt': prompt, 'response': reply})
    print(f"{results[-1]}")

# Convert to DataFrame and save results
results_df = pd.DataFrame(results)
results_df.to_csv('llamaguard3_harmbench_results.csv', index=False)

print("Testing complete. Results saved to llamaguard3_harmbench_results.csv")
