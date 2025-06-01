from transformers import pipeline

pipe = pipeline("text-generation", model="deepseek-ai/deepseek-llm-7b-base", trust_remote_code=True)
prompt = "<|user|>\nWho are you?\n<|assistant|>\n"

response = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
print(response[0]['generated_text'])

