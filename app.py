import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")  # Removed .cuda()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("DeepSeek R1 Chatbot")

user_input = st.text_input("You:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        full_prompt = "".join([f"<|user|>{u}<|assistant|>{a}" for u, a in st.session_state.history])
        full_prompt += f"<|user|>{user_input}<|assistant|>"

        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

        st.session_state.history.append((user_input, response))

for user, bot in st.session_state.history:
    st.markdown(f"**You**: {user}")
    st.markdown(f"**Bot**: {bot}")
