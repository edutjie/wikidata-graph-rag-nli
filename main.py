import streamlit as st
import os

from wikidata_rag import WikidataGraphRAG

DEVICE = "cpu"
HF_TOKEN = st.secrets["hf_token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

st.title("🔎 Knowledge Graph Open NLI LLM-based System - Wikidata")

with st.expander("See Question Examples"):
    st.write(
        """
        - Mosque, with countries
        - Hourses, limitted to 5
        - Picture of a cat
        - Number of humans
        - Nationality of Shawn Mendes
    """
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can answer your question based on Wikidata. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    search_agent = WikidataGraphRAG(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        device=DEVICE,
        hf_token=HF_TOKEN,
    )
    with st.chat_message("assistant"):
        question = [msg for msg in st.session_state.messages if msg["role"] == "user"][
            -1
        ]["content"]
        try:
            response = search_agent.chat(question, verbose=1)
        except Exception as e:
            print(e)
            response = f"Sorry, I couldn't find an answer to your question. Please try again with another question."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
