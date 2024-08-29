import streamlit as st
import os

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

from wikidata_rag import WikidataGraphRAG

DEVICE = "cpu"
HF_TOKEN = st.secrets["hf_token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

st.title("🔎 Knowledge Graph Open NLI LLM-based System - Wikidata")


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
        # response = search_agent.run(st.session_state.messages,
        #                             # callbacks=[st_cb]
        #                             )
        question = [msg for msg in st.session_state.messages if msg["role"] == "user"][
            -1
        ]["content"]
        response = search_agent.chat(question, verbose=1)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
