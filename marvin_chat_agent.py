from langchain.llms import OpenAI
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Vectara
import streamlit as st
import numpy as np
import langchain

langchain.verbose = False
import os

# loading environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.header("Marvin AI")

VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_CORPUS_ID = os.getenv("VECTARA_CORPUS_ID")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initializing Vectara
vectorstore = Vectara(
    vectara_customer_id=VECTARA_CUSTOMER_ID
    if VECTARA_CUSTOMER_ID
    else st.secrets["VECTARA_CUSTOMER_ID"],
    vectara_corpus_id=VECTARA_CORPUS_ID if VECTARA_CORPUS_ID else st.secrets["VECTARA_CORPUS_ID"],
    vectara_api_key=VECTARA_API_KEY if VECTARA_API_KEY else st.secrets["VECTARA_API_KEY"],
)

# chat completion llm
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    streaming=True,
    openai_api_key=OPENAI_API_KEY if OPENAI_API_KEY else st.secrets["OPENAI_API_KEY"],
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

# retrieval qa chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

tools = [
    Tool(
        name="Vectara News",
        func=qa.run,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about the topic"
        ),
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate",
    memory=conversational_memory,
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # stream_handler = StreamHandler(st.empty())
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
