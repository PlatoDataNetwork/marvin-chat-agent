# Import necessary packages
import os
import numpy as np
import pandas as pd
import requests as r
import streamlit as st
import regex as re
from bs4 import BeautifulSoup
from dateutil import parser
from dotenv import load_dotenv, find_dotenv

# Import necessary modules from langchain
import langchain
from langchain.llms import OpenAI
import os
import langchain
from langchain.llms import OpenAI
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMMathChain
from langchain.vectorstores import Vectara
from rss_reader import RssReader
from PIL import Image

langchain.verbose = False

# loading environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title="Marvin by PlatoAI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

rss_reader = RssReader(["https://zephyrnet.com/artificial-intelligence/feed/"])
feed_items = rss_reader.load_feeds()

# feed_items = load_feeds()


with st.sidebar:
    aiTab, toolsTab = st.tabs(["AI News", "AI Tools"])
    with aiTab:
        with st.container():
            st.markdown(feed_items, unsafe_allow_html=True)


st.title("Chat with Marvin about AI Intelligence")

VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_CORPUS_ID = os.getenv("VECTARA_CORPUS_ID")
CORPUS_ID_AI_TOOLS = os.getenv("CORPUS_ID_AI_TOOLS")
CORPUS_ID_AI_PAPERS = os.getenv("CORPUS_ID_AI_PAPERS")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initializing Vectara
vectorstoreAINews = Vectara(
    vectara_customer_id=VECTARA_CUSTOMER_ID
    if VECTARA_CUSTOMER_ID
    else st.secrets["VECTARA_CUSTOMER_ID"],
    vectara_corpus_id=VECTARA_CORPUS_ID if VECTARA_CORPUS_ID else st.secrets["VECTARA_CORPUS_ID"],
    vectara_api_key=VECTARA_API_KEY if VECTARA_API_KEY else st.secrets["VECTARA_API_KEY"],
)

vectorstoreAITools = Vectara(
    vectara_customer_id=VECTARA_CUSTOMER_ID
    if VECTARA_CUSTOMER_ID
    else st.secrets["VECTARA_CUSTOMER_ID"],
    vectara_corpus_id=CORPUS_ID_AI_TOOLS
    if CORPUS_ID_AI_TOOLS
    else st.secrets["CORPUS_ID_AI_TOOLS"],
    vectara_api_key=VECTARA_API_KEY if VECTARA_API_KEY else st.secrets["VECTARA_API_KEY"],
)

vectorstoreAIPapers = Vectara(
    vectara_customer_id=VECTARA_CUSTOMER_ID
    if VECTARA_CUSTOMER_ID
    else st.secrets["VECTARA_CUSTOMER_ID"],
    vectara_corpus_id=CORPUS_ID_AI_PAPERS
    if CORPUS_ID_AI_PAPERS
    else st.secrets["CORPUS_ID_AI_PAPERS"],
    vectara_api_key=VECTARA_API_KEY if VECTARA_API_KEY else st.secrets["VECTARA_API_KEY"],
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "responses" not in st.session_state:
    st.session_state["responses"] = []

if "requests" not in st.session_state:
    st.session_state["requests"] = []

image = Image.open("PlatoLogo.png")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] is "assistant":
        with st.chat_message(message["role"], avatar=image):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# chat completion llm
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4",
    streaming=True,
    openai_api_key=OPENAI_API_KEY if OPENAI_API_KEY else st.secrets["OPENAI_API_KEY"],
)

# conversational memory
if "conversational_memory" not in st.session_state:
    st.session_state.conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True
    )

# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstoreAINews.as_retriever()
)

qaTools = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstoreAITools.as_retriever()
)

qaPapers = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstoreAIPapers.as_retriever()
)


tools = [
    Tool(
        name="PlatoAi",
        func=qa.run,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about anything related to artificial intelligence news and developments"
        ),
    ),
    Tool(
        name="AITools",
        func=qaTools.run,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about anything related to artifical intelligence tools, platforms, software, and services"
        ),
    ),
    Tool(
        name="AIPapers",
        func=qaPapers.run,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about anything related to artifical intelligence academic knowledge and published papers"
        ),
    ),
]

llm_math = LLMMathChain(llm=llm)

# initialize the math tool
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful for when you need to answer questions about math.",
)

# tools2 = load_tools(["arxiv"])

# tools.append(tools2[0])
tools.append(math_tool)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    max_iterations=3,
    early_stopping_method="generate",
    memory=st.session_state.conversational_memory,
)

if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant", avatar=image):
        st_callback = StreamlitCallbackHandler(st.container())

    try:
        response = agent.run(prompt, callbacks=[st_callback])
    except Exception as e:
        response = str(e)
        st.exception(e)
        if response.startswith("Could not parse LLM output: `"):
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        else:
            raise Exception(str(e))
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant", avatar=image):
        st.markdown(response)

    # with st.chat_message(message["role"]):
    #    st.markdown(response)

    # st.write(response)
