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
from workflow_builder import check_password, main_task
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

if check_password():
    main_task()

# st.session_state
