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


def check_password():
    """Returns `True` if the user had a correct password."""
    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        login_form()
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        login_form()
        return False
    else:
        # Password correct.
        # st.write("logged in (from check_password)")
        return True


def login_form():
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False
            del st.session_state["password"]

    with st.form(key="login"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.form_submit_button("login", on_click=password_entered)


def main_task():
    st.title("Managment Console")
    task_workflow = "Manage workflows"
    task_keywords = "Manage Meta Keywords"
    task_methods = "Manage all types of methods"
    radio = st.radio("Select a task to preform", [task_workflow, task_methods, task_keywords])
    if radio == task_workflow:
        workflow_form()
    elif radio == task_methods:
        method_form()
    elif radio == task_keywords:
        keyword_form()


def workflow_form():
    st.subheader("Manage workflows")

    if "workflow_save" not in st.session_state:
        edit_workflow_form()
        return False
    elif not st.session_state["workflow_save"]:
        edit_workflow_form()
        return False
    else:
        st.session_state["workflow_save"] = False
        return True


def edit_workflow_form():
    def run_workflow():
        get_extract()
        get_keywords()
        get_cluster_method()
        prepare_prompts()
        st.write(st.session_state)
        st.session_state["workflow_save"] = True

    def refresh_topics():
        topic_selections_option = get_topics_option()
        return topic_selections_option

    def load_workflow():
        if "selected_workflow" not in st.session_state:
            return False
        elif not st.session_state["selected_workflow"]:
            return False
        else:
            get_workflow()
            # st.session_state["selected_workflow"] = False
            return True

    # with st.form(key='workflow_form'):
    saved_workflows = get_saved_workflows_options()
    st.selectbox(
        "Select saved workflow",
        [workflow for workflow in saved_workflows],
        key="selected_workflow",
        on_change=load_workflow,
    )

    st.select_slider("Pick Time frame", ["30Min", "1H", "3H", "12H", "1D", "3D"], key="time_limit")
    category_options = get_category_options()

    if st.checkbox(
        "Limit categories", key="limit_categories", help="go horizontally or limit categories"
    ):
        st.multiselect(
            "Pick categories", [cat for cat in category_options], key="selected_categories"
        )
    if st.checkbox(
        "Filter MetaKeywords",
        key="limit_keywords",
        help="Limit only to keywords within selected MetaKeywords",
    ):
        metakeyword_option = get_metakeyword_option()
        st.multiselect(
            "Pick MetaKeywords",
            [metakey for metakey in metakeyword_option],
            key="selected_meta_keywords",
        )
    if st.checkbox(
        "Extract and filter topics",
        key="extract_topics",
        help="Limit only to keywords within selected MetaKeywords",
    ):
        method_topics_option = get_method_topics_option()
        st.selectbox(
            "Select extract method",
            [method for method in method_topics_option],
            key="selected_extract_method",
            on_change=refresh_topics,
        )
        topic_selections_option = get_topics_option()
        st.multiselect(
            "Select topics to include",
            [topic for topic in topic_selections_option],
            key="selected_topics",
        )
    st.number_input("Select number of clusters", 6, 20, key="number_of_clusters")
    if st.checkbox(
        "Find reasoning for clusters", key="reason_clusters", help="Request reasons for clusters"
    ):
        reason_cluster_method = get_cluster_method_option()
        st.selectbox(
            "Select reasoning method",
            [reason for reason in reason_cluster_method],
            key="selected_reason_method",
        )

    st.button("Continue", on_click=run_workflow)


def get_workflow():
    if st.session_state["selected_workflow"] == "Podcaster":
        st.session_state["limit_categories"] = True
        st.session_state["limit_keywords"] = True
        st.session_state["extract_topics"] = True
        st.session_state["reason_clusters"] = True
        st.session_state["selected_categories"] = get_category_options()
        st.session_state["selected_extract_method"] = "Podcast Classification"
        st.session_state["selected_meta_keywords"] = ["CryptoCurrency"]
        st.session_state["selected_topics"] = ["Exciting News", "Positive News"]
        st.session_state["selected_reason_method"] = "Clusters are from multiple categories"
        st.session_state["time_limit"] = "3H"
    elif st.session_state["selected_workflow"] == "Financial Advisor":
        st.session_state["limit_categories"] = True
        st.session_state["limit_keywords"] = True
        st.session_state["extract_topics"] = True
        st.session_state["reason_clusters"] = True
        st.session_state["selected_categories"] = ["Press Releasse"]
        st.session_state["selected_extract_method"] = "Financial classification"
        st.session_state["selected_meta_keywords"] = ["CryptoCurrency", "PR"]
        st.session_state["selected_topics"] = ["Financial Alert", "Financial Opportunity"]
        st.session_state["selected_reason_method"] = "Clusters are from single category"
        st.session_state["time_limit"] = "30Min"


def prepare_prompts():
    if "topic_extract_prompt" in st.session_state and "topic_response_prompt" in st.session_state:
        st.session_state["topic_prompt"] = (
            st.session_state["topic_extract_prompt"]
            + " "
            + st.session_state["topic_response_prompt"]
        )
        del st.session_state["topic_extract_prompt"]
        del st.session_state["topic_response_prompt"]
    if (
        "cluster_reason_prompt" in st.session_state
        and "cluster_response_prompt" in st.session_state
    ):
        st.session_state["reason_prompt"] = (
            st.session_state["cluster_reason_prompt"]
            + " "
            + st.session_state["cluster_response_prompt"]
        )
        del st.session_state["cluster_reason_prompt"]
        del st.session_state["cluster_response_prompt"]


def get_extract():
    if st.session_state["selected_extract_method"] == "Podcast Classification":
        st.session_state[
            "topic_extract_prompt"
        ] = """You are a helpful assistant that helps retrieve topics talked about in a podcast transcript
- Your goal is to extract the topic names and brief 1-sentence description of the topic
- Topics include:
  - Exciting news
  - Positive news
  - Interesting Stories
  - Alarming news
"""
        st.session_state[
            "topic_response_prompt"
        ] = """- Provide a brief description of the topics after the topic name. Example: 'Topic: Brief Description'
- Use the same words and terminology that is said in the podcast
- Do not respond with anything outside of the podcast. If you don't see any topics, say, 'No Topics'
- Do not respond with numbers, just bullet points
"""

    elif st.session_state["selected_extract_method"] == "Financial classification":
        st.session_state[
            "topic_extract_prompt"
        ] = """You are a helpful assistant that helps retrieve topics talked about in a rss news feed
- Your goal is to extract the topic names and brief 1-sentence description of the topic
- Topics include:
  - Financial Alert
  - Business Ideas
  - Financial Opportunity
  - Money making businesses
  - New Cooperation
  - Advice or words of caution
"""
        st.session_state[
            "topic_response_prompt"
        ] = """- Provide a brief description of the topics after the topic name. Example: 'Topic: Brief Description'
- Do not respond with anything outside of the news item. If you don't see any topics, say, 'No Topics'
- Do not respond with numbers, just bullet points
"""


def get_cluster_method():
    if st.session_state["selected_reason_method"] == "Clusters are from single category":
        st.session_state[
            "cluster_reason_prompt"
        ] = """You are a helpful assistant that helps reason clusters of news in a single topic from embedded vectors
- Your goal is to find a reason for the cluster that will be given to you.
"""
        st.session_state[
            "cluster_response_prompt"
        ] = """- Provide a description of the reason for this cluster and name it
- Do not respond with numbers, just bullet points
"""

    elif st.session_state["selected_reason_method"] == "Clusters are from multiple categories":
        st.session_state[
            "cluster_reason_prompt"
        ] = """You are a helpful assistant that helps reason clusters of news in a multiple topics from embedded vectors
- Your goal is to find a reason for the cluster that will be given to you.
"""
        st.session_state[
            "cluster_response_prompt"
        ] = """- Provide a description of the reason for this cluster and name it
- Do not respond with numbers, just bullet points
"""


def get_keywords():
    keywords = []
    if "selected_meta_keywords" in st.session_state:
        if "CryptoCurrency" in st.session_state["selected_meta_keywords"]:
            keywords.extend(["Bitcoin", "Litecoin", "Dogecoin", "Ethereum"])
        if "PR" in st.session_state["selected_meta_keywords"]:
            keywords.extend(["Press Release", "announcement", "media release", "press statement"])
    st.session_state["selected_keywords"] = keywords


def get_saved_extract_options():
    # get category options from db or query
    podcaster = "Podcaster"
    financial_advisor = "Financial Advisor"
    options = [podcaster, financial_advisor]
    return options


def get_saved_workflows_options():
    # get category options from db or query
    podcaster = "Podcaster"
    financial_advisor = "Financial Advisor"
    options = [podcaster, financial_advisor]
    return options


def get_cluster_method_option():
    # get ,saved method options from db or query
    multiple_categories = "Clusters are from multiple categories"
    single_category = "Clusters are from single category"

    options = [multiple_categories, single_category]
    return options


def get_method_topics_option():
    # get ,saved method options from db or query
    podcast_news = "Podcast Classification"
    financial_advisor = "Financial classification"

    options = [podcast_news, financial_advisor]
    return options


def get_topics_option():
    # get ,metakeywords options from db or query
    fa_alart = "Financial Alert"
    fa_coop = "New Cooperation"
    fa_opportunity = "Financial Opportunity"
    pc_exciting_news = "Exciting News"
    pc_good_news = "Positive News"
    pc_alarming_news = "Alarming News"

    if "selected_extract_method" not in st.session_state:
        return False
    elif not st.session_state["selected_extract_method"]:
        return False
    else:
        method = st.session_state["selected_extract_method"]
    if method == "Podcast Classification":
        options = [pc_exciting_news, pc_good_news, pc_alarming_news]
    elif method == "Financial classification":
        options = [fa_alart, fa_opportunity, fa_coop]
    return options


def get_metakeyword_option():
    # get ,metakeywords options from db or query

    blockchain = "CryptoCurrency"
    press_releasse = "PR"
    options = [blockchain, press_releasse]
    return options


def get_category_options():
    # get category options from db or query
    blockchain = "Blockchain"
    press_releasse = "Press Releasse"
    options = [blockchain, press_releasse]
    return options


def method_form():
    st.subheader("Methods and prompts")
    method_extract = "Topic extraction methods"
    method_detection = "Cluster reason detection methods"
    radio = st.radio("Which type of method to manage", [method_extract, method_detection])
    if radio == method_extract:
        if "loaded_reason" in st.session_state:
            del st.session_state["loaded_reason"]
        topic_extract_form()
    elif radio == method_detection:
        if "loaded_extract" in st.session_state:
            del st.session_state["loaded_extract"]
        cluster_reason_detection_method_form()


def topic_extract_form():
    st.subheader("Topic extraction method")
    # if "loaded_extract" in st.session_state:
    #     del st.session_state["loaded_extract"]
    if "extract_save" not in st.session_state:
        topic_extract_form_edit()
        return False
    elif not st.session_state["extract_save"]:
        topic_extract_form_edit()
        return False
    else:
        st.session_state["extract_save"] = False
        return True


def cluster_reason_form_edit():
    def save_reason():
        st.write(st.session_state)
        st.session_state["reason_save"] = True

    def load_reasons():
        if "selected_reason_method" not in st.session_state:
            return False
        elif not st.session_state["selected_reason_method"]:
            return False
        else:
            get_cluster_method()
            st.session_state["loaded_reason"] = True
            return True

    cluster_reason_option = get_cluster_method_option()
    st.selectbox(
        "Select Cluster Reason method",
        [method for method in cluster_reason_option],
        key="selected_reason_method",
        on_change=load_reasons,
    )
    st.text_area("Cluster prompt", key="cluster_reason_prompt", height=240)
    st.text_area("Response prompt", key="cluster_response_prompt", height=140)
    st.button("Continue", on_click=save_reason)


def topic_extract_form_edit():
    def save_topic():
        st.write(st.session_state)
        st.session_state["extract_save"] = True

    def load_extract():
        if "selected_extract_method" not in st.session_state:
            return False
        elif not st.session_state["selected_extract_method"]:
            return False
        else:
            get_extract()
            st.session_state["loaded_extract"] = True
            return True

    method_topics_option = get_method_topics_option()
    st.selectbox(
        "Select extract method",
        [method for method in method_topics_option],
        key="selected_extract_method",
        on_change=load_extract,
    )
    st.text_area("Extract prompt", key="topic_extract_prompt", height=240)
    st.text_area("Context prompt", key="topic_response_prompt", height=140)
    st.button("Continue", on_click=save_topic)


def cluster_reason_detection_method_form():
    st.subheader("Cluster reason method")
    if "reason_save" not in st.session_state:
        cluster_reason_form_edit()
        return False
    elif not st.session_state["reason_save"]:
        cluster_reason_form_edit()
        return False
    else:
        st.session_state["reason_save"] = False
        return True


def keyword_form():
    st.subheader("Meta Keywords")
    if "keyword_save" not in st.session_state:
        keyword_edit_form()
        return False
    elif not st.session_state["keyword_save"]:
        keyword_edit_form()
        return False
    else:
        st.session_state["keyword_save"] = False
        return True


def keyword_edit_form():
    def save_metakeyword():
        st.write(st.session_state)
        st.session_state["keyword_save"] = True

    def load_keyword():
        if "selected_meta_keywords" not in st.session_state:
            return False
        elif not st.session_state["selected_meta_keywords"]:
            return False
        else:
            get_keywords()
            st.session_state["loaded_keyword"] = True
            return True

    meta_keyword_option = get_metakeyword_option()
    keywords = [
        "Press Release",
        "announcement",
        "media release",
        "press statement",
        "Bitcoin",
        "Litecoin",
        "Dogecoin",
        "Ethereum",
    ]
    st.selectbox(
        "Select MetaKeyword",
        [meta for meta in meta_keyword_option],
        key="selected_meta_keywords",
        on_change=load_keyword,
    )
    st.multiselect("Select Keywords", [keyword for keyword in keywords], key="selected_keywords")
    st.button("Continue", on_click=save_metakeyword)


def demo_form():
    # col1, col2 = st.columns(2)

    with st.form(key="demo_form"):
        # button_status = st.button('Click me')
        # data_edit = st.data_editor('Edit data', data)
        checkbox = st.checkbox("I agree")
        radio = st.radio("Pick one", ["cats", "dogs"])
        selectbox = st.selectbox("Pick one", ["cats", "dogs"])
        multi = st.multiselect("Buy", ["milk", "apples", "potatoes"])

        slider = st.slider("Pick a number", 0, 100)
        select_slider = st.select_slider("Pick a size", ["S", "M", "L"])
        # st.text_input('First name')
        number = st.number_input("Pick a number", 0, 10)
        text_area = st.text_area("Text to translate")
        date_input = st.date_input("Your birthday")
        time = st.time_input("Meeting time")
        file = st.file_uploader("Upload a CSV")
        # st.download_button('Download file', data)
        # st.camera_input("Take a picture")
        color = st.color_picker("Pick a color")
        username = st.text_input("Username")
        password = st.text_input("Password")

        st.form_submit_button("Login")
