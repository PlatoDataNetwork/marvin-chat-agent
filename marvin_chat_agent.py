import os
import langchain
from langchain.llms import OpenAI
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Vectara
from langchain.chains import LLMMathChain
from langchain.agents import Tool
import streamlit as st
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests as r
import regex as re
from dateutil import parser


langchain.verbose = False

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


st.set_page_config(
    page_title="Marvin by PlatoAI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


########################################
######## LIST OF RSS FEED URLs #########
########################################
rss = ["https://zephyrnet.com/artificial-intelligence/feed/"]


def date_time_parser(dt):
    """
    Returns the time elapsed (in minutes) since the news was published

    dt: str
        published date

    Returns
    int: time elapsed (in minutes)
    """
    return int(np.round((dt.now(dt.tz) - dt).total_seconds() / 60, 0))


def elapsed_time_str(mins):
    """
    Returns the word form of the time elapsed (in minutes) since the news was published

    mins: int
        time elapsed (in minutes)

    Returns
    str: word form of time elapsed (in minutes)
    """
    time_str = ""  # Initializing a variable that stores the word form of time
    hours = int(
        mins / 60
    )  # integer part of hours. Example: if time elapsed is 2.5 hours, then hours = 2
    days = np.round(mins / (60 * 24), 1)  # days elapsed
    # minutes portion of time elapsed in hours. Example: if time elapsed is 2.5 hours, then remaining_mins = 30
    remaining_mins = int(mins - (hours * 60))

    if days >= 1:
        time_str = f"{str(days)} days ago"  # Example: days = 1.2 => time_str = 1.2 days ago
        if days == 1:
            time_str = "a day ago"  # Example: days = 1 => time_str = a day ago

    elif (days < 1) & (hours < 24) & (mins >= 60):
        time_str = f"{str(hours)} hours and {str(remaining_mins)} mins ago"  # Example: 2 hours and 15 mins ago
        if (hours == 1) & (remaining_mins > 1):
            time_str = (
                f"an hour and {str(remaining_mins)} mins ago"  # Example: an hour and 5 mins ago
            )
        if (hours == 1) & (remaining_mins == 1):
            time_str = f"an hour and a min ago"  # Example: an hour and a min ago
        if (hours > 1) & (remaining_mins == 1):
            time_str = f"{str(hours)} hours and a min ago"  # Example: 5 hours and a min ago
        if (hours > 1) & (remaining_mins == 0):
            time_str = f"{str(hours)} hours ago"  # Example: 4 hours ago
        if ((mins / 60) == 1) & (remaining_mins == 0):
            time_str = "an hour ago"  # Example: an hour ago

    elif (days < 1) & (hours < 24) & (mins == 0):
        time_str = "Just in"  # if minutes == 0 then time_str = 'Just In'

    else:
        time_str = f"{str(mins)} minutes ago"  # Example: 5 minutes ago
        if mins == 1:
            time_str = "a minute ago"
    return time_str


def text_clean(desc):
    """
    Returns cleaned text by removing the unparsed HTML characters from a news item's description/title

    dt: str
        description/title of a news item

    Returns
    str: cleaned description/title of a news item
    """
    desc = desc.replace("&lt;", "<")
    desc = desc.replace("&gt;", ">")
    desc = re.sub("<.*?>", "", desc)  # Removing HTML tags from the description/title
    desc = desc.replace("#39;", "'")
    desc = desc.replace("&quot;", '"')
    desc = desc.replace("&nbsp;", '"')
    desc = desc.replace("#32;", " ")
    return desc


# Function to extract the source of the news from RSS feed’s URL
def src_parse(rss):
    """
    Returns the source (root domain of RSS feed) from the RSS feed URL.

    rss: str
         RSS feed URL

    Returns
    str: root domain of RSS feed URL
    """
    # RSS feed URL of NDTV profit (http://feeds.feedburner.com/ndtvprofit-latest?format=xml) doesn't contain NDTV's root domain
    if rss.find("ndtvprofit") >= 0:
        rss = "ndtv profit"
    rss = rss.replace("https://www.", "")  # removing "https://www." from RSS feed URL
    rss = rss.split("/")  # splitting the remaining portion of RSS feed URL by '/'
    return rss[0]  # first element/item of the split RSS feed URL is the root domain


# Function to process individual news item of an RSS feed
def rss_parser(i):
    """
    Processes an individual news item.

    i: bs4.element.Tag
       single news item (<item>) of an RSS Feed

    Returns
    DataFrame: data frame of a processed news item (title, url, description, date, parsed_date)
    """
    b1 = BeautifulSoup(
        str(i), features="xml"
    )  # Parsing a news item (<item>) to BeautifulSoup object

    title = (
        "" if b1.find("title") is None else b1.find("title").get_text()
    )  # If <title> is absent then title = ""
    title = text_clean(title)  # cleaning title

    url = (
        "" if b1.find("link") is None else b1.find("link").get_text()
    )  # If <link> is absent then url = "". url is the URL of the news article

    desc = (
        "" if b1.find("description") is None else b1.find("description").get_text()
    )  # If <description> is absent then desc = "". desc is the short description of the news article
    desc = text_clean(desc)  # cleaning the description
    desc = (
        f"{desc[:300]}..." if len(desc) >= 300 else desc
    )  # limiting the length of description to 300 chars

    # If <pubDate> i.e. published date is absent then date is some random date 11 yesrs ago so the the article appears at the end
    date = (
        "Sat, 12 Aug 2000 13:39:15 +0530"
        if b1.find("pubDate") is None
        else b1.find("pubDate").get_text()
    )

    if (
        url.find("businesstoday.in") >= 0
    ):  # Time zone in the feed of 'businesstoday.in' is wrong, hence, correcting it
        date = date.replace("GMT", "+0530")

    date1 = parser.parse(date)  # parsing the date to Timestamp object

    # data frame of the processed data
    return pd.DataFrame(
        {"title": title, "url": url, "description": desc, "date": date, "parsed_date": date1},
        index=[0],
    )


# Function to parse each RSS feed
def news_agg(rss):
    """
    Processes each RSS Feed URL passed as an input argument

    rss: str
         RSS feed URL

    Returns
    DataFrame: data frame of data processed from the passed RSS Feed URL
    """
    rss_df = pd.DataFrame()  # Initializing an empty data frame
    # Response from HTTP request
    resp = r.get(
        rss,
        headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
        },
    )
    b = BeautifulSoup(resp.content, features="xml")  # Parsing the HTTP response
    items = b.find_all("item")  # Storing all the news items
    for i in items:
        rss_df = pd.concat([rss_df, rss_parser(i).copy()])  # parsing each news item (<item>)
    rss_df["description"] = rss_df["description"].replace(
        [" NULL", ""], np.nan
    )  # Few items have 'NULL' as description so replacing NULL with NA
    rss_df.dropna(
        inplace=True
    )  # dropping news items with either of title, URL, description or date, missing
    rss_df["src"] = src_parse(rss)  # extracting the source name from RSS feed URL
    rss_df["elapsed_time"] = rss_df["parsed_date"].apply(
        date_time_parser
    )  # Computing the time elapsed (in minutes) since the news was published
    rss_df["elapsed_time_str"] = rss_df["elapsed_time"].apply(
        elapsed_time_str
    )  # Converting the the time elapsed (in minutes) since the news was published into string format
    return rss_df


@st.cache_data
def load_feeds():
    final_df = (
        pd.DataFrame()
    )  # initializing the data frame to store all the news items from all the RSS Feed URLs

    if final_df.empty:
        for i in rss:
            final_df = pd.concat([final_df, news_agg(i)])

    result_str = '<html><table style="border: none;"><tr style="border: none;"><td style="border: none; height: 10px;"></td></tr>'

    for n, i in final_df.iterrows():  # iterating through the search results
        href = i["url"]
        description = i["description"]
        url_txt = i["title"]

        result_str += (
            f'<a href="{href}" target="_blank" style="background-color: whitesmoke; display: block; height:100%; text-decoration: none; color: black; line-height: 1.2;">'
            + f'<tr style="align:justify; border-left: 5px solid transparent; border-top: 5px solid transparent; border-bottom: 5px solid transparent; font-weight: bold; font-size: 18px; background-color: whitesmoke;">{url_txt}</tr></a>'
            + f'<a href="{href}" target="_blank" style="background-color: whitesmoke; display: block; height:100%; text-decoration: none; color: dimgray; line-height: 1.25;">'
            + f'<tr style="align:justify; border-left: 5px solid transparent; border-top: 0px; border-bottom: 5px solid transparent; font-size: 14px; padding-bottom:5px;">{description}</tr></a>'
            + f'<a href="{href}" target="_blank" style="background-color: whitesmoke; display: block; height:100%; text-decoration: none; color: black;">'
            + f'<tr style="border: none;"><td style="border: none; height: 10px;"></td></tr>'
        )
    result_str += "</table></html>"

    return result_str


feed_items = load_feeds()


with st.sidebar:
    with st.container():
        st.markdown(feed_items, unsafe_allow_html=True)


st.title("Chat with Marvin about AI Intelligence")

VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_CORPUS_ID = os.getenv("VECTARA_CORPUS_ID")
CORPUS_ID_AI_TOOLS = os.getenv("CORPUS_ID_AI_TOOLS")
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "responses" not in st.session_state:
    st.session_state["responses"] = []

if "requests" not in st.session_state:
    st.session_state["requests"] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
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
]

llm_math = LLMMathChain(llm=llm)

# initialize the math tool
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful for when you need to answer questions about math.",
)

tools2 = load_tools(["arxiv"])

tools.append(tools2[0])
tools.append(math_tool)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    max_iterations=3,
    early_stopping_method="generate",
    memory=st.session_state.conversational_memory,
)

if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
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

    with st.chat_message("assistant"):
        st.markdown(response)

    # with st.chat_message(message["role"]):
    #    st.markdown(response)

    # st.write(response)
