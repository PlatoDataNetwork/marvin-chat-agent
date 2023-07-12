import numpy as np
import pandas as pd
import requests as r
import regex as re
from bs4 import BeautifulSoup
from dateutil import parser
import streamlit as st


# Define RssReader class
class RssReader:
    def __init__(self, rss_urls):
        self.rss_urls = rss_urls

    def date_time_parser(self, dt):
        return int(np.round((dt.now(dt.tz) - dt).total_seconds() / 60, 0))

    def elapsed_time_str(self, mins):
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

    def text_clean(self, desc):
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

    def src_parse(self, rss):
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

    def rss_parser(self, i):
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
        title = self.text_clean(title)  # cleaning title

        url = (
            "" if b1.find("link") is None else b1.find("link").get_text()
        )  # If <link> is absent then url = "". url is the URL of the news article

        desc = (
            "" if b1.find("description") is None else b1.find("description").get_text()
        )  # If <description> is absent then desc = "". desc is the short description of the news article
        desc = self.text_clean(desc)  # cleaning the description
        desc = (
            f"{desc[:300]}..." if len(desc) >= 300 else desc
        )  # limiting the length of description to 300 chars

        # If <pubDate> i.e. published date is absent then date is some random date 11 years ago so the the article appears at the end
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

    def news_agg(self, rss):
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
            rss_df = pd.concat(
                [rss_df, self.rss_parser(i).copy()]
            )  # parsing each news item (<item>)
        rss_df["description"] = rss_df["description"].replace(
            [" NULL", ""], np.nan
        )  # Few items have 'NULL' as description so replacing NULL with NA
        rss_df.dropna(
            inplace=True
        )  # dropping news items with either of title, URL, description or date, missing
        rss_df["src"] = self.src_parse(rss)  # extracting the source name from RSS feed URL
        rss_df["elapsed_time"] = rss_df["parsed_date"].apply(
            self.date_time_parser
        )  # Computing the time elapsed (in minutes) since the news was published
        rss_df["elapsed_time_str"] = rss_df["elapsed_time"].apply(
            self.elapsed_time_str
        )  # Converting the the time elapsed (in minutes) since the news was published into string format
        return rss_df

    @st.cache_data
    def load_feeds(_self):
        final_df = (
            pd.DataFrame()
        )  # initializing the data frame to store all the news items from all the RSS Feed URLs

        if final_df.empty:
            for i in _self.rss_urls:
                final_df = pd.concat([final_df, _self.news_agg(i)])

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
