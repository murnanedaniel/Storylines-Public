# Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from newspaper import Article
import openai

import nest_asyncio
import aiohttp
import asyncio

# import common types
from typing import List

def parse_articles(article_list):
    for article in article_list:
        try:
            article.download()
            article.parse()
        except: 
            print(f"{article} could not be downloaded")
            article_list.remove(article)
    return article_list

def interpret_articles(article_list):
    for i, article in enumerate(article_list):
        try:
            article.nlp()
            if not article.summary:
                print(f"{i, article.title} could not be summarized")
                article_list.remove(article)
        except: 
            print(f"{i, article.title} could not be processed")
            article_list.remove(article)
        
    return article_list

def embed_articles(articles: List[Article]) -> List[List[float]]:
    return [get_embedding(article.summary) for article in articles]

def get_embedding(text: str, engine="text-similarity-babbage-001") -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    embedding = openai.Embedding.create(input=text, engine=engine)["data"][0]["embedding"]

    return np.array(embedding)

def get_topic(summaries = "", engine="babbage"):

    prompt = summaries.replace("\n", " ") + ". What is the topic of this text in less than five words?"

    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=25
        )

async def get(keywords, recency, api_key):

    api_host = "google-search3.p.rapidapi.com"
    google_url = "https://google-search3.p.rapidapi.com/api/v1/news/"

    query = google_url + "q=" + keywords + "+when:" + str(recency) + "d"

    headers = {"x-rapidapi-key": api_key,
                "x-rapidapi-host": api_host}

    async with aiohttp.ClientSession() as session:
        async with session.get(query, headers=headers) as response:
            html = await response.json()
            # print(html)
            return html

def get_news_results(keywords, recency=180):

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    coroutines = [get(keywords, recency)]

    results = loop.run_until_complete(asyncio.gather(*coroutines))

    return results[0]["entries"]