# Imports
import numpy as np
from tqdm import tqdm

from newspaper import Article
import openai

import nest_asyncio
import aiohttp
import asyncio

# import common types
from typing import List
import ipywidgets as widgets


def parse_articles(article_list):
    for article in tqdm(article_list):
        parsed_article = parse_article(article)
        if parsed_article is None:
            article_list.remove(article)
    return article_list

def parse_article(article):
    try:
        article.download()
        article.parse()
    except: 
        print(f"{article} could not be downloaded")
        return None
    return article

def interpret_articles(article_list):
    for article in tqdm(article_list):
        interpreted_article = interpret_article(article)
        if interpreted_article is None:
            article_list.remove(article)        
    return article_list

def interpret_article(article):
    try:
        article.nlp()
        if not article.summary:
            print(f"{article.title} could not be summarized")
            return None
    except: 
        print(f"{article.title} could not be processed")
        return None
    return article

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

def get_news_results(keywords, api_key, recency=180):

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    coroutines = [get(keywords, recency, api_key)]

    results = loop.run_until_complete(asyncio.gather(*coroutines))

    return results[0]["entries"]

def get_comparison(article_0, article_1, engine="text-davinci-002"):
    
    article_0_subset = " ".join(article_0["text"].replace("\n", " ").split(" ")[:300])
    article_1_subset = " ".join(article_1["text"].replace("\n", " ").split(" ")[:300])

    input_text = "Article 1: \n\n" + article_0_subset + "\n\nArticle 2: \n\n" + article_1_subset + "\n\nCompare the sentiment, context and political bias of these two articles."

    return openai.Completion.create(
        engine=engine,
        prompt=input_text,
        max_tokens=256,
        temperature=0.8,
    )


def get_representative(cluster_embeddings, cluster_labels, cluster_articles, target_label):
    cluster_mean = cluster_embeddings[cluster_labels == target_label].mean(axis=0)
    cluster_distances = np.sqrt(np.sum((cluster_embeddings - cluster_mean)**2, axis=1))
    cluster_representative_article = cluster_articles[np.argmin(cluster_distances)]
    return cluster_representative_article

# def build_button(func, description_text=None, tooltip_text=None, output_text=None):
#     button = widgets.Button(description = description_text, tooltip = tooltip_text, layout = Layout(width = 'auto'))
#     output = widgets.Output()

#     def on_button_clicked(b):
#         with output:
#             func()
