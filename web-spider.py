#  Copyright (c) 18.04.2024 [D. P.] aka duskop; after the call a day after from a IPO-agency from Japan, i'm adding my patreon ID: https://www.patreon.com/florkz_com
#  All rights reserved.

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import urllib.parse
import os

class Crawler:
    def __init__(self, crawl_depth=2, relevance_threshold=5):
        self.crawl_depth = crawl_depth
        self.classifier = MultinomialNB()
        self.vectorizer = CountVectorizer()
        self.relevance_keywords = ['AI', 'LLM']
        self.irrelevance_keywords = ['paid']
        self.classifications = {}
        self.relevance_threshold = relevance_threshold
        self.base_url = None
        self.max_relevance_links = []
        self.visited_urls = set()  # Set to store visited URLs

    def fetch_url(self, url_or_curl):
        try:
            if url_or_curl.startswith("curl"):
                self.url, self.method, self.data, self.headers, self.cookie_dict = self.__parse_uncurl(url_or_curl)
                response = self.__fetch_url(self.url, method=self.method, data=self.data, headers=self.headers, cookies=self.cookie_dict)
            else:
                response = self.__fetch_url(url_or_curl)
            if response.status_code == 200:
                self.base_url = response.url  # Update base URL after fetching
                html = response.text
                text = self.extract_text(html)
                classification = self.classify_with_keyword_count(text)
                if classification > 0:  # Only add to visited set if relevant
                    self.visited_urls.add(url_or_curl)
                return html, classification
        except Exception as e:
            print(f"Failed to fetch URL: {url_or_curl}, Error: {e}")
        return None, 0  # Return (None, 0) when URL fetch fails

    def extract_text(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def train_classifier(self, X, y):
        X_train = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_train, y)

    def classify_with_keyword_count(self, text):
        if text is None:  # Check if text is None (failed to fetch)
            return 0
        keyword_count = sum(text.lower().count(keyword.lower()) for keyword in self.relevance_keywords)
        if keyword_count >= self.relevance_threshold:
            return 2  # More relevant
        elif keyword_count > 0:
            return 1  # Relevant
        else:
            return 0  # Irrelevant

    def crawl(self, url, depth=0):
        if depth > self.crawl_depth:
            return

        html, classification = self.fetch_url(url)
        if html is not None:
            self.classifications[url] = classification

            print(f"URL: {url}, Classification: {classification}")

            # Store links with maximum relevance
            if classification == 2:
                self.max_relevance_links.append(url)
                self.save_max_relevance_links()  # Save the link immediately

            # Process links recursively
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                next_url = urllib.parse.urljoin(self.base_url, link['href'])
                if next_url not in self.visited_urls:
                    self.crawl(next_url, depth + 1)

    def save_max_relevance_links(self, filename='max_relevance_links.txt'):
        with open(filename, 'w') as f:
            for link in self.max_relevance_links:
                f.write(link + '\n')

    def __fetch_url(self, url, method='GET', data=None, headers=None, cookies=None):
        if method == 'GET':
            response = requests.get(url, headers=headers, cookies=cookies)
        elif method == 'POST':
            response = requests.post(url, headers=headers, data=data, cookies=cookies)
        return response

    def __parse_uncurl(self, url_or_curl):
        from webgather.utils import uncurl
        if url_or_curl.startswith("curl"):
            return uncurl.parse(url_or_curl)

if __name__ == "__main__":
    # Sample URLs
    seed_urls = ['https://www.zdnet.com/article/best-ai-chatbot/']

    # Initialize crawler
    crawler = Crawler(crawl_depth=10, relevance_threshold=5)

    # Fetch and preprocess text data for training
    X_train = []
    for url in seed_urls:
        html, classification = crawler.fetch_url(url)
        if html:
            text = crawler.extract_text(html)
            X_train.append(text)

    # Check if X_train is empty
    if not X_train:
        print("No valid data fetched for training.")
    else:
        # Train classifier with labeled data
        y_train = [1] * len(X_train)  # All URLs in seed_urls are considered relevant
        crawler.train_classifier(X_train, y_train)

        # Start crawling
        for url in seed_urls:
            crawler.crawl(url)

        # Save links with maximum relevance to a file
        crawler.save_max_relevance_links()
