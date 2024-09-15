# crawling-pre-filter
a pre-filter to fetch sites depending on key-words, which the seed-URL(s) have
Description by ChatGPT (Version from 240913)

The main advantages of the crawl-script you’re working on are related to its ability to perform customized, targeted web scraping and classification of web content. Here are the key benefits of this crawler:

### 1. **Customizable Classification Logic:**
   - **Keyword-based Content Classification:** The script checks both relevant and irrelevant keywords in the page headers (like `<h1>`, `<h2>`, etc.) to classify pages. This allows the crawler to filter out content that doesn't match your needs or target specific types of information.
   - **Multiple Classification Levels:** Based on the keywords found and other content structures (like `<pre>` tags for code), the script assigns different classification scores, helping in prioritizing or filtering content further.

### 2. **Handling Technical Content:**
   - **Pre-tag Detection:** The script specifically checks for the presence of `<pre>` tags, which are often used to enclose code snippets. This makes it particularly useful for scraping technical blogs, tutorials, or documentation pages that contain code, allowing you to focus on pages with relevant technical content.
   
### 3. **Efficient Web Crawling and Link Traversal:**
   - **Recursively Follows Links:** The script is capable of extracting and following outlinks on a page to crawl a broader set of web pages, thus increasing the coverage and efficiency of gathering content across a website.
   - **Maximizes Relevance:** By scoring and saving only the most relevant links (`self.max_relevance_links`), the crawler optimizes the process of gathering important content while ignoring irrelevant or less important pages.

### 4. **Automated Content Collection:**
   - **Automates Web Page Fetching:** This automation reduces the manual effort of visiting web pages, especially for collecting structured data from multiple sources.
   - **Handles Errors Gracefully:** The script handles exceptions like connection issues, timeouts, or unexpected responses, allowing it to keep running without crashing.

### 5. **Flexible URL Input:**
   - **Multiple Starting URLs:** You can provide multiple starting URLs, allowing the crawler to visit several different websites or web sections in parallel, rather than being restricted to a single site.

### 6. **Performance Optimizations:**
   - **Session Management:** It uses session handling with `requests.Session()`, which reuses the same TCP connection across requests to the same site, improving the speed and efficiency of crawling.
   - **Keyword and Header Prioritization:** By focusing on headers (`<h1>`, `<h2>`, etc.) and specific tags (like `<pre>`), the script reduces unnecessary parsing of the entire HTML content, which can save time and computational resources.

### 7. **Adaptability to Different Scenarios:**
   - **Can be Extended for More Complex Scenarios:** If needed, the script can be adapted to include more complex classification criteria, handle different HTML structures, or interact with web APIs.
   - **Support for Different Types of Content:** With minor modifications, it could handle not just technical content like code but also other types of information depending on the use case (e.g., research articles, product reviews, etc.).

### 8. **Supports Scalable Data Gathering:**
   - **Link Accumulation and Visit Tracking:** The crawler maintains a set of visited URLs and relevant links, preventing redundant crawling and helping build a structured repository of the most useful web pages.
  
In summary, the script is highly customizable and useful for gathering, filtering, and classifying large amounts of web content, particularly for technical or content-specific scraping tasks. It’s designed for efficiency, with built-in performance optimizations, error handling, and adaptability to different use cases.

** **
The provided code defines two classes, `SimpleVectorizer` and `SimpleNaiveBayes`, which together implement a basic text classification pipeline using a bag-of-words model and a Naive Bayes classifier. Here's a breakdown of what each class does:

### `SimpleVectorizer`
This class is responsible for converting text data into numerical vectors that can be used by machine learning algorithms.

1. **`fit_transform(self, texts)`**:
   - **Purpose**: Converts a list of text documents into a matrix of token counts.
   - **Process**:
     - **Tokenization**: Splits each text into tokens (words) and counts their occurrences.
     - **Vocabulary Building**: Creates a vocabulary (a mapping of words to unique indices) from all tokens in the provided texts.
     - **Matrix Creation**: Constructs a document-term matrix `X`, where each row corresponds to a text document and each column corresponds to a token from the vocabulary. The value at each cell `(i, j)` represents the count of token `j` in document `i`.

2. **`transform(self, texts)`**:
   - **Purpose**: Transforms new text documents into the same vector space as created by `fit_transform`.
   - **Process**:
     - **Tokenization**: Similar to `fit_transform`, but this is for transforming new data.
     - **Matrix Creation**: Constructs a matrix `X` based on the existing vocabulary, where each row represents a new text document, and columns represent the frequency of each token from the vocabulary in the new documents.

### `SimpleNaiveBayes`
This class implements a Naive Bayes classifier, a probabilistic classifier based on Bayes' theorem with the assumption of independence between features.

1. **`fit(self, X, y)`**:
   - **Purpose**: Trains the Naive Bayes classifier on the provided data.
   - **Process**:
     - **Classes**: Identifies the unique classes/labels in the target variable `y`.
     - **Class Prior Probabilities**: Computes the prior probability of each class (how often each class appears in the training data).
     - **Feature Probabilities**: Computes the probability of each feature (word) given each class. It uses Laplace smoothing to handle cases where a word might not appear in the training data for a particular class.

2. **`predict(self, X)`**:
   - **Purpose**: Predicts the class labels for new data.
   - **Process**:
     - **Log Probabilities Calculation**: Computes the log-probability of each class given the features in the new data.
     - **Class Prediction**: Determines the class with the highest log-probability for each document.

### Summary
- **`SimpleVectorizer`** converts text documents into numerical feature vectors using a bag-of-words approach.
- **`SimpleNaiveBayes`** uses these feature vectors to train a Naive Bayes classifier and then predicts the class labels for new documents based on learned probabilities.

This implementation is a basic approach to text classification and is useful for understanding how text can be vectorized and classified with simple models.

** **
In the context of implementing a basic text classification pipeline using a Bag-of-Words (BoW) model, a large language model (LLM) isn't specifically used in the code snippet you provided. Instead, the focus is on traditional machine learning techniques. Here’s a breakdown of how text classification is performed in this setup and how predictions are made:

### Text Classification Pipeline with Bag-of-Words

1. **Vectorization (Bag-of-Words Model)**:
   - **Tokenization**: Split the text into individual words (tokens).
   - **Vocabulary Creation**: Build a vocabulary of unique words from the training data.
   - **Feature Extraction**: Convert each text into a numerical vector where each element represents the frequency of a word from the vocabulary in the document.

   This is done using the `SimpleVectorizer` class:

   ```python
   class SimpleVectorizer:
       def fit_transform(self, texts):
           # Tokenize texts and build vocabulary
           ...
           return X  # Document-term matrix

       def transform(self, texts):
           # Convert new texts to feature vectors based on the existing vocabulary
           ...
           return X
   ```

2. **Training (Naive Bayes Classifier)**:
   - **Calculate Class Probabilities**: Determine the prior probability of each class based on the training labels.
   - **Calculate Feature Probabilities**: Estimate the likelihood of each word given a class.

   This is done using the `SimpleNaiveBayes` class:

   ```python
   class SimpleNaiveBayes:
       def fit(self, X, y):
           # Calculate prior probabilities and feature probabilities
           ...
       
       def predict(self, X):
           # Compute log-probabilities for each class and select the class with the highest probability
           ...
           return predictions
   ```

### Making Predictions

**How Predictions Are Made:**

1. **Transform New Texts**: Convert new text documents into feature vectors using the same vocabulary as the training data. This ensures consistency between the training and prediction phases.

   ```python
   X_test = vectorizer.transform(new_texts)
   ```

2. **Compute Class Probabilities**:
   - For each class, compute the probability of the new text belonging to that class based on the feature probabilities learned during training.
   - The Naive Bayes classifier uses the log of probabilities for numerical stability.

   ```python
   def predict(self, X):
       log_prob = np.log(self.class_prior_) + X @ np.log(self.feature_prob_.T)
       return self.classes_[np.argmax(log_prob, axis=1)]
   ```

   - `np.log(self.class_prior_)` is the log of the prior probability of each class.
   - `X @ np.log(self.feature_prob_.T)` calculates the log of the likelihood of the features given each class.
   - `np.argmax(log_prob, axis=1)` selects the class with the highest log-probability for each document.

**Example**:
Suppose you have a new document `"I enjoy coding in Python"`, and you've already trained your model. Here’s how it processes the new document:

1. **Vectorization**:
   Convert `"I enjoy coding in Python"` into a feature vector using the fitted `SimpleVectorizer`.

2. **Prediction**:
   - Compute the log-probabilities for each class using the feature vector and the learned probabilities.
   - The class with the highest log-probability is selected as the prediction.

**No LLM Involvement**:
In this pipeline, a large language model (LLM) like GPT-3 or BERT is not used. The pipeline relies on traditional machine learning methods: a Bag-of-Words model for feature extraction and a Naive Bayes classifier for prediction. LLMs typically involve more complex models and embeddings for text representation and are not used in this basic approach.

### Summary

- **Vectorization**: Converts text into numerical feature vectors using a Bag-of-Words approach.
- **Training**: Calculates probabilities for each class based on the feature vectors.
- **Prediction**: Uses the trained model to determine the class of new text documents based on computed probabilities.

This basic pipeline provides a foundational approach to text classification without leveraging advanced LLMs.

** **

# Web Crawler Script Summary

## Overview
This script implements a web crawler designed to navigate websites, extract information, and classify pages based on relevance. It's built with a focus on efficiency, ethical crawling practices, and scalability.

## Key Components

1. **Crawler Class**
   - Main class that orchestrates the crawling process
   - Manages URL fetching, content extraction, and classification
   - Implements caching and domain banning mechanisms

2. **SimpleVectorizer Class**
   - Converts text data into numerical vectors for machine learning purposes
   - Uses a bag-of-words approach with term frequency

3. **SimpleNaiveBayes Class**
   - Implements a basic Naive Bayes classifier for page relevance prediction

4. **URL Fetching and Parsing**
   - Utilizes `requests` library for HTTP requests
   - Implements retry logic and connection pooling
   - Uses `BeautifulSoup` for HTML parsing

5. **Robots.txt Compliance**
   - `can_fetch` method to check and respect robots.txt rules

6. **Concurrency**
   - Uses `ThreadPoolExecutor` for parallel processing of URLs

7. **Caching and Resource Management**
   - Implements an in-memory cache for fetched URLs
   - Uses `psutil` to determine optimal number of worker threads

8. **Logging and Error Handling**
   - Comprehensive logging throughout the script
   - Try-except blocks for robust error handling

9. **Classification System**
   - Combines keyword-based and machine learning approaches
   - Classifies pages based on relevance to specified topics

10. **Main Execution Logic**
    - Initializes the crawler with seed URLs
    - Trains the classifier on initial data
    - Executes the crawling process

## Key Features
- Ethical crawling with robots.txt compliance
- Scalable design with concurrent processing
- Adaptive resource utilization
- Combination of rule-based and ML-based classification
- Robust error handling and logging

## Usage
The script is designed to be run as a standalone program. It starts with a set of seed URLs, trains a classifier on their content, and then crawls the web based on the learned patterns and specified rules.
