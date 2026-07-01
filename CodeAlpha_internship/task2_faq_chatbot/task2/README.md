# FAQ Chatbot

## Overview

A FAQ chatbot for technology and programming topics. It uses NLP techniques (tokenization, stopword removal, TF-IDF vectorization, and cosine similarity) to match user questions to the most relevant FAQ answer. No external NLP libraries are required. The project ships in two forms: a standalone web interface (index.html) and a Python terminal version (chatbot.py).

## Features

- 30 FAQs covering Python, machine learning, deep learning, NLP, APIs, Docker, Git, databases, SQL vs NoSQL, JavaScript, React, algorithms, data structures, blockchain, LLMs, and more
- NLP pipeline: tokenization, punctuation removal, stopword filtering, TF-IDF vectorization, cosine similarity matching
- Threshold-based fallback response when no good match is found
- Web UI with suggested question chips, chat bubbles, and keyboard support (Enter to send)
- Terminal version for command-line use

## NLP Approach

1. Tokenization: Text is lowercased and split into words after removing punctuation.
2. Stopword removal: Common English words that carry no meaning are filtered out.
3. TF-IDF: Term Frequency-Inverse Document Frequency is computed for both the FAQ corpus and each user query.
4. Cosine similarity: The query vector is compared to each FAQ vector. The FAQ with the highest similarity score above the threshold (0.05) is returned as the answer.
5. Fallback: If no FAQ meets the threshold, a default message is returned.

## Project Structure

  task2/
    index.html    - Browser-based chat UI (self-contained, no server needed)
    chatbot.py    - Python terminal chatbot (no dependencies beyond stdlib)
    README.md     - This file

## How to Run

### Option 1: Web UI (simplest, no installation needed)

1. Open a terminal in the task2 folder.
2. Open index.html directly in a browser by double-clicking it, or serve it:

   Python 3:
     python -m http.server 8001

   Then open: http://localhost:8001

3. Type your question in the input box and press Enter or click the send button.
4. Click the suggested question chips for quick demo queries.

### Option 2: Python Terminal Chatbot

Requirements: Python 3.6 or higher. No external packages needed (uses only Python standard library).

1. Open a terminal in the task2 folder.
2. Run:
     python chatbot.py

3. Type questions when prompted. Type "quit" or "exit" to stop.

Example session:
  You: What is machine learning?
  Bot: Machine learning is a branch of artificial intelligence...

  You: what is docker
  Bot: Docker is a platform for developing, shipping, and running applications...

  You: exit
  Bot: Goodbye!

## FAQ Topics Covered

- Python, JavaScript, React
- Machine learning, deep learning, neural networks, LSTM, overfitting
- NLP (natural language processing)
- APIs, REST, HTTP methods (GET vs POST)
- Docker, Git
- Databases, SQL vs NoSQL
- Algorithms, recursion, data structures, hash tables
- OOP, compilers
- Cloud computing
- Blockchain, large language models
- Conversational intents (hello, thank you, what can you do)

## Extending the FAQ

To add new FAQs, open chatbot.py (for Python) or index.html (for web UI) and add entries to the FAQ_DATA list in the format:
  {"question": "Your question here?", "answer": "The answer here."}

The NLP pipeline automatically includes new entries in the next run.
