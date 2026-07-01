import json
import math
import re
import string

# FAQ data - Technology / Programming topic
FAQ_DATA = [
    {
        "question": "What is Python?",
        "answer": "Python is a high-level, interpreted, general-purpose programming language known for its readability and simplicity. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a branch of artificial intelligence where systems learn from data to improve their performance on tasks without being explicitly programmed. It includes techniques like supervised learning, unsupervised learning, and reinforcement learning."
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "answer": "Supervised learning uses labeled training data where the model learns to map inputs to known outputs. Unsupervised learning uses unlabeled data and the model finds hidden patterns or structures on its own. Examples: supervised - classification, regression; unsupervised - clustering, dimensionality reduction."
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a computing system loosely inspired by biological neural networks in the brain. It consists of layers of interconnected nodes (neurons) that process data and learn patterns. Deep neural networks with many layers are the foundation of deep learning."
    },
    {
        "question": "What is deep learning?",
        "answer": "Deep learning is a subset of machine learning that uses neural networks with many layers (deep networks) to learn representations of data with multiple levels of abstraction. It has achieved state-of-the-art results in image recognition, natural language processing, and speech recognition."
    },
    {
        "question": "What is natural language processing?",
        "answer": "Natural Language Processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language. It includes tasks like text classification, sentiment analysis, machine translation, question answering, and text summarization."
    },
    {
        "question": "What is an API?",
        "answer": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. It defines methods and data formats for requesting and exchanging information between systems."
    },
    {
        "question": "What is the difference between GET and POST requests?",
        "answer": "GET requests retrieve data from a server and parameters are sent in the URL. POST requests send data to a server in the request body, typically to create or update resources. GET is idempotent and cacheable; POST is not."
    },
    {
        "question": "What is a RESTful API?",
        "answer": "A RESTful API follows the REST (Representational State Transfer) architectural style. It uses standard HTTP methods (GET, POST, PUT, DELETE), stateless communication, and resource-based URLs. REST APIs are widely used for web services."
    },
    {
        "question": "What is Docker?",
        "answer": "Docker is a platform for developing, shipping, and running applications in containers. Containers package code and its dependencies together so applications run consistently across different environments. Docker simplifies deployment and improves resource utilization."
    },
    {
        "question": "What is Git?",
        "answer": "Git is a distributed version control system that tracks changes in source code during software development. It allows multiple developers to collaborate, maintain history, create branches, and merge changes. GitHub, GitLab, and Bitbucket are popular hosting platforms for Git repositories."
    },
    {
        "question": "What is a database?",
        "answer": "A database is an organized collection of structured data stored electronically. Common types include relational databases (SQL) like MySQL, PostgreSQL which store data in tables, and NoSQL databases like MongoDB, Redis which store data in documents, key-value pairs, or graphs."
    },
    {
        "question": "What is the difference between SQL and NoSQL?",
        "answer": "SQL databases use structured tables with predefined schemas and support complex queries with joins. They are ACID-compliant and great for structured data. NoSQL databases are schema-flexible, horizontally scalable, and suited for unstructured or semi-structured data with high velocity."
    },
    {
        "question": "What is JavaScript?",
        "answer": "JavaScript is a lightweight, interpreted programming language primarily used to make web pages interactive. It runs in the browser and on the server (via Node.js). It is the core language of the web alongside HTML and CSS."
    },
    {
        "question": "What is React?",
        "answer": "React is a JavaScript library for building user interfaces, developed by Meta. It uses a component-based architecture and a virtual DOM for efficient rendering. React is widely used for single-page applications and can be used with Next.js for server-side rendering."
    },
    {
        "question": "What is an algorithm?",
        "answer": "An algorithm is a finite sequence of well-defined instructions or rules to solve a problem or perform a computation. Algorithms are evaluated by their time complexity (how fast they run) and space complexity (how much memory they use)."
    },
    {
        "question": "What is object-oriented programming?",
        "answer": "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects which contain data (attributes) and code (methods). Key principles are encapsulation, inheritance, polymorphism, and abstraction. Languages like Java, Python, and C++ support OOP."
    },
    {
        "question": "What is cloud computing?",
        "answer": "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, and analytics over the internet. Major providers are AWS, Microsoft Azure, and Google Cloud. It offers scalability, flexibility, and pay-as-you-go pricing."
    },
    {
        "question": "What is a compiler?",
        "answer": "A compiler is a program that translates source code written in a high-level programming language into machine code or an intermediate form. Compilation happens before execution. Examples: GCC for C/C++, javac for Java. This differs from interpreters which execute code line by line."
    },
    {
        "question": "What is recursion?",
        "answer": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. Every recursive function needs a base case to stop the recursion. Classic examples include factorial computation, Fibonacci sequence, and tree traversal."
    },
    {
        "question": "What is a data structure?",
        "answer": "A data structure is a way of organizing and storing data to enable efficient access and modification. Common data structures include arrays, linked lists, stacks, queues, trees, graphs, and hash tables. Choosing the right data structure is key to writing efficient algorithms."
    },
    {
        "question": "What is a hash table?",
        "answer": "A hash table (or hash map) is a data structure that stores key-value pairs. It uses a hash function to compute an index into an array of buckets. Average time complexity for insert, delete, and search is O(1). Hash tables are used in dictionaries, caches, and sets."
    },
    {
        "question": "What is the difference between a list and a tuple in Python?",
        "answer": "In Python, a list is mutable (can be changed after creation) and defined with square brackets []. A tuple is immutable (cannot be changed) and defined with parentheses (). Tuples are faster and can be used as dictionary keys; lists are more flexible."
    },
    {
        "question": "What is an LSTM?",
        "answer": "LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture designed to learn long-term dependencies in sequential data. It uses gates (input, forget, output) to control information flow, solving the vanishing gradient problem of standard RNNs."
    },
    {
        "question": "What is overfitting?",
        "answer": "Overfitting occurs when a machine learning model learns the training data too well, including noise, and performs poorly on unseen data. It is addressed by techniques like regularization (L1/L2), dropout, cross-validation, early stopping, and collecting more training data."
    },
    {
        "question": "Hello",
        "answer": "Hello! I am the Tech FAQ Chatbot. Ask me anything about programming, machine learning, databases, web development, and more. Type your question and I will find the best matching answer."
    },
    {
        "question": "Hi",
        "answer": "Hi there! I am here to answer your tech and programming questions. What would you like to know?"
    },
    {
        "question": "What can you do?",
        "answer": "I can answer frequently asked questions about programming, machine learning, databases, web development, algorithms, data structures, and general computer science topics. Type any question and I will find the closest matching answer from my knowledge base."
    },
    {
        "question": "Thank you",
        "answer": "You are welcome! Feel free to ask more questions anytime."
    },
    {
        "question": "What is blockchain?",
        "answer": "Blockchain is a distributed ledger technology that stores data in a chain of blocks, each containing a cryptographic hash of the previous block, transaction data, and a timestamp. It is decentralized, transparent, and tamper-resistant. Bitcoin and Ethereum are well-known blockchain platforms."
    },
    {
        "question": "What is a large language model?",
        "answer": "A large language model (LLM) is a deep learning model trained on vast amounts of text data to understand and generate human language. LLMs like GPT-4, Claude, and LLaMA use transformer architectures and can perform tasks like text generation, summarization, translation, and question answering."
    }
]


def preprocess(text):
    """Tokenize and clean text: lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    # Simple stopword removal
    stopwords = {'a','an','the','is','it','in','on','at','to','for','of','and',
                 'or','but','not','with','as','by','from','this','that','are',
                 'was','be','been','have','has','had','do','does','did','will',
                 'would','could','should','may','might','can','what','which',
                 'who','how','when','where','why','i','you','we','they','he',
                 'she','me','him','her','us','them','my','your','our','their'}
    return [t for t in tokens if t not in stopwords]


def build_tf(tokens):
    """Compute term frequency for a list of tokens."""
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) if tokens else 1
    return {k: v / total for k, v in tf.items()}


def build_idf(documents):
    """Compute inverse document frequency across all documents."""
    n = len(documents)
    df = {}
    for doc in documents:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    idf = {}
    for term, count in df.items():
        idf[term] = math.log((n + 1) / (count + 1)) + 1
    return idf


def tfidf_vector(tokens, idf):
    """Compute TF-IDF vector for tokens given precomputed IDF."""
    tf = build_tf(tokens)
    vec = {}
    for term, tf_val in tf.items():
        vec[term] = tf_val * idf.get(term, 1.0)
    return vec


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two TF-IDF vectors."""
    keys = set(vec1.keys()) & set(vec2.keys())
    if not keys:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in keys)
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# Preprocess all FAQs at startup
processed_questions = [preprocess(faq["question"] + " " + faq["answer"]) for faq in FAQ_DATA]
idf = build_idf(processed_questions)
faq_vectors = [tfidf_vector(tokens, idf) for tokens in processed_questions]


def find_best_match(user_input, threshold=0.05):
    """Find the FAQ with the highest cosine similarity to the user input."""
    tokens = preprocess(user_input)
    if not tokens:
        return None, 0.0
    query_vec = tfidf_vector(tokens, idf)
    best_idx = -1
    best_score = 0.0
    for i, faq_vec in enumerate(faq_vectors):
        score = cosine_similarity(query_vec, faq_vec)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx >= 0 and best_score >= threshold:
        return FAQ_DATA[best_idx], best_score
    return None, best_score


def chat(user_input):
    """Return a chatbot response for the given user input."""
    user_input = user_input.strip()
    if not user_input:
        return "Please type a question."
    match, score = find_best_match(user_input)
    if match:
        return match["answer"]
    return ("I am sorry, I could not find a relevant answer in my knowledge base. "
            "Try rephrasing your question or ask about programming, machine learning, "
            "databases, algorithms, or web development topics.")


if __name__ == "__main__":
    print("Tech FAQ Chatbot (NLP-powered)")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye!")
            break
        response = chat(user_input)
        print(f"Bot: {response}\n")
