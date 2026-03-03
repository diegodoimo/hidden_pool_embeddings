dataset_dict = {
    # Retrieval — question answering
    "arguana": (
        "Retrieval",
        "Given a question, retrieve passages that answer the question.",
    ),
    "squad": (
        "Retrieval",
        "Given a question, retrieve passages that answer the question.",
    ),
    "bioasq": (
        "Retrieval",
        "Given a question, retrieve passages that answer the question.",
    ),
    "nfcorpus": (
        "Retrieval",
        "Given a question, retrieve passages that answer the question.",
    ),
    "miracl": (
        "Retrieval",
        "Given a question, retrieve passages that answer the question.",
    ),
    "mr_tydi": (
        "Retrieval",
        "Given a question, retrieve passages that answer the question.",
    ),
    # Retrieval — web search
    "paq": (
        "Retrieval",
        "Given a web search query, retrieve relevant passages that answer the query.",
    ),
    "stackexchange": (
        "Retrieval",
        "Given a web search query, retrieve relevant passages that answer the query.",
    ),
    "msmarco": (
        "Retrieval",
        "Given a web search query, retrieve relevant passages that answer the query.",
    ),
    "natural_questions": (
        "Retrieval",
        "Given a web search query, retrieve relevant passages that answer the query.",
    ),
    # Retrieval — NLI / entailment
    "snli": (
        "Retrieval",
        "Given a premise, retrieve hypotheses that are entailed by the premise.",
    ),
    "mnli": (
        "Retrieval",
        "Given a premise, retrieve hypotheses that are entailed by the premise.",
    ),
    "anli": (
        "Retrieval",
        "Given a premise, retrieve hypotheses that are entailed by the premise.",
    ),
    # Retrieval — misc
    "hotpotqa": (
        "Retrieval",
        "Given a multi-hop question, retrieve passages that answer the question.",
    ),
    "fever": (
        "Retrieval",
        "Given a claim, retrieve documents that support or refute the claim.",
    ),
    "eli5": (
        "Retrieval",
        "Given a question from Reddit ELI5 forum, retrieve passages that answer it.",
    ),
    "fiqa2018": (
        "Retrieval",
        "Given a financial question, retrieve passages that answer the question.",
    ),
    "scifact": (
        "Retrieval",
        "Given a scientific claim, retrieve passages that support or refute the claim.",
    ),
    "triviaqa": (
        "Retrieval",
        "Given a trivia question, retrieve passages that can answer it.",
    ),
    "coliee": (
        "Retrieval",
        "Given a legal statement, retrieve articles that support it.",
    ),
    "pubmedqa": (
        "Retrieval",
        "Given a question, retrieve paper abstracts from PubMed that can answer it.",
    ),
    "s2orc_title_abstract": (
        "Retrieval",
        "Given a paper's title, retrieve the corresponding abstract.",
    ),
    "s2orc_title_citation": (
        "Retrieval",
        "Given a paper's title, retrieve papers that cite it.",
    ),
    "s2orc_abstract_citation": (
        "Retrieval",
        "Given a paper's abstract, retrieve abstract of papers that cite it.",
    ),
    "amazon_qa": (
        "Retrieval",
        "Given a question about a product, retrieve Amazon reviews that can help answer it.",
    ),
    "specter": (
        "Retrieval",
        "Given a scientific paper title, retrieve paper titles that are cited by the given paper.",
    ),
    "xsum": ("Retrieval", "Given a news summary, retrieve the original news article."),
    "cnn_dm": (
        "Retrieval",
        "Given a news summary, retrieve the original news article.",
    ),
    "sentence_compression": (
        "Retrieval",
        "Given a compressed sentence, retrieve the original sentence before compression.",
    ),
    "qqp": (
        "Retrieval",
        "Given a question, retrieve questions that are semantically equivalent.",
    ),
    "stackexchange_dup_questions_s2s": (
        "Retrieval",
        "Given a question, retrieve questions that are semantically equivalent.",
    ),
    "stackexchange_dup_questions_p2p": (
        "Retrieval",
        "Given a question, retrieve questions that are semantically equivalent.",
    ),
    "stackoverflow_dup_questions": (
        "Retrieval",
        "Retrieve duplicate questions from StackOverflow forum.",
    ),
    "sts12": ("Retrieval", "Retrieve semantically similar text."),
    "sts22": ("Retrieval", "Retrieve semantically similar text."),
    "sts_benchmark": ("Retrieval", "Retrieve semantically similar text."),
    # Classification
    "amazon_counterfactual": (
        "Classification",
        "Classify a given Amazon customer review text as either counterfactual or not counterfactual.",
    ),
    "amazon_polarity": (
        "Classification",
        "Classify the given Amazon review into positive or negative sentiment.",
    ),
    "imdb": (
        "Classification",
        "Classify the sentiment expressed in the given movie review text from the IMDB dataset.",
    ),
    "toxic_conversations": (
        "Classification",
        "Classify the given comments as either toxic or not toxic.",
    ),
    "cola": (
        "Classification",
        "Classify the given sentence as linguistically acceptable or not acceptable.",
    ),
    # Clustering
    "amazon_reviews": (
        "Clustering",
        "Classify the given Amazon review into its appropriate rating category.",
    ),
    "banking77": (
        "Clustering",
        "Given an online banking query, find the corresponding intents.",
    ),
    "emotion": (
        "Clustering",
        "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise.",
    ),
    "mtop_intent": (
        "Clustering",
        "Classify the intent of the given utterance in task-oriented conversation.",
    ),
    "mtop_domain": (
        "Clustering",
        "Classify the intent domain of the given utterance in task-oriented conversation.",
    ),
    "massive_scenario": (
        "Clustering",
        "Given a user utterance as query, find the user scenarios.",
    ),
    "massive_intent": (
        "Clustering",
        "Given a user utterance as query, find the user intents.",
    ),
    "tweet_sentiment_extraction": (
        "Clustering",
        "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
    ),
    "arxiv_clustering_p2p": (
        "Clustering",
        "Identify the main and secondary category of arXiv papers based on the titles and abstracts.",
    ),
    "arxiv_clustering_s2s": (
        "Clustering",
        "Identify the main and secondary category of arXiv papers based on the titles.",
    ),
    "biorxiv_clustering_p2p": (
        "Clustering",
        "Identify the main category of bioRxiv papers based on the titles and abstracts.",
    ),
    "biorxiv_clustering_s2s": (
        "Clustering",
        "Identify the main category of bioRxiv papers based on the titles.",
    ),
    "medrxiv_clustering_p2p": (
        "Clustering",
        "Identify the main category of medRxiv papers based on the titles and abstracts.",
    ),
    "medrxiv_clustering_s2s": (
        "Clustering",
        "Identify the main category of medRxiv papers based on the titles.",
    ),
    "reddit_clustering_p2p": (
        "Clustering",
        "Identify the topic or theme of Reddit posts based on the titles and posts.",
    ),
    "reddit_clustering_s2s": (
        "Clustering",
        "Identify the topic or theme of Reddit posts based on the titles.",
    ),
    "stackexchange_clustering_p2p": (
        "Clustering",
        "Identify the topic or theme of StackExchange posts based on the given paragraphs.",
    ),
    "stackexchange_clustering_s2s": (
        "Clustering",
        "Identify the topic or theme of StackExchange posts based on the titles.",
    ),
    "twenty_newsgroups": (
        "Clustering",
        "Identify the topic or theme of the given news articles.",
    ),
}
