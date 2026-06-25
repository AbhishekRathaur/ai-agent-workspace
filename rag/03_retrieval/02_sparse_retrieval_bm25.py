from rank_bm25 import BM25Okapi

def run_sparse_search():
    # 1. Simulate our chunked document corpus
    corpus = [
        "The primary database node handles all transaction writes.",
        "Set the connectTimeout parameter to 5000ms in production.",
        "Booking BK-130 was cancelled due to a network timeout error."
    ]

    # 2. Tokenize the corpus (BM25 requires pre-split arrays of words)
    # In production, you use standard NLP tokenizers (like NLTK or SpaCy) for clean punctuation splitting
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]

    # 3. Initialize the hardware-efficient Sparse Search matrix
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Execute a highly specific keyword query
    query = "connectTimeout"
    tokenized_query = query.lower().split(" ")

    # 5. Calculate keyword frequency scores (TF-IDF mapping)
    doc_scores = bm25.get_scores(tokenized_query)
    
    print(f"🔍 Lexical Search for exact syntax: '{query}'\n")
    for index, score in enumerate(doc_scores):
        print(f"Score: {score:.4f} | Chunk: {corpus[index]}")

if __name__ == "__main__":
    run_sparse_search()