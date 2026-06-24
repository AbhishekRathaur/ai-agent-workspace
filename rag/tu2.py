#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Instantiate the newer ChatOllama instance
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Instantiate the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("✓ Core LCEL structures loaded and linked to Ollama.")


# In[2]:


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Knowledge base content
kb_data = """
MeshQuery is our internal high-scale distributed indexing engine built in 2026. 
It uses a specialized sharding strategy optimized for high-throughput Kafka topics.
Unlike standard relational databases, MeshQuery caches query indexes using H3 spatial indexing clusters 
and routes requests across nodes using an ultra-low latency RSocket protocol layer.
The default port for MeshQuery cluster coordination is 9092.
"""

# 2. Partition text split strategy
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(kb_data)]

# 3. Populate FAISS Vector Store
vector_store = FAISS.from_documents(docs, embeddings)

# 4. Expose the vector store as a standard Retreiver object
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

print("✓ FAISS vector index built successfully. Retriever exposed.")


# In[3]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Helper function to extract and merge text from retrieved FAISS documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Design a strict system prompt to constrain the model's knowledge base
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are an advanced software systems helper. Answer the user's query using ONLY the provided context block below. "
        "If you do not know the answer or if it's not explicitly in the context, say 'I do not have that information.'\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])

# 3. Define the declarative LCEL chain layout using the pipe operator
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

print("✓ LCEL RAG chain defined and compiled successfully.")


# In[8]:


# Define the query targeting our indexed system documentation
query = "What is capital of france?"

print(f"Query: {query}\n")
print("--- Streaming Chain Output ---")

# Execute the LCEL engine and print tokens instantly as they arrive
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)
print("\n")    

