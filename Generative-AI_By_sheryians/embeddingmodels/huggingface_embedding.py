from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name ='sentence-transformers/all-MiniLM-L6-v2',
)

text = [
    'Hello i am Programmer',
    'I am from India.',
    'We are here to change the world'
]

# vector = embeddings.embed_query("You are going to learn Gen AI")

vector = embeddings.embed_documents(texts=text)

print(vector)