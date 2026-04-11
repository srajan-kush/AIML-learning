from gensim.models import FastText

sentences = [["i", "love", "ai"], ["fasttext", "is", "cool"]]

model = FastText(sentences, vector_size=100, window=3, min_count=1)

print(model.wv['love'])