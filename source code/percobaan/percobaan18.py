# Solusi Exercise 4.18 - Analisis Teks Lain dengan NLTK
import nltk

sentence = """
Deep learning is a subset of machine learning based on artificial neural
networks.
Guido van Rossum released Python in 1991.
"""
tokens = nltk.word_tokenize(sentence)
print("Tokens: \n", tokens)
tagged = nltk.pos_tag(tokens)
print("POS Tagged: \n", tagged)
entities = nltk. chunk.ne_chunk(tagged)
print("Named Entities: \n", entities)