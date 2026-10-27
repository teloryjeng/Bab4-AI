# Solusi Exercise 4.19 - Analisis Sentimen Berbagai Kalimat
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
"I absolutely love this new smartphone, it works wonderfully!",
"The customer service was terrible and I waited for hours.",
"The movie had a great plot and amazing visual effects.",
"I am very disappointed with the product quality."]

for s in sentences:
    result = classifier(s)
    print(f"Kalimat: '{s}' -> Hasil: {result}")