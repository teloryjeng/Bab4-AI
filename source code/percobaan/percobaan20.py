# Solusi Exercise 4.20 - Question Answering dengan Konteks Baru
from transformers import pipeline

question_answerer = pipeline('question-answering')

context_text = """
Python is an interpreted high-level general-purpose programming language.
Python was designed by Guido van Rossum and was first released in 1991.
"""

q1 = question_answerer({'question': 'Who designed Python?', 'context': context_text})
q2 = question_answerer({'question': 'When was Python first released?', 'context': context_text})

print("Jawaban 1:", q1)
print("Jawaban 2:", q2)