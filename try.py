import random
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Knowledge base
data = {
    "hi": "Hello! How can I assist you?",
    "hello": "Hi there!",
    "how are you": "I'm just a bot, but I'm functioning properly!",
    "bye": "Goodbye! Have a great day.",
    "what is your name": "I'm a mini AI chatbot.",
    "who created you": "A Python programmer!",
    "thank you": "You're welcome!",
    "can you help me": "Of course, ask me anything.",
}

# Normalize and tokenize
def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

questions = [normalize(q) for q in data.keys()]
answers = list(data.values())

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chat loop
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ").lower()
    if user_input == "exit":
        print("Chatbot: Bye!")
        break

    user_input_clean = normalize(user_input)
    input_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(input_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.3:
        print("Chatbot: I'm not sure how to respond to that.")
    else:
        print("Chatbot:", answers[index])
