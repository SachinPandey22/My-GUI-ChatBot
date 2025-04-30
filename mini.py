import tkinter as tk
from tkinter import Canvas, Frame, Scrollbar
import json
import string
import random
import time
import threading
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Load dataset
with open("realdata.json", "r") as file:
    data = json.load(file)

# Normalize
def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

questions = list(data.keys())
answers = list(data.values())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Predict reply
def get_bot_response(user_input):
    user_input_clean = normalize(user_input)
    input_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(input_vec, X)
    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.3:
        return "Hmm... I’m not sure how to respond 🤔"
    return random.choice(answers[index])

# Save history
def save_to_history(user, bot):
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"You: {user}\nBot: {bot}\n\n")

# GUI setup
root = tk.Tk()
root.title("ChatPy Messenger")
root.geometry("420x550")
root.config(bg="white")

# Chat canvas
canvas = Canvas(root, bg="white", bd=0, highlightthickness=0)
frame = Frame(canvas, bg="white")
scrollbar = Scrollbar(root, command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas.create_window((0, 0), window=frame, anchor='nw')

# Auto-resize scroll area
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox('all'))

frame.bind("<Configure>", on_configure)

# Bubbles
def add_bubble(text, sent_by_user=True):
    bubble = tk.Label(frame, text=text, bg="#DCF8C6" if sent_by_user else "#E6E6E6",
                      fg="black", font=("Arial", 12), wraplength=250,
                      justify="left", padx=10, pady=7, bd=0,
                      anchor='w' if sent_by_user else 'e')
    bubble.pack(padx=10, pady=4, anchor='e' if sent_by_user else 'w')

# Message sending
def respond(user_input):
    add_bubble(user_input, sent_by_user=True)
    add_bubble("Bot is typing...", sent_by_user=False)
    frame.update()
    time.sleep(1)

    for widget in frame.winfo_children():
        if widget.cget("text") == "Bot is typing...":
            widget.destroy()

    bot_reply = get_bot_response(user_input)
    add_bubble(bot_reply, sent_by_user=False)
    save_to_history(user_input, bot_reply)

# Entry + Send Button
entry = tk.Entry(root, font=("Arial", 14), bg="#f0f0f0")
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=10, ipady=6)

def on_send(event=None):
    text = entry.get().strip()
    if text:
        threading.Thread(target=respond, args=(text,)).start()
        entry.delete(0, tk.END)

entry.bind("<Return>", on_send)

send_btn = tk.Button(root, text="Send", command=on_send, font=("Arial", 11, "bold"),
                     bg="#4CAF50", fg="white", padx=10, pady=6)
send_btn.pack(side=tk.RIGHT, padx=10)

root.mainloop()
