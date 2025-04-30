# ChatPy Messenger-Style GUI Chatbot (Light Theme Only)
# Features:
# ✔ Messenger-style blue/gray bubbles
# ✔ Simple scrollable chat layout
# ✔ Entry + send button (text box on left, wide)
# ✔ User vs Bot alignment
# ✔ Responsive resizing

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
normalize = lambda text: text.lower().translate(str.maketrans('', '', string.punctuation))
questions = list(data.keys())
answers = list(data.values())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_bot_response(user_input):
    user_input_clean = normalize(user_input)
    input_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(input_vec, X)
    index = similarity.argmax()
    score = similarity[0][index]
    return random.choice(answers[index]) if score > 0.3 else "Hmm... I’m not sure how to respond 🤔"

def save_to_history(u, b):
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"You: {u}\nBot: {b}\n\n")

# GUI theme: Messenger style
BG_COLOR = "#FFFFFF"
USER_BUBBLE = "#DCF8C6"
BOT_BUBBLE = "#E6E6E6"

root = tk.Tk()
root.title("ChatPy Messenger")
root.geometry("600x600")
root.minsize(400, 400)
root.config(bg=BG_COLOR)
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)

canvas = Canvas(main_frame, bg=BG_COLOR, bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, sticky="nsew")
scrollbar = Scrollbar(main_frame, command=canvas.yview)
scrollbar.grid(row=0, column=1, sticky="ns")
canvas.configure(yscrollcommand=scrollbar.set)

bubble_container = Frame(canvas, bg=BG_COLOR)
bubble_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=bubble_container, anchor='nw')
canvas.bind("<Configure>", lambda e: canvas.itemconfig("all", width=e.width))

# Adjust bubble width dynamically on resize
canvas.bind("<Configure>", lambda e: canvas.itemconfig("all", width=e.width))


def add_bubble(text, sender):
    bubble_bg = USER_BUBBLE if sender == "user" else BOT_BUBBLE
    bubble = tk.Label(bubble_container, text=text, bg=bubble_bg, fg="black", font=("Arial", 11), wraplength=300, justify="left", padx=10, pady=6, bd=0)
    bubble.pack(padx=10, pady=4, anchor='e' if sender == "user" else 'w')

def respond(user_input):
    add_bubble(user_input, "user")
    bubble_container.update()
    time.sleep(1)
    bot_reply = get_bot_response(user_input)
    add_bubble(bot_reply, "bot")
    save_to_history(user_input, bot_reply)

# Bottom frame for input
entry_frame = tk.Frame(root, bg=BG_COLOR)
entry_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
entry_frame.columnconfigure(0, weight=1)

entry = tk.Entry(entry_frame, font=("Arial", 14), bg="#f0f0f0")
entry.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=10, ipady=6)

send_btn = tk.Button(entry_frame, text="Send", command=lambda: on_send(), font=("Arial", 11, "bold"),
                     bg="#4CAF50", fg="white", padx=10, pady=6)
send_btn.grid(row=0, column=1, sticky="e", padx=(4, 10), pady=10)

def on_send(event=None):
    text = entry.get().strip()
    if text:
        threading.Thread(target=respond, args=(text,)).start()
        entry.delete(0, tk.END)

entry.bind("<Return>", on_send)

root.mainloop()
