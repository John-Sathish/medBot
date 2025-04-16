import tkinter as tk
from datetime import datetime
import json

with open('intents.json', 'r') as file:
    intents = json.load(file)

def preprocess_text(text):
    return text.lower().strip()

def find_intent(user_input):
    user_input = preprocess_text(user_input)
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input:
                return intent
    return None

def get_response(user_input):
    intent = find_intent(user_input)
    if intent:
        return intent["responses"]
    else:
        return "I'm sorry, I didn't quite catch that. Could you describe your symptom differently?"

root = tk.Tk()
root.title("Symptom Checker")
root.geometry("400x600")
root.configure(bg="#f5f5f5")

header = tk.Frame(root, bg="#0078FF", height=60)
header.pack(fill="x")

bot_label = tk.Label(header, text="SymptomBot", font=("Helvetica", 14, "bold"), fg="white", bg="#0078FF")
bot_label.pack(side="left", padx=15, pady=15)

timestamp = tk.Label(root, text=datetime.now().strftime("%m/%d/%y, %I:%M %p"), font=("Helvetica", 9), bg="#f5f5f5", fg="#888")
timestamp.pack(pady=5)

chat_frame = tk.Frame(root, bg="#f5f5f5")
chat_frame.pack(fill="both", expand=True, padx=10)

canvas = tk.Canvas(chat_frame, bg="#f5f5f5", highlightthickness=0)
scrollbar = tk.Scrollbar(chat_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#f5f5f5")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

input_frame = tk.Frame(root, bg="#ffffff", height=50)
input_frame.pack(fill="x")

user_input = tk.Entry(input_frame, font=("Helvetica", 12), bd=0)
user_input.pack(side="left", fill="x", expand=True, padx=10, pady=10)

def on_send():
    text = user_input.get().strip()
    if text == "":
        return

    add_user_message(text)
    user_input.delete(0, tk.END)

    bot_response = get_response(text)
    root.after(500, lambda: add_bot_message(bot_response))  # Simulated delay

def on_enter(event):
    on_send()

user_input.bind("<Return>", on_enter)

send_button = tk.Button(input_frame, text="Send", command=on_send, bg="#0078FF", fg="white", padx=10, pady=5)
send_button.pack(side="right", padx=10)

def add_bot_message(text):
    msg_frame = tk.Frame(scrollable_frame, bg="#f5f5f5")

    bubble = tk.Label(
        msg_frame,
        text=text,
        bg="#ffffff",
        fg="#333333",
        font=("Helvetica", 12),
        wraplength=250,
        justify="left",
        padx=10,
        pady=8,
        bd=1,
        relief="solid"
    )
    bubble.pack(side="left", padx=5)

    msg_frame.pack(anchor="w", pady=10, fill="x")
    canvas.yview_moveto(1.0)

def add_user_message(text):
    msg_frame = tk.Frame(scrollable_frame, bg="#f5f5f5")

    bubble = tk.Label(
        msg_frame,
        text=text,
        bg="#f5f5f5",
        fg="#0078FF",
        font=("Helvetica", 12, "bold"),
        wraplength=250,
        justify="right",
        padx=10,
        pady=8,
        bd=1,
        relief="solid",
        highlightbackground="#0078FF"
    )
    bubble.pack(side="right", padx=5)

    msg_frame.pack(anchor="e", pady=10, fill="x")
    canvas.yview_moveto(1.0)

root.after(100, lambda: add_bot_message("Hello! I'm SymptomBot 🤖\nPlease describe your symptoms, and I'll try to help!"))

root.mainloop()
