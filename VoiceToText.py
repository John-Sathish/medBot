import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import threading
import queue
import speech_recognition as sr


class MedicalAppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Notification")
        self.root.geometry("400x700")  # Adjust window size as desired

        # Queue for passing recognized speech/results from background thread to UI
        self.queue = queue.Queue()

        # Flag to control recognition loop
        self.listening = False

        # 1. Header Frame (Title + Date)
        header_frame = tk.Frame(self.root, bg="white")
        header_frame.pack(fill=tk.X, side=tk.TOP)

        self.title_label = tk.Label(
            header_frame, text="Notification",
            font=("Arial", 16, "bold"), bg="white"
        )
        self.title_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.date_label = tk.Label(
            header_frame, text="Date : 15/03/2025",
            font=("Arial", 12), bg="white"
        )
        self.date_label.pack(side=tk.RIGHT, padx=10)

        # 2. Main Content Frame
        content_frame = tk.Frame(self.root, bg="#f2f2f2")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Symptoms
        self.symptoms_frame = tk.Frame(content_frame, bg="white", bd=1, relief=tk.FLAT)
        self.symptoms_frame.pack(fill=tk.X, pady=5)

        self.symptoms_label = tk.Label(
            self.symptoms_frame, text="Symptoms", font=("Arial", 12, "bold"), bg="white"
        )
        self.symptoms_label.pack(anchor="w", padx=5, pady=5)

        self.symptoms_text = ScrolledText(self.symptoms_frame, wrap=tk.WORD, height=5)
        self.symptoms_text.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Examinations
        self.exams_frame = tk.Frame(content_frame, bg="white", bd=1, relief=tk.FLAT)
        self.exams_frame.pack(fill=tk.X, pady=5)

        self.exams_label = tk.Label(
            self.exams_frame, text="Examinations", font=("Arial", 12, "bold"), bg="white"
        )
        self.exams_label.pack(anchor="w", padx=5, pady=5)

        self.exams_text = ScrolledText(self.exams_frame, wrap=tk.WORD, height=5)
        self.exams_text.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Prescriptions
        self.prescriptions_frame = tk.Frame(content_frame, bg="white", bd=1, relief=tk.FLAT)
        self.prescriptions_frame.pack(fill=tk.X, pady=5)

        self.prescriptions_label = tk.Label(
            self.prescriptions_frame, text="Prescriptions", font=("Arial", 12, "bold"), bg="white"
        )
        self.prescriptions_label.pack(anchor="w", padx=5, pady=5)

        self.prescriptions_text = ScrolledText(self.prescriptions_frame, wrap=tk.WORD, height=5)
        self.prescriptions_text.pack(fill=tk.X, padx=5, pady=(0, 5))

        # 3. "View Summary" Button
        self.summary_button = tk.Button(
            content_frame, text="View Summary", bg="#007BFF", fg="white",
            font=("Arial", 12, "bold"), command=self.view_summary
        )
        self.summary_button.pack(fill=tk.X, pady=10, ipady=5)

        # 4. Floating Microphone Button
        #    We'll place it with .place() to simulate a "floating" effect at bottom-right.
        self.mic_button = tk.Button(
            content_frame, text="🎤", font=("Arial", 14, "bold"), fg="white",
            bg="#4285F4", bd=0, relief=tk.RAISED, command=self.toggle_listening
        )
        self.mic_button.place(relx=0.9, rely=0.85, anchor=tk.CENTER, width=50, height=50)
        self.mic_button.config(cursor="hand2")

        # 5. Bottom Navigation Bar
        nav_frame = tk.Frame(self.root, bg="white", height=50)
        nav_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Four navigation placeholders
        home_btn = tk.Label(nav_frame, text="Home", bg="white", font=("Arial", 10))
        home_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        reports_btn = tk.Label(nav_frame, text="Reports", bg="white", font=("Arial", 10))
        reports_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        notif_btn = tk.Label(nav_frame, text="Notification", bg="white", font=("Arial", 10, "bold"))
        notif_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        profile_btn = tk.Label(nav_frame, text="Profile", bg="white", font=("Arial", 10))
        profile_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Start periodic check of queue for recognized speech
        self.check_queue()

    def view_summary(self):
        # Here, you might collect data from the text boxes and display or process it.
        print("View Summary button clicked.")

    def toggle_listening(self):
        """
        Start/Stop the speech recognition loop in a background thread.
        """
        if not self.listening:
            self.listening = True
            self.mic_button.config(bg="#DB4437")  # Change color to indicate 'listening'
            threading.Thread(target=self.recognition_loop, daemon=True).start()
        else:
            self.listening = False
            self.mic_button.config(bg="#4285F4")  # Revert color

    def recognition_loop(self):
        """
        Continuously listens for audio and attempts to recognize speech using Google API.
        """
        r = sr.Recognizer()
        while self.listening:
            try:
                with sr.Microphone() as source:
                    self.post_message("Adjusting for ambient noise...")
                    r.adjust_for_ambient_noise(source, duration=1)
                    self.post_message("Listening for speech...")
                    audio_data = r.listen(source, timeout=5, phrase_time_limit=10)

                text = r.recognize_google(audio_data)
                self.post_message(f"You said: {text}")

                # Example logic: if user says "stop", stop listening
                if "stop" in text.lower():
                    self.post_message("Stop command recognized.")
                    self.listening = False
                    self.mic_button.config(bg="#4285F4")
            except sr.WaitTimeoutError:
                self.post_message("Listening timed out; no speech detected.")
            except sr.UnknownValueError:
                self.post_message("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                self.post_message(f"Could not request results; {e}")

    def post_message(self, msg):
        """
        Send a message from the recognition thread to the main thread via the queue.
        """
        self.queue.put(msg)

    def check_queue(self):
        """
        Periodically checks the queue for new messages from the recognition thread
        and inserts them into the 'Symptoms' box (or anywhere you prefer).
        """
        while not self.queue.empty():
            message = self.queue.get()
            # For demonstration, we append recognized text to the "Symptoms" text box
            self.symptoms_text.insert(tk.END, message + "\n")
            self.symptoms_text.see(tk.END)
        self.root.after(200, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAppUI(root)
    root.mainloop()
