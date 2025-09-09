# --- homepage.py ---

import tkinter as tk
from tkinter import Canvas
from PIL import ImageTk, Image

class HomePage:
    def __init__(self, master, launch_app_callback):
        self.master = master
        self.master.title("AirPredict - Home")
        self.master.attributes('-fullscreen', True)

        # --- Load and keep a reference to the background image ---
        # This is crucial to prevent the image from disappearing.
        self.bg_image = ImageTk.PhotoImage(Image.open("./bg.jpeg"))

        # Create a canvas that fills the whole window
        self.canvas = Canvas(self.master)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

        # --- Create Buttons ---
        # The "Write" button will now call the function passed in from main.py
        self.btn_write = tk.Button(
            self.master, 
            text="Start Writing", 
            command=launch_app_callback,
            fg="white", bg="#6699cc", width=20, height=2,
            activebackground="#99ccff", font=('times', 15, ' bold ')
        )
        # Place the button on the canvas
        self.btn_write.place(x=1100, y=450)

        self.btn_quit = tk.Button(
            self.master, 
            text="Quit", 
            command=self.master.destroy, # Safely closes the application
            fg="white", bg="#6699cc", width=20, height=2,
            activebackground="#99ccff", font=('times', 15, ' bold ')
        )
        self.btn_quit.place(x=1100, y=580)