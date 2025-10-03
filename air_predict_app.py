# --- air_predict_app.py ---

import tkinter as tk
from tkinter import Canvas, Label, Button, Frame, Toplevel
import cv2
import numpy as np
from hand_tracker import HandTracker
from PIL import Image, ImageTk, EpsImagePlugin
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyttsx3
import threading
import pickle

# : The HelpWindow class 
class HelpWindow:
    def __init__(self, parent):
        self.top = Toplevel(parent)
        self.top.title("Gesture Instructions")
        # REMOVED: self.top.geometry("650x750") to allow auto-sizing
        self.top.resizable(False, False)
        self.top.configure(bg="white")
        self.top.grab_set()

        main_frame = Frame(self.top, bg="white", padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)


        gestures = {
            "draw_gesture.png": ("1. Draw", "Point with only your Index Finger to draw on the canvas."),
            "move_gesture.png": ("2. Move Cursor", "To move the cursor without drawing, raise your Index and Middle fingers."),
            "recognize_gesture.png": ("3. Recognize Character", "After drawing, show your Open Palm (all fingers up) to trigger the recognition."),
            "pinch_gesture.png": ("4. Click a Button", "Hover the cursor over a button and Pinch your Thumb and Index Finger together.")
        }

        self.images = []

        for i, (img_file, (title, desc)) in enumerate(gestures.items()):
            gesture_frame = Frame(main_frame, bg="white", pady=10)
            gesture_frame.pack(fill=tk.X)
            
            try:
                img = Image.open(img_file)
                img = img.resize((150, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.images.append(photo)

                img_label = Label(gesture_frame, image=photo, bg="white")
                img_label.grid(row=0, column=0, rowspan=2, padx=(0, 20))
            except FileNotFoundError:
                print(f"Warning: Could not find image '{img_file}'.")
                img_label = Label(gesture_frame, text=f"[ Image Not Found:\n{img_file} ]", width=20, height=8, bg="#eee")
                img_label.grid(row=0, column=0, rowspan=2, padx=(0, 20))

            title_label = Label(gesture_frame, text=title, font=("Helvetica", 16, "bold"), bg="white", anchor="w")
            title_label.grid(row=0, column=1, sticky="w")

            desc_label = Label(gesture_frame, text=desc, font=("Helvetica", 12), bg="white", wraplength=400, justify="left", anchor="w")
            desc_label.grid(row=1, column=1, sticky="w")

        close_button = Button(main_frame, text="Got it!", font=("Helvetica", 14, "bold"), command=self.top.destroy)
        close_button.pack(pady=20)


class AirPredictApp:
    # --- Constants ---
    CURSOR_SENSITIVITY = 1
    FRAME_PADDING = 100
    DRAWING_COLOR = "blue"
    BRUSH_SIZE = 5
    CURSOR_SIZE = 10
    SMOOTHING_ALPHA = 0.3
    CURSOR_COLOR_DRAW = "green"
    CURSOR_COLOR_IDLE = "red"
    PINCH_COOLDOWN_MS = 500
    BUTTON_HIGHLIGHT_COLOR = "#d0e0ff"
    BUTTON_DEFAULT_COLOR = "SystemButtonFace"
    ALPHABETS_MODEL_PATH = "alphabets_model.h5"
    NUMBERS_MODEL_PATH = "numbers_model.h5"
    ALPHABET_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUMBER_LABELS = "0123456789"
    IMG_SIZE = 28
    CLEAR_CANVAS_DELAY_MS = 1000
    LSTM_MODEL_PATH = "lstm_model_v3.h5"
    TOKENIZER_PATH = "tokenizer.pickle"
    MAX_SEQUENCE_LEN = 20

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x700")
        self.window.configure(bg="#f0f0f0")

        self.alphabets_model = None
        self.numbers_model = None
        self.lstm_model = None
        self.tokenizer = None

        self.tracker = HandTracker()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 130)
        try:
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        except Exception: pass
        self.tts_engine.startLoop(False)

        self.sentence_text = ""
        self.last_x, self.last_y = None, None
        self.smooth_x, self.smooth_y = 0, 0
        self.pinch_active = False
        self.hovered_button = None
        self.current_mode = "ALPHABETS"
        self.help_win = None # ✅ NEW: A placeholder for our help window instance

        self.loading_label = Label(self.window, text="Loading Models, Please Wait...", font=("Helvetica", 24))
        self.loading_label.pack(pady=100)
        threading.Thread(target=self._load_models_background, daemon=True).start()

    def _load_models_background(self):
        print(f"Loading Alphabets model from: {self.ALPHABETS_MODEL_PATH}")
        self.alphabets_model = load_model(self.ALPHABETS_MODEL_PATH)
        print("Alphabets model loaded successfully!")
        print(f"Loading Numbers model from: {self.NUMBERS_MODEL_PATH}")
        self.numbers_model = load_model(self.NUMBERS_MODEL_PATH)
        print("Numbers model loaded successfully!")
        print(f"Loading LSTM model from: {self.LSTM_MODEL_PATH}")
        self.lstm_model = load_model(self.LSTM_MODEL_PATH)
        print("LSTM model loaded successfully!")
        print(f"Loading Tokenizer from: {self.TOKENIZER_PATH}")
        with open(self.TOKENIZER_PATH, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully!")
        self.window.after(0, self._finish_setup)

    def _finish_setup(self):
        print("Loading complete. Building main UI.")
        self.loading_label.pack_forget()
        self._setup_ui()
        self._setup_key_bindings()
        self.update()

    def _setup_ui(self):
        self.window.grid_columnconfigure(0, weight=5)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=5)
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_rowconfigure(2, weight=1)
        self.canvas = Canvas(self.window, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        toolbar_frame = Frame(self.window, bg="#f0f0f0")
        toolbar_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        toolbar_frame.grid_columnconfigure(0, weight=1)
        toolbar_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        
        self.btn_switch_mode = Button(toolbar_frame, text="Mode: Alpha", font=("Helvetica", 16, "bold"), command=self.on_switch_mode)
        self.btn_switch_mode.grid(row=0, column=0, sticky="nsew", pady=5)
        self.btn_space = Button(toolbar_frame, text="Space", font=("Helvetica", 20, "bold"), command=self.on_space_press)
        self.btn_space.grid(row=1, column=0, sticky="nsew", pady=5)
        self.btn_backspace = Button(toolbar_frame, text="Backspace", font=("Helvetica", 20, "bold"), command=self.on_backspace_press)
        self.btn_backspace.grid(row=2, column=0, sticky="nsew", pady=5)
        self.btn_clear = Button(toolbar_frame, text="Clear", font=("Helvetica", 20, "bold"), command=self.on_clear_press)
        self.btn_clear.grid(row=3, column=0, sticky="nsew", pady=5)
        
        self.btn_help = Button(toolbar_frame, text="Help (?)", font=("Helvetica", 16, "bold"), command=self.open_help_window)
        self.btn_help.grid(row=4, column=0, sticky="nsew", pady=5)

        suggestions_frame = Frame(self.window, bg="#f0f0f0")
        suggestions_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        suggestions_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.suggestion_buttons = []
        for i in range(3):
            btn = Button(suggestions_frame, text="---", font=("Helvetica", 14, "bold"), state="disabled")
            btn.config(command=lambda b=btn: self.on_suggestion_press(b['text']))
            btn.grid(row=0, column=i, sticky="ew", padx=5)
            self.suggestion_buttons.append(btn)
        self.sentence_bar = Label(self.window, text="", font=("Courier", 40, "bold"), bg="white", anchor="w", relief="sunken", padx=10)
        self.sentence_bar.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.clickable_buttons = [self.btn_switch_mode, self.btn_space, self.btn_backspace, self.btn_clear, self.btn_help] + self.suggestion_buttons
        self.cursor = self.canvas.create_oval(0, 0, 0, 0, fill=self.CURSOR_COLOR_IDLE, outline="")

    # ✅ FIX 2: This function is rewritten to be robust
    def open_help_window(self):
        """Creates a new HelpWindow, or brings an existing one to the front."""
        if self.help_win and self.help_win.top.winfo_exists():
            self.help_win.top.lift()
        else:
            self.help_win = HelpWindow(self.window)

    def _setup_key_bindings(self):
        self.window.bind("<space>", lambda e: self.on_space_press())
        self.window.bind("<BackSpace>", lambda e: self.on_backspace_press())
        self.window.bind("<c>", lambda e: self.on_clear_press())
        self.window.bind("<q>", lambda e: self.on_close())
        self.window.bind("<m>", lambda e: self.on_switch_mode())

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(10, self.update)
            return
        if self.window.winfo_width() > 1:
            frame = cv2.flip(frame, 1)
            tracker_output = self.tracker.process_frame(frame)
            self._handle_gestures(tracker_output)
        self.tts_engine.iterate()
        self.window.after(10, self.update)

    def _map_coords(self, hand_coords):
        if not hand_coords: return None
        cam_x, cam_y = hand_coords
        win_width = self.window.winfo_width()
        win_height = self.window.winfo_height()
        mapped_x = np.interp(cam_x, [self.FRAME_PADDING, self.cam_width - self.FRAME_PADDING], [0, win_width])
        mapped_y = np.interp(cam_y, [self.FRAME_PADDING, self.cam_height - self.FRAME_PADDING], [0, win_height])
        return int(np.clip(mapped_x, 0, win_width)), int(np.clip(mapped_y, 0, win_height))

    def _handle_gestures(self, tracker_output):
        gesture = tracker_output["gesture"]
        win_coords = self._map_coords(tracker_output["cursor_coords"])
        if win_coords:
            self.smooth_x = int(self.SMOOTHING_ALPHA * win_coords[0] + (1 - self.SMOOTHING_ALPHA) * self.smooth_x)
            self.smooth_y = int(self.SMOOTHING_ALPHA * win_coords[1] + (1 - self.SMOOTHING_ALPHA) * self.smooth_y)
            self._update_button_hovers(self.smooth_x, self.smooth_y)
            canvas_x = self.smooth_x - self.canvas.winfo_x()
            canvas_y = self.smooth_y - self.canvas.winfo_y()
            self.canvas.coords(self.cursor, canvas_x - self.CURSOR_SIZE, canvas_y - self.CURSOR_SIZE, canvas_x + self.CURSOR_SIZE, canvas_y + self.CURSOR_SIZE)
            if gesture == "DRAW":
                self.canvas.itemconfig(self.cursor, fill=self.CURSOR_COLOR_DRAW)
                if self.last_x is not None and 0 < canvas_x < self.canvas.winfo_width() and 0 < canvas_y < self.canvas.winfo_height():
                    self.canvas.create_line(self.last_x, self.last_y, canvas_x, canvas_y, fill=self.DRAWING_COLOR, width=self.BRUSH_SIZE, capstyle=tk.ROUND, tags="drawing")
                self.last_x, self.last_y = canvas_x, canvas_y
            else:
                self.canvas.itemconfig(self.cursor, fill=self.CURSOR_COLOR_IDLE)
                if gesture == "MOVE" and self.last_x is not None:
                    self._recognize_character()
                if gesture == "PINCH":
                    self._handle_pinch_click(win_coords)
                self.last_x, self.last_y = None, None

    def _recognize_character(self):
        image = self._preprocess_canvas_image()
        if image is None:
            self._clear_drawing_canvas()
            return
        if self.current_mode == "ALPHABETS":
            model_to_use = self.alphabets_model
            labels_to_use = self.ALPHABET_LABELS
        else:
            model_to_use = self.numbers_model
            labels_to_use = self.NUMBER_LABELS
        prediction = model_to_use.predict(image, verbose=0)
        predicted_index = np.argmax(prediction)
        if predicted_index < len(labels_to_use):
            predicted_char = labels_to_use[predicted_index]
            self.tts_engine.say(predicted_char)
            self.sentence_text += predicted_char
            self.sentence_bar.config(text=self.sentence_text)
        self.window.after(self.CLEAR_CANVAS_DELAY_MS, self._clear_drawing_canvas)

    def _clear_drawing_canvas(self):
        self.canvas.delete("drawing")

    def _preprocess_canvas_image(self):
        self.canvas.postscript(file="temp_char.eps", colormode='color')
        img = None
        try:
            with Image.open("temp_char.eps") as img_eps:
                img = img_eps.convert('L')
        except Exception: return None
        finally:
            if os.path.exists("temp_char.eps"):
                try: os.remove("temp_char.eps")
                except PermissionError: pass
        if img is None: return None
        img_np = np.array(img)
        img_np = cv2.bitwise_not(img_np)
        contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        x, y, w, h = cv2.boundingRect(contours[0])
        char_img = img_np[y:y+h, x:x+w]
        side_len = max(w, h)
        padding = [(side_len - h) // 2, side_len - h - ((side_len - h) // 2), (side_len - w) // 2, side_len - w - ((side_len - w) // 2)]
        char_img = cv2.copyMakeBorder(char_img, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=0)
        char_img = cv2.resize(char_img, (20, 20), interpolation=cv2.INTER_AREA)
        final_img = cv2.copyMakeBorder(char_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        final_img = final_img.astype('float32') / 255.0
        return np.reshape(final_img, (1, self.IMG_SIZE, self.IMG_SIZE, 1))

    def _update_button_hovers(self, cursor_x, cursor_y):
        currently_hovered = None
        cursor_screen_x = self.window.winfo_rootx() + cursor_x
        cursor_screen_y = self.window.winfo_rooty() + cursor_y
        for button in self.clickable_buttons:
            if button.winfo_exists() and button['state'] != 'disabled':
                x1 = button.winfo_rootx()
                y1 = button.winfo_rooty()
                x2 = x1 + button.winfo_width()
                y2 = y1 + button.winfo_height()
                if x1 < cursor_screen_x < x2 and y1 < cursor_screen_y < y2:
                    currently_hovered = button
                    break
        if self.hovered_button != currently_hovered:
            if self.hovered_button:
                self.hovered_button.config(bg=self.BUTTON_DEFAULT_COLOR)
            if currently_hovered:
                currently_hovered.config(bg=self.BUTTON_HIGHLIGHT_COLOR)
            self.hovered_button = currently_hovered

    def _handle_pinch_click(self, pinch_coords):
        if not self.pinch_active and self.hovered_button:
            self.pinch_active = True
            self.hovered_button.invoke()
            self.window.after(self.PINCH_COOLDOWN_MS, self.reset_pinch)

    def reset_pinch(self):
        self.pinch_active = False

    def on_switch_mode(self):
        if self.current_mode == "ALPHABETS":
            self.current_mode = "NUMBERS"
            self.btn_switch_mode.config(text="Mode: Nums")
        else:
            self.current_mode = "ALPHABETS"
            self.btn_switch_mode.config(text="Mode: Alpha")
        print(f"Switched to {self.current_mode} mode.")

    def on_space_press(self):
        if self.sentence_text.strip():
            self.tts_engine.say(self.sentence_text)
        self.sentence_text += " "
        self.sentence_bar.config(text=self.sentence_text)
        self._predict_next_word()

    def on_suggestion_press(self, word):
        self.tts_engine.say(word)
        self.sentence_text += word.upper() + " "
        self.sentence_bar.config(text=self.sentence_text)
        for btn in self.suggestion_buttons:
            btn.config(text="---", state="disabled")
        self._predict_next_word()

    def _predict_next_word(self):
        if not self.lstm_model or not self.tokenizer: return
        seed_text = self.sentence_text.lower().strip()
        try:
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            padded_sequence = pad_sequences([token_list], maxlen=self.MAX_SEQUENCE_LEN-1, padding='pre')
            predicted_probs = self.lstm_model.predict(padded_sequence, verbose=0)[0]
            top_indices = np.argsort(predicted_probs)[-3:][::-1]
            predicted_words = []
            reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))
            for i in top_indices:
                word = reverse_word_map.get(i)
                if word:
                    predicted_words.append(word)
            for i, word in enumerate(predicted_words):
                if i < len(self.suggestion_buttons):
                    self.suggestion_buttons[i].config(text=word.upper(), state="normal")
        except Exception as e:
            print(f"Prediction error: {e}")

    def on_backspace_press(self):
        self.sentence_text = self.sentence_text[:-1]
        self.sentence_bar.config(text=self.sentence_text)

    def on_clear_press(self):
        self._clear_drawing_canvas()

    def on_close(self, event=None):
        print("Closing application...")
        if self.tts_engine and self.tts_engine._inLoop:
            self.tts_engine.endLoop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe'
    root = tk.Tk()
    app = AirPredictApp(root, "AirPredict")
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
