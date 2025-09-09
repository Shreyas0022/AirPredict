# --- air_predict_app.py (Final, Fully Commented Version) ---

import tkinter as tk
from tkinter import Canvas, Label, Button, Frame
import cv2
import numpy as np
from hand_tracker import HandTracker
from PIL import Image, ImageOps, EpsImagePlugin
import os
from tensorflow.keras.models import load_model
import pyttsx3
import threading

class AirPredictApp:
    """
    The main class for the AirPredict application.
    This class manages the UI, webcam, hand tracking, and model predictions.
    """
    # --- Constants for easy configuration of the application's behavior and appearance ---
    CURSOR_SENSITIVITY = 1.5      # Multiplier for cursor speed. >1 is faster, <1 is slower.
    FRAME_PADDING = 100           # Pixels to ignore around the camera's edge, creating a smaller active area.
    DRAWING_COLOR = "blue"        # Color of the drawing line on the canvas.
    BRUSH_SIZE = 5                # Thickness of the drawing line.
    CURSOR_SIZE = 10              # Radius of the virtual cursor dot.
    SMOOTHING_ALPHA = 0.3         # Smoothing factor for the cursor. Lower values are smoother but have more lag.
    CURSOR_COLOR_DRAW = "green"   # Cursor color when in DRAW mode.
    CURSOR_COLOR_IDLE = "red"     # Cursor color when not drawing.
    PINCH_COOLDOWN_MS = 500       # Prevents a single pinch from causing multiple rapid clicks.
    BUTTON_HIGHLIGHT_COLOR = "#d0e0ff" # Background color for a button when hovered over.
    BUTTON_DEFAULT_COLOR = "SystemButtonFace" # Default system color for buttons.
    
    # --- Model and Label Configuration ---
    ALPHABETS_MODEL_PATH = "alphabets_model.h5"
    NUMBERS_MODEL_PATH = "numbers_model.h5"
    ALPHABET_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUMBER_LABELS = "0123456789"
    IMG_SIZE = 28                 # The image size the CNN model expects (28x28).
    CLEAR_CANVAS_DELAY_MS = 1000  # Delay in milliseconds before the canvas is auto-cleared after a prediction.

    def __init__(self, window, window_title):
        """
        Initializes the application, sets up the splash screen, and starts background loading.
        """
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x700")
        self.window.configure(bg="#f0f0f0")

        # --- Core Components will be initialized in a background thread to prevent UI freeze ---
        self.tracker = None
        self.cap = None
        self.alphabets_model = None
        self.numbers_model = None
        self.tts_engine = None

        # --- State Variables to track the application's current status ---
        self.sentence_text = ""
        self.last_x, self.last_y = None, None # Last known coordinates for drawing lines.
        self.smooth_x, self.smooth_y = 0, 0   # Smoothed cursor coordinates for display.
        self.pinch_active = False             # A flag to manage the pinch cooldown.
        self.hovered_button = None            # The button currently under the virtual cursor.
        self.current_mode = "ALPHABETS"       # The current recognition mode ("ALPHABETS" or "NUMBERS").

        # --- Splash Screen Setup ---
        # Display a "Loading..." message immediately while heavy components load.
        self.loading_label = Label(self.window, text="Loading Models, Please Wait...", font=("Helvetica", 24))
        self.loading_label.pack(pady=100)
        
        # Start loading all heavy components (models, camera) in a separate thread.
        threading.Thread(target=self._initialize_components, daemon=True).start()

    def _initialize_components(self):
        """
        Loads all heavy resources (camera, models, TTS) in a background thread
        to keep the UI responsive.
        """
        print("Initializing components in background...")
        self.tracker = HandTracker()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"Loading Alphabets model from: {self.ALPHABETS_MODEL_PATH}")
        self.alphabets_model = load_model(self.ALPHABETS_MODEL_PATH)
        print("Alphabets model loaded successfully!")
        
        print(f"Loading Numbers model from: {self.NUMBERS_MODEL_PATH}")
        self.numbers_model = load_model(self.NUMBERS_MODEL_PATH)
        print("Numbers model loaded successfully!")

        # Initialize and configure the Text-to-Speech engine.
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 130) # Set a child-friendly speech rate.
        try:
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if "female" in voice.name.lower() or "zira" in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    print(f"TTS Voice set to: {voice.name}")
                    break
        except Exception as e:
            print(f"Could not set a specific TTS voice: {e}")
        
        # Start the TTS engine's event loop in a non-blocking way.
        self.tts_engine.startLoop(False)
        
        # Once loading is complete, schedule the main UI setup to run on the main thread.
        self.window.after(0, self._finish_setup)

    def _finish_setup(self):
        """
        Builds the main UI after all components have been loaded in the background.
        This function is called on the main UI thread.
        """
        print("Loading complete. Building main UI.")
        # Remove the "Loading..." message.
        self.loading_label.pack_forget()
        
        # Build the main application UI widgets.
        self._setup_ui()
        # Set up keyboard shortcuts.
        self._setup_key_bindings()
        
        # Start the main update loop for video and gestures.
        self.update()

    def _setup_ui(self):
        """
        Creates and arranges all the main UI widgets using a grid layout.
        """
        # Configure the main window's grid.
        self.window.grid_columnconfigure(0, weight=5) # Canvas column is wider.
        self.window.grid_columnconfigure(1, weight=1) # Toolbar column is narrower.
        self.window.grid_rowconfigure(0, weight=5)    # Canvas/Toolbar row is taller.
        self.window.grid_rowconfigure(1, weight=1)    # Suggestions row.
        self.window.grid_rowconfigure(2, weight=1)    # Sentence bar row.

        # Create the main writing canvas.
        self.canvas = Canvas(self.window, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        # Create the toolbar on the right.
        toolbar_frame = Frame(self.window, bg="#f0f0f0")
        toolbar_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        toolbar_frame.grid_columnconfigure(0, weight=1)
        toolbar_frame.grid_rowconfigure((0, 1, 2, 3), weight=1) # 4 rows for 4 buttons.

        # Create all buttons for the toolbar.
        self.btn_switch_mode = Button(toolbar_frame, text="Mode: Alpha", font=("Helvetica", 16, "bold"), command=self.on_switch_mode)
        self.btn_switch_mode.grid(row=0, column=0, sticky="nsew", pady=5)
        self.btn_space = Button(toolbar_frame, text="Space", font=("Helvetica", 20, "bold"), command=self.on_space_press)
        self.btn_space.grid(row=1, column=0, sticky="nsew", pady=5)
        self.btn_backspace = Button(toolbar_frame, text="Backspace", font=("Helvetica", 20, "bold"), command=self.on_backspace_press)
        self.btn_backspace.grid(row=2, column=0, sticky="nsew", pady=5)
        self.btn_clear = Button(toolbar_frame, text="Clear", font=("Helvetica", 20, "bold"), command=self.on_clear_press)
        self.btn_clear.grid(row=3, column=0, sticky="nsew", pady=5)

        # Create the suggestion buttons area.
        suggestions_frame = Frame(self.window, bg="#f0f0f0")
        suggestions_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        suggestions_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.suggestion_buttons = []
        for i in range(3):
            btn = Button(suggestions_frame, text="---", font=("Helvetica", 14), state="disabled")
            btn.grid(row=0, column=i, sticky="ew", padx=5)
            self.suggestion_buttons.append(btn)

        # Create the sentence bar at the bottom.
        self.sentence_bar = Label(self.window, text="", font=("Courier", 40, "bold"), bg="white", anchor="w", relief="sunken", padx=10)
        self.sentence_bar.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # A list of all buttons that can be clicked with a pinch gesture.
        self.clickable_buttons = [self.btn_switch_mode, self.btn_space, self.btn_backspace, self.btn_clear] + self.suggestion_buttons
        
        # Create the virtual cursor on the canvas.
        self.cursor = self.canvas.create_oval(0, 0, 0, 0, fill=self.CURSOR_COLOR_IDLE, outline="")

    def _setup_key_bindings(self):
        """
        Binds keyboard shortcuts to their respective functions for easier testing and use.
        """
        self.window.bind("<space>", lambda e: self.on_space_press())
        self.window.bind("<BackSpace>", lambda e: self.on_backspace_press())
        self.window.bind("<c>", lambda e: self.on_clear_press())
        self.window.bind("<q>", lambda e: self.on_close())
        self.window.bind("<m>", lambda e: self.on_switch_mode())

    def update(self):
        """
        The main application loop. It reads the webcam, processes gestures, and
        gives the TTS engine time to run.
        """
        ret, frame = self.cap.read()
        if not ret:
            # If the camera fails, try again after a short delay.
            self.window.after(10, self.update)
            return
            
        # Only process frames once the window is fully drawn to get correct dimensions.
        if self.window.winfo_width() > 1:
            frame = cv2.flip(frame, 1) # Flip for a mirror-like view.
            tracker_output = self.tracker.process_frame(frame)
            self._handle_gestures(tracker_output)
        
        # Give the TTS engine a slice of time to process its speech queue.
        self.tts_engine.iterate()
        
        # Schedule the next update.
        self.window.after(10, self.update)

    def _map_coords(self, hand_coords):
        """
        Scales hand coordinates from the camera's resolution to the window's resolution,
        applying a sensitivity multiplier to make the cursor feel faster.
        """
        if not hand_coords: return None
        cam_x, cam_y = hand_coords
        
        win_width = self.window.winfo_width()
        win_height = self.window.winfo_height()

        # Use numpy.interp for robust linear scaling.
        mapped_x = np.interp(cam_x, [self.FRAME_PADDING, self.cam_width - self.FRAME_PADDING], [0, win_width])
        mapped_y = np.interp(cam_y, [self.FRAME_PADDING, self.cam_height - self.FRAME_PADDING], [0, win_height])
        
        # Use numpy.clip to ensure the cursor stays within the window boundaries.
        mapped_x = np.clip(mapped_x, 0, win_width)
        mapped_y = np.clip(mapped_y, 0, win_height)

        return int(mapped_x), int(mapped_y)

    def _handle_gestures(self, tracker_output):
        """
        The main logic hub. Processes the gesture data from the HandTracker and
        updates the UI state (drawing, moving, pinching).
        """
        gesture = tracker_output["gesture"]
        win_coords = self._map_coords(tracker_output["cursor_coords"])

        if win_coords:
            # Apply exponential moving average for smoother cursor movement.
            self.smooth_x = int(self.SMOOTHING_ALPHA * win_coords[0] + (1 - self.SMOOTHING_ALPHA) * self.smooth_x)
            self.smooth_y = int(self.SMOOTHING_ALPHA * win_coords[1] + (1 - self.SMOOTHING_ALPHA) * self.smooth_y)
            
            # Check for button hovers in every frame.
            self._update_button_hovers(self.smooth_x, self.smooth_y)
            
            # Translate window coordinates to be relative to the canvas for drawing.
            canvas_x = self.smooth_x - self.canvas.winfo_x()
            canvas_y = self.smooth_y - self.canvas.winfo_y()
            
            # Move the virtual cursor on the canvas.
            self.canvas.coords(self.cursor, canvas_x - self.CURSOR_SIZE, canvas_y - self.CURSOR_SIZE, canvas_x + self.CURSOR_SIZE, canvas_y + self.CURSOR_SIZE)

            # --- Gesture Logic ---
            if gesture == "DRAW":
                self.canvas.itemconfig(self.cursor, fill=self.CURSOR_COLOR_DRAW)
                # Only draw if the cursor is inside the canvas boundaries.
                if self.last_x is not None and 0 < canvas_x < self.canvas.winfo_width() and 0 < canvas_y < self.canvas.winfo_height():
                    self.canvas.create_line(self.last_x, self.last_y, canvas_x, canvas_y, fill=self.DRAWING_COLOR, width=self.BRUSH_SIZE, capstyle=tk.ROUND, tags="drawing")
                self.last_x, self.last_y = canvas_x, canvas_y
            else: # Handles MOVE, PINCH, and NONE gestures.
                self.canvas.itemconfig(self.cursor, fill=self.CURSOR_COLOR_IDLE)
                # If we were just drawing and switched to MOVE, it's time to recognize.
                if gesture == "MOVE" and self.last_x is not None:
                    self._recognize_character()
                # If the gesture is a pinch, handle the click.
                if gesture == "PINCH":
                    self._handle_pinch_click(win_coords)
                # "Lift the pen" by resetting the last coordinates.
                self.last_x, self.last_y = None, None

    def _recognize_character(self):
        """
        The core prediction pipeline: preprocesses the canvas drawing, feeds it to the
        active CNN model, updates the UI, speaks the result, and schedules a delayed clear.
        """
        image = self._preprocess_canvas_image()
        if image is None:
            self._clear_drawing_canvas()
            return

        # Select the correct model and labels based on the current mode.
        if self.current_mode == "ALPHABETS":
            model_to_use = self.alphabets_model
            labels_to_use = self.ALPHABET_LABELS
        else:
            model_to_use = self.numbers_model
            labels_to_use = self.NUMBER_LABELS

        # Make the prediction.
        prediction = model_to_use.predict(image)
        predicted_index = np.argmax(prediction)
        
        if predicted_index < len(labels_to_use):
            predicted_char = labels_to_use[predicted_index]
            print(f"Mode: {self.current_mode} | Predicted: {predicted_char}")
            
            # Add the speech task to the TTS engine's queue.
            self.tts_engine.say(predicted_char)
            
            # Update the sentence bar.
            self.sentence_text += predicted_char
            self.sentence_bar.config(text=self.sentence_text)
        else:
            print(f"Error: Predicted index {predicted_index} is out of bounds.")
        
        # Schedule the canvas to be cleared after a delay.
        self.window.after(self.CLEAR_CANVAS_DELAY_MS, self._clear_drawing_canvas)

    def _clear_drawing_canvas(self):
        """
        Clears only the drawing lines (items tagged with "drawing") from the canvas.
        """
        self.canvas.delete("drawing")

    def _preprocess_canvas_image(self):
        """
        The critical image processing pipeline. Converts the Tkinter canvas drawing
        into a standardized 28x28 image that the CNN model can understand.
        """
        # 1. Save the canvas content as a temporary postscript file.
        self.canvas.postscript(file="temp_char.eps", colormode='color')
        img = None
        try:
            # 2. Use Pillow to open the postscript file (requires Ghostscript).
            with Image.open("temp_char.eps") as img_eps:
                img = img_eps.convert('L') # Convert to grayscale.
        except Exception as e:
            print(f"Error processing EPS file: {e}. Ensure Ghostscript is installed.")
            return None
        finally:
            # 3. Clean up the temporary file.
            if os.path.exists("temp_char.eps"):
                try: os.remove("temp_char.eps")
                except PermissionError: pass
        if img is None: return None

        # 4. Convert to NumPy array and invert colors (white character on black background).
        img_np = np.array(img)
        img_np = cv2.bitwise_not(img_np)

        # 5. Find the bounding box of the character to remove empty space.
        contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        # 6. Crop the image to the character.
        x, y, w, h = cv2.boundingRect(contours[0])
        char_img = img_np[y:y+h, x:x+w]

        # 7. Add padding to make the image square, preserving aspect ratio.
        side_len = max(w, h)
        padding_top = (side_len - h) // 2
        padding_bottom = side_len - h - padding_top
        padding_left = (side_len - w) // 2
        padding_right = side_len - w - padding_left
        char_img = cv2.copyMakeBorder(char_img, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=0)

        # 8. Resize to 20x20 and add a 4-pixel border to standardize the size and position.
        char_img = cv2.resize(char_img, (20, 20), interpolation=cv2.INTER_AREA)
        final_img = cv2.copyMakeBorder(char_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

        # 9. Normalize pixel values to be between 0 and 1.
        final_img = final_img.astype('float32') / 255.0
        
        # 10. Reshape the image to the format the model expects: (1, 28, 28, 1).
        final_img = np.reshape(final_img, (1, self.IMG_SIZE, self.IMG_SIZE, 1))
        return final_img

    def _update_button_hovers(self, cursor_x, cursor_y):
        """
        Checks if the cursor is over any button and updates its background color
        to provide visual feedback.
        """
        currently_hovered = None
        for button in self.clickable_buttons:
            if button.winfo_exists() and button['state'] != 'disabled':
                # Get button coordinates relative to the window.
                win_root_x = self.window.winfo_rootx()
                win_root_y = self.window.winfo_rooty()
                x1 = button.winfo_rootx() - win_root_x
                y1 = button.winfo_rooty() - win_root_y
                x2 = x1 + button.winfo_width()
                y2 = y1 + button.winfo_height()
                # Check if the cursor is within the button's bounds.
                if x1 < cursor_x < x2 and y1 < cursor_y < y2:
                    currently_hovered = button
                    break
        
        # Update colors only when the hovered button changes to prevent flickering.
        if self.hovered_button != currently_hovered:
            # Reset the previously hovered button to its default color.
            if self.hovered_button is not None and self.hovered_button.winfo_exists():
                self.hovered_button.config(bg=self.BUTTON_DEFAULT_COLOR)
            # Highlight the new button.
            if currently_hovered is not None:
                currently_hovered.config(bg=self.BUTTON_HIGHLIGHT_COLOR)
            self.hovered_button = currently_hovered

    def _handle_pinch_click(self, pinch_coords):
        """
        Triggers a button's command if a pinch occurs while hovering over it.
        """
        # Check for the cooldown flag and if a button is currently hovered.
        if not self.pinch_active and self.hovered_button:
            self.pinch_active = True
            self.hovered_button.invoke() # Programmatically "click" the button.
            # Start the cooldown timer to prevent multiple clicks.
            self.window.after(self.PINCH_COOLDOWN_MS, self.reset_pinch)

    def reset_pinch(self):
        """
        Resets the pinch cooldown flag after the cooldown period.
        """
        self.pinch_active = False

    def on_switch_mode(self):
        """
        Toggles between "ALPHABETS" and "NUMBERS" recognition modes.
        """
        if self.current_mode == "ALPHABETS":
            self.current_mode = "NUMBERS"
            self.btn_switch_mode.config(text="Mode: Nums")
        else:
            self.current_mode = "ALPHABETS"
            self.btn_switch_mode.config(text="Mode: Alpha")
        print(f"Switched to {self.current_mode} mode.")

    def on_space_press(self):
        """
        Adds a space to the sentence bar.
        """
        self.sentence_text += " "
        self.sentence_bar.config(text=self.sentence_text)

    def on_backspace_press(self):
        """
        Deletes the last character from the sentence bar.
        """
        self.sentence_text = self.sentence_text[:-1]
        self.sentence_bar.config(text=self.sentence_text)

    def on_clear_press(self):
        """
        Manually clears the drawing canvas.
        """
        self._clear_drawing_canvas()

    def on_close(self, event=None):
        """
        Robustly closes the application and releases all resources.
        """
        print("Closing application...")
        # Stop the TTS engine if it's running.
        if self.tts_engine and self.tts_engine._inLoop:
            self.tts_engine.endLoop()
        # Release the camera.
        if self.cap and self.cap.isOpened():
            self.cap.release()
        # This function is called by main.py, which will handle destroying the window.

# --- Main Execution Block ---
if __name__ == "__main__":
    # This block is for testing this file directly.
    # The main entry point for the full application is main.py.
    
    # IMPORTANT: Set the path to your Ghostscript installation.
    # This is required for Pillow to read the .eps file from the canvas.
    EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe'
    
    root = tk.Tk()
    app = AirPredictApp(root, "AirPredict (Standalone Test)")
    
    # When testing standalone, we need to handle the window close event differently.
    def on_standalone_close():
        app.on_close()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_standalone_close)
    root.mainloop()
