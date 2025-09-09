# --- main.py (Definitive Shutdown Logic) ---

import tkinter as tk
from homepage import HomePage
from air_predict_app import AirPredictApp

def launch_air_predict_app(root):
    """Hides the homepage and launches the main AirPredict application."""
    root.withdraw()
    app_window = tk.Toplevel(root)
    
    app = AirPredictApp(app_window, "AirPredict")
    
    def on_app_close():
        # Call the app's cleanup function first
        app.on_close()
        # Then destroy the Toplevel window
        app_window.destroy()
        # And finally, re-show the homepage
        root.deiconify()
        
    app_window.protocol("WM_DELETE_WINDOW", on_app_close)

# --- Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk()
    
    # Pass the root window and the launch function to the HomePage
    home_page = HomePage(root, lambda: launch_air_predict_app(root))
    
    root.mainloop()