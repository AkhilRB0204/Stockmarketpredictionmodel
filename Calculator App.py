import tkinter as tk
from math import sin, cos, tan, log, sqrt, pi, e
from tkinter import messagebox

def click(button_text):
    """Handle button click events."""
    current = entry.get()
    if button_text == "C":
        entry.delete(0, tk.END)
    elif button_text == "=":
        try:
            result = eval(current)
            entry.delete(0, tk.END)
            entry.insert(0, str(result))
        except Exception as e:
            messagebox.showerror("Error", "Invalid Input")
    else:
        entry.insert(tk.END, button_text)

# Create the main window
root = tk.Tk()
root.title("Advanced Calculator")

# Create the display entry field
entry = tk.Entry(root, width=25, font=("Arial", 18), borderwidth=2, relief=tk.RIDGE, justify="right")
entry.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

# Define button layout
buttons = [
    ["7", "8", "9", "/", "C"],
    ["4", "5", "6", "*", "("],
    ["1", "2", "3", "-", ")"],
    ["0", ".", "=", "+", "^"],
    ["sin", "cos", "tan", "log", "sqrt"],
    ["pi", "e", "//", "%", "**"],
]

# Create buttons dynamically
for i, row in enumerate(buttons):
    for j, button_text in enumerate(row):
        if button_text in ["sin", "cos", "tan", "log", "sqrt"]:
            action = lambda text=button_text: entry.insert(tk.END, f"{text}(")
        elif button_text == "^":
            action = lambda: entry.insert(tk.END, "**")
        else:
            action = lambda text=button_text: click(text)

        tk.Button(root, text=button_text, width=5, height=2, font=("Arial", 14),
                  command=action).grid(row=i + 1, column=j, padx=5, pady=5)

# Run the application
root.mainloop()
