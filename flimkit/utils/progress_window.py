import tkinter as tk
from tkinter import ttk
import threading

class ProgressWindow:
    def __init__(self, parent, task_name="Working..."):
        self.parent = parent
        self.cancelled = threading.Event()
        self.top = tk.Toplevel(parent)
        self.top.title(task_name)
        self.top.geometry("400x120")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)

        ttk.Label(self.top, text=task_name, font=("Arial", 12, "bold")).pack(pady=(16, 8))
        self.progress = ttk.Progressbar(self.top, orient="horizontal", length=340, mode="determinate")
        self.progress.pack(pady=6)
        self.status = tk.StringVar(value="Starting...")
        ttk.Label(self.top, textvariable=self.status).pack(pady=(2, 8))
        self.btn_cancel = ttk.Button(self.top, text="Cancel", command=self.cancel)
        self.btn_cancel.pack(pady=(0, 8))

    def set_progress(self, value, maximum=None):
        if maximum is not None:
            self.progress["maximum"] = maximum
        self.progress["value"] = value
        self.status.set(f"{int(value)}/{int(self.progress['maximum'])} completed")
        self.top.update_idletasks()

    def set_status(self, msg):
        self.status.set(msg)
        self.top.update_idletasks()

    def cancel(self):
        self.cancelled.set()
        self.set_status("Cancelling...")
        self.btn_cancel.config(state="disabled")

    def close(self):
        self.top.grab_release()
        self.top.destroy()

    def was_cancelled(self):
        return self.cancelled.is_set()
