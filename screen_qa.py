#!/usr/bin/env python3
"""
screen_qa.py - Select a region of your screen containing a question, send it to
Google Gemini, and show the formatted answer in a new window.

Setup
-----
    pip install google-genai pillow
    export GEMINI_API_KEY="your-key"      # free key: https://aistudio.google.com/apikey
    # Windows (PowerShell):  $env:GEMINI_API_KEY="your-key"

Run
---
    python screen_qa.py

Optional environment variables
    GEMINI_MODEL   model name (default: gemini-2.5-flash)
"""

import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from PIL import ImageEnhance, ImageGrab, ImageTk

try:
    from google import genai
except ImportError:
    sys.exit("Missing dependency. Run:  pip install google-genai pillow")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PROMPT = (
    "The image is a screenshot containing a question (possibly with answer options "
    "or code). Read it carefully and answer it. Give the answer first, then a brief "
    "explanation. Use Markdown for formatting."
)


# --------------------------------------------------------------------------- #
# Step 1: screenshot of a user-selected region
# --------------------------------------------------------------------------- #
def select_region(root):
    """Show a dimmed full-screen overlay; the user drags a rectangle.
    Returns a PIL image of that region, or None if cancelled."""
    full = ImageGrab.grab()

    top = tk.Toplevel(root)
    top.attributes("-fullscreen", True)
    top.attributes("-topmost", True)
    top.update_idletasks()

    sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
    sx, sy = full.width / sw, full.height / sh  # handles HiDPI scaling

    dimmed = ImageEnhance.Brightness(full.resize((sw, sh))).enhance(0.45)
    photo = ImageTk.PhotoImage(dimmed, master=top)

    canvas = tk.Canvas(top, width=sw, height=sh, highlightthickness=0,
                       cursor="crosshair", bg="black")
    canvas.pack(fill="both", expand=True)
    canvas.photo = photo  # keep a reference
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.create_text(sw // 2, 30, fill="white", font=("TkDefaultFont", 16),
                       text="Drag to select the question   •   Esc to cancel")

    state = {"x0": 0, "y0": 0, "rect": None, "box": None}

    def on_press(e):
        state["x0"], state["y0"] = e.x, e.y
        state["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                outline="#00e0ff", width=2)

    def on_drag(e):
        if state["rect"]:
            canvas.coords(state["rect"], state["x0"], state["y0"], e.x, e.y)

    def on_release(e):
        x0, x1 = sorted((state["x0"], e.x))
        y0, y1 = sorted((state["y0"], e.y))
        if x1 - x0 > 5 and y1 - y0 > 5:
            state["box"] = (x0, y0, x1, y1)
        top.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    top.bind("<Escape>", lambda e: top.destroy())
    top.focus_force()

    root.wait_window(top)

    if state["box"] is None:
        return None
    x0, y0, x1, y1 = state["box"]
    return full.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))


# --------------------------------------------------------------------------- #
# Step 2 + 3a: send to Gemini and get the reply
# --------------------------------------------------------------------------- #
def ask_gemini(image):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set the GEMINI_API_KEY environment variable.\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL, contents=[image, PROMPT])
    return response.text or "(The model returned an empty response.)"


# --------------------------------------------------------------------------- #
# Step 3b: display with formatting (lightweight Markdown renderer for Tk)
# --------------------------------------------------------------------------- #
INLINE = re.compile(r"(\*\*\*.+?\*\*\*|\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*)")


class MarkdownText(tk.Text):
    def __init__(self, master, **kw):
        super().__init__(master, wrap="word", padx=16, pady=12, borderwidth=0,
                         highlightthickness=0, bg="#ffffff", fg="#1f2328",
                         spacing1=2, spacing3=4, **kw)
        base = tkfont.nametofont("TkDefaultFont").copy()
        base.configure(size=12)
        self.f_base = base
        self.f_bold = base.copy(); self.f_bold.configure(weight="bold")
        self.f_ital = base.copy(); self.f_ital.configure(slant="italic")
        self.f_bi = base.copy(); self.f_bi.configure(weight="bold", slant="italic")
        self.f_mono = tkfont.nametofont("TkFixedFont").copy(); self.f_mono.configure(size=11)
        self.f_h = {
            1: base.copy(), 2: base.copy(), 3: base.copy(),
        }
        for lvl, size in ((1, 20), (2, 17), (3, 14)):
            self.f_h[lvl].configure(size=size, weight="bold")

        self.configure(font=self.f_base)
        self.tag_configure("bold", font=self.f_bold)
        self.tag_configure("italic", font=self.f_ital)
        self.tag_configure("bolditalic", font=self.f_bi)
        self.tag_configure("code", font=self.f_mono, background="#eef0f3", foreground="#b3261e")
        self.tag_configure("codeblock", font=self.f_mono, background="#f3f4f6",
                           lmargin1=16, lmargin2=16, rmargin=16, spacing1=0, spacing3=0)
        self.tag_configure("table", font=self.f_mono, lmargin1=8, lmargin2=8)
        self.tag_configure("quote", foreground="#57606a", lmargin1=18, lmargin2=18)
        self.tag_configure("hr", foreground="#aaaaaa")
        for lvl in (1, 2, 3):
            self.tag_configure(f"h{lvl}", font=self.f_h[lvl], spacing1=10, spacing3=6)
        for lvl in range(6):
            self.tag_configure(f"list{lvl}", lmargin1=14 + 22 * lvl, lmargin2=34 + 22 * lvl)

    # -- inline formatting -------------------------------------------------- #
    def _inline(self, text, tags=()):
        pos = 0
        for m in INLINE.finditer(text):
            self.insert("end", text[pos:m.start()], tags)
            tok = m.group(0)
            if tok.startswith("***"):
                self.insert("end", tok[3:-3], tags + ("bolditalic",))
            elif tok.startswith("**"):
                self.insert("end", tok[2:-2], tags + ("bold",))
            elif tok.startswith("`"):
                self.insert("end", tok[1:-1], tags + ("code",))
            else:
                self.insert("end", tok[1:-1], tags + ("italic",))
            pos = m.end()
        self.insert("end", text[pos:], tags)

    # -- block formatting --------------------------------------------------- #
    def render(self, md):
        self.configure(state="normal")
        self.delete("1.0", "end")

        in_code, code_buf = False, []
        for line in md.splitlines():
            if line.strip().startswith("```"):
                if in_code:
                    self.insert("end", "\n".join(code_buf) + "\n", "codeblock")
                    code_buf, in_code = [], False
                else:
                    in_code = True
                continue
            if in_code:
                code_buf.append(line)
                continue

            if not line.strip():
                self.insert("end", "\n")
                continue

            if m := re.match(r"^(#{1,6})\s+(.*)", line):
                lvl = min(len(m.group(1)), 3)
                self._inline(m.group(2), (f"h{lvl}",))
            elif re.match(r"^\s*([-*_])\1{2,}\s*$", line):
                self.insert("end", "─" * 60, "hr")
            elif m := re.match(r"^(\s*)[-*+]\s+(.*)", line):
                lvl = min(len(m.group(1)) // 2, 5)
                self.insert("end", "•  ", f"list{lvl}")
                self._inline(m.group(2), (f"list{lvl}",))
            elif m := re.match(r"^(\s*)(\d+)[.)]\s+(.*)", line):
                lvl = min(len(m.group(1)) // 2, 5)
                self.insert("end", f"{m.group(2)}.  ", f"list{lvl}")
                self._inline(m.group(3), (f"list{lvl}",))
            elif line.startswith(">"):
                self._inline(line.lstrip("> ").rstrip(), ("quote",))
            elif line.lstrip().startswith("|"):
                if not re.match(r"^\s*\|[\s:|-]+\|?\s*$", line):  # skip |---| rows
                    self.insert("end", line.strip(), "table")
                else:
                    continue
            else:
                self._inline(line)
            self.insert("end", "\n")

        if in_code and code_buf:  # unterminated fence
            self.insert("end", "\n".join(code_buf) + "\n", "codeblock")

        self.configure(state="disabled")


class ResultWindow(tk.Toplevel):
    def __init__(self, root, image):
        super().__init__(root)
        self.title("Gemini answer")
        self.geometry("760x640")
        self.raw = ""
        self.protocol("WM_DELETE_WINDOW", root.destroy)

        # Thumbnail of the captured region
        thumb = image.copy()
        thumb.thumbnail((720, 140))
        self._thumb = ImageTk.PhotoImage(thumb, master=self)
        tk.Label(self, image=self._thumb, bg="#e9ecef", pady=6).pack(fill="x")

        # Answer area
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.text = MarkdownText(body)
        scroll = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        # Buttons
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Copy answer", command=self.copy).pack(side="left")
        ttk.Button(bar, text="Close", command=root.destroy).pack(side="right")

        self.set_text("⏳ Asking Gemini…")
        self.lift()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))

    def set_text(self, md):
        self.raw = md
        self.text.render(md)

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.raw)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    root = tk.Tk()
    root.withdraw()

    image = select_region(root)
    if image is None:
        print("Cancelled.")
        root.destroy()
        return

    win = ResultWindow(root, image)
    results = queue.Queue()

    def worker():
        try:
            results.put(("ok", ask_gemini(image)))
        except Exception as exc:  # network, quota, bad key, etc.
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                msg += "\n\nYou've hit the rate/quota limit. Wait a bit and try again."
            results.put(("err", f"**Error**\n\n{msg}"))

    threading.Thread(target=worker, daemon=True).start()

    def poll():
        try:
            _, text = results.get_nowait()
            win.set_text(text)
        except queue.Empty:
            root.after(150, poll)

    poll()
    root.mainloop()


if __name__ == "__main__":
    main()
