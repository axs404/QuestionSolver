#!/usr/bin/env python3
"""
screen_qa.py - Press a global hotkey, drag a box around a question on your screen,
and get Gemini's answer in a new window, with Markdown AND LaTeX math rendered.

Setup
-----
    pip install google-genai pillow matplotlib pynput
    export GEMINI_API_KEY="your-key"      # free key: https://aistudio.google.com/apikey
    # Windows (PowerShell):  $env:GEMINI_API_KEY="your-key"

Run (leave it running in a terminal)
---
    python screen_qa.py            # wait for the hotkey
    python screen_qa.py --now      # also capture immediately on start

Hotkeys (pynput syntax, override with environment variables)
    HOTKEY       default "<ctrl>+<alt>+g"   -> start a capture
    QUIT_HOTKEY  default "<ctrl>+<alt>+x"   -> quit the program

Other environment variables
    GEMINI_MODEL     default "gemini-2.5-flash"
    MATH_FONTSET     default "dejavusans" (also: cm, stix, stixsans, dejavuserif)
"""

import io
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image, ImageEnhance, ImageGrab, ImageTk
    from google import genai
except ImportError as exc:
    sys.exit(f"Missing dependency ({exc.name}). Run:\n"
             "  pip install google-genai pillow matplotlib pynput")

matplotlib.rcParams["mathtext.fontset"] = os.environ.get("MATH_FONTSET", "dejavusans")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
HOTKEY = os.environ.get("HOTKEY", "<ctrl>+<alt>+g")
QUIT_HOTKEY = os.environ.get("QUIT_HOTKEY", "<ctrl>+<alt>+x")

PROMPT = (
    "The image is a screenshot containing a question (possibly with answer options, "
    "code or equations). Read it carefully and answer it. Give the answer first, then "
    "a brief explanation. Use Markdown for formatting. Write ALL math in LaTeX: "
    "$...$ for inline and $$...$$ on their own lines for display equations. Use only "
    "basic commands (\\frac, \\sqrt, ^, _, \\sum, \\int, \\lim, Greek letters, "
    "\\left( \\right), \\mathbb, \\mathrm). Do NOT use matrix/cases/array environments; "
    "for those, use a code block with plain text instead."
)


# --------------------------------------------------------------------------- #
# Step 1: screenshot of a user-selected region
# --------------------------------------------------------------------------- #
def select_region(root):
    """Dimmed full-screen overlay; the user drags a rectangle.
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
# Step 2: send to Gemini
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
# LaTeX helpers (rendered with matplotlib's mathtext - no TeX install needed)
# --------------------------------------------------------------------------- #
class UnsupportedTeX(Exception):
    pass


_WRAP_ENVS = r"align\*?|aligned|equation\*?|gather\*?|gathered|eqnarray\*?|multline\*?|split"


def prep_tex(tex):
    """Convert common LaTeX into what mathtext understands.
    Returns a list of lines (multi-line align/gather blocks become several)."""
    tex = tex.strip()
    tex = re.sub(r"\\(?:begin|end)\{(?:%s)\}" % _WRAP_ENVS, "", tex)
    tex = re.sub(r"\\(?:label|tag)\{[^{}]*\}|\\nonumber|\\notag|\\displaystyle|"
                 r"\\textstyle|\\limits", "", tex)
    tex = re.sub(r"\\[dt]frac", r"\\frac", tex)
    tex = re.sub(r"\\(?:text|textrm|textit|textbf|mbox|operatorname)\{([^{}]*)\}",
                 lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}", tex)
    tex = tex.replace(r"\&", "\x00").replace("&", " ").replace("\x00", r"\&")
    if re.search(r"\\begin\{", tex):          # matrices, cases, arrays...
        raise UnsupportedTeX(tex)
    lines = [ln.strip() for ln in re.split(r"\\\\", tex)]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise UnsupportedTeX(tex)
    return lines


def split_blocks(md):
    """Yield ('code'|'math'|'line', text) so fenced code and display math
    (which can span several lines) are handled before line-by-line rendering."""
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line, s = lines[i], lines[i].strip()

        if s.startswith("```"):
            buf, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield "code", "\n".join(buf)
            continue

        if s.startswith("$$") and s.count("$$") < 2:        # multi-line $$ ... $$
            buf, i = [s[2:]], i + 1
            while i < len(lines):
                t = lines[i].strip()
                i += 1
                if t.endswith("$$"):
                    buf.append(t[:-2])
                    break
                buf.append(t)
            yield "math", "\n".join(buf)
            continue
        if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
            yield "math", s[2:-2]
            i += 1
            continue

        if s.startswith("\\[") and not s.endswith("\\]"):    # multi-line \[ ... \]
            buf, i = [s[2:]], i + 1
            while i < len(lines):
                t = lines[i].strip()
                i += 1
                if t.endswith("\\]"):
                    buf.append(t[:-2])
                    break
                buf.append(t)
            yield "math", "\n".join(buf)
            continue
        if s.startswith("\\[") and s.endswith("\\]") and len(s) >= 4:
            yield "math", s[2:-2]
            i += 1
            continue

        m = re.match(r"\\begin\{(%s)\}" % _WRAP_ENVS, s)
        if m:                                               # bare \begin{align} ...
            buf, end = [s], "\\end{%s}" % m.group(1)
            i += 1
            while end not in buf[-1] and i < len(lines):
                buf.append(lines[i].strip())
                i += 1
            yield "math", "\n".join(buf)
            continue

        yield "line", line
        i += 1


# --------------------------------------------------------------------------- #
# Step 3: display with formatting (Markdown + LaTeX) in a Tk Text widget
# --------------------------------------------------------------------------- #
INLINE = re.compile(
    r"(\*\*\*.+?\*\*\*"                                   # bold italic
    r"|\*\*.+?\*\*"                                       # bold
    r"|`[^`]+`"                                           # inline code
    r"|\$\$.+?\$\$"                                       # $$ inline $$
    r"|(?<![\\$\w])\$(?!\s)[^$\n]+?(?<!\s)\$(?![\d$])"    # $ math $ (not currency)
    r"|\\\(.+?\\\)"                                       # \( math \)
    r"|\*[^*\s][^*]*?\*)"                                 # italic
)


class MarkdownText(tk.Text):
    def __init__(self, master, **kw):
        super().__init__(master, wrap="word", padx=16, pady=12, borderwidth=0,
                         highlightthickness=0, bg="#ffffff", fg="#1f2328",
                         spacing1=2, spacing3=4, **kw)
        self.dpi = max(72.0, min(200.0, float(master.winfo_fpixels("1i"))))
        self._math_cache = {}

        base = tkfont.nametofont("TkDefaultFont").copy()
        base.configure(size=12)
        self.f_base = base
        self.f_bold = base.copy(); self.f_bold.configure(weight="bold")
        self.f_ital = base.copy(); self.f_ital.configure(slant="italic")
        self.f_bi = base.copy(); self.f_bi.configure(weight="bold", slant="italic")
        self.f_mono = tkfont.nametofont("TkFixedFont").copy(); self.f_mono.configure(size=11)
        self.f_h = {lvl: base.copy() for lvl in (1, 2, 3)}
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
        self.tag_configure("center", justify="center")
        for lvl in (1, 2, 3):
            self.tag_configure(f"h{lvl}", font=self.f_h[lvl], spacing1=10, spacing3=6)
        for lvl in range(6):
            self.tag_configure(f"list{lvl}", lmargin1=14 + 22 * lvl, lmargin2=34 + 22 * lvl)

    # -- LaTeX -> image ------------------------------------------------------ #
    def _math_image(self, tex, display):
        key = (tex, display)
        if key not in self._math_cache:
            fig = Figure(dpi=self.dpi)
            FigureCanvasAgg(fig)
            fig.text(0, 0, f"${tex}$", fontsize=15 if display else 12, color="#1f2328")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=self.dpi, transparent=True,
                        bbox_inches="tight", pad_inches=0.04)
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            self._math_cache[key] = ImageTk.PhotoImage(img, master=self)
        return self._math_cache[key]

    def _inline_math(self, tex, tags=()):
        try:
            lines = prep_tex(tex)
            if len(lines) != 1:
                raise UnsupportedTeX(tex)
            self.image_create("end", image=self._math_image(lines[0], False),
                              align="center", padx=2)
        except Exception:
            self.insert("end", tex, tuple(tags) + ("code",))   # show raw TeX

    def _display_math(self, tex):
        try:
            photos = [self._math_image(ln, True) for ln in prep_tex(tex)]
        except Exception:
            self.insert("end", tex.strip() + "\n", "codeblock")
            return
        for photo in photos:
            start = self.index("end-1c")
            self.image_create("end", image=photo, pady=4)
            self.tag_add("center", start, "end-1c")
            self.insert("end", "\n")

    # -- inline formatting --------------------------------------------------- #
    def _inline(self, text, tags=()):
        pos = 0
        for m in INLINE.finditer(text):
            self.insert("end", text[pos:m.start()], tags)
            tok = m.group(0)
            if tok.startswith("***"):
                self._inline(tok[3:-3], tags + ("bolditalic",))
            elif tok.startswith("**"):
                self._inline(tok[2:-2], tags + ("bold",))
            elif tok.startswith("`"):
                self.insert("end", tok[1:-1], tags + ("code",))
            elif tok.startswith("$$"):
                self._inline_math(tok[2:-2], tags)
            elif tok.startswith("$"):
                self._inline_math(tok[1:-1], tags)
            elif tok.startswith("\\("):
                self._inline_math(tok[2:-2], tags)
            else:
                self._inline(tok[1:-1], tags + ("italic",))
            pos = m.end()
        self.insert("end", text[pos:], tags)

    # -- block formatting ---------------------------------------------------- #
    def render(self, md):
        self.configure(state="normal")
        self.delete("1.0", "end")

        for kind, content in split_blocks(md):
            if kind == "code":
                self.insert("end", content + "\n", "codeblock")
                continue
            if kind == "math":
                self._display_math(content)
                continue

            line = content
            if not line.strip():
                self.insert("end", "\n")
                continue

            if m := re.match(r"^(#{1,6})\s+(.*)", line):
                self._inline(m.group(2), (f"h{min(len(m.group(1)), 3)}",))
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
                if re.match(r"^\s*\|[\s:|-]+\|?\s*$", line):   # skip |---| rows
                    continue
                self.insert("end", line.strip(), "table")
            else:
                self._inline(line)
            self.insert("end", "\n")

        self.configure(state="disabled")


class ResultWindow(tk.Toplevel):
    def __init__(self, root, image):
        super().__init__(root)
        self.title("Gemini answer")
        self.geometry("780x660")
        self.image = image
        self.raw = ""
        self.results = queue.Queue()

        thumb = image.copy()
        thumb.thumbnail((740, 140))
        self._thumb = ImageTk.PhotoImage(thumb, master=self)
        tk.Label(self, image=self._thumb, bg="#e9ecef", pady=6).pack(fill="x")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.text = MarkdownText(body)
        scroll = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Copy answer", command=self.copy).pack(side="left")
        ttk.Button(bar, text="Close", command=self.destroy).pack(side="right")
        self.bind("<Escape>", lambda e: self.destroy())

        self.set_text("⏳ Asking Gemini…")
        self.lift()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def start(self):
        """Run the API call in a thread so the UI never freezes."""
        def worker():
            try:
                self.results.put(ask_gemini(self.image))
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    msg += "\n\nRate/quota limit reached. Wait a bit and try again."
                self.results.put(f"**Error**\n\n{msg}")

        threading.Thread(target=worker, daemon=True).start()
        self._poll()

    def _poll(self):
        if not self.winfo_exists():
            return
        try:
            self.set_text(self.results.get_nowait())
        except queue.Empty:
            self.after(150, self._poll)

    def set_text(self, md):
        self.raw = md
        self.text.render(md)

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.raw)


# --------------------------------------------------------------------------- #
# Main: hidden Tk root + global hotkey listener
# --------------------------------------------------------------------------- #
def main():
    if sys.platform == "win32":          # sharp, correctly-scaled screenshots
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    try:
        from pynput import keyboard
    except ImportError:
        sys.exit("Missing dependency (pynput). Run:  pip install pynput")

    root = tk.Tk()
    root.withdraw()

    commands = queue.Queue()
    busy = {"capturing": False}

    def do_capture():
        try:
            image = select_region(root)
        finally:
            busy["capturing"] = False
        if image is not None:
            ResultWindow(root, image).start()

    def pump():
        try:
            while True:
                cmd = commands.get_nowait()
                if cmd == "capture" and not busy["capturing"]:
                    busy["capturing"] = True
                    root.after(150, do_capture)       # let the hotkey keys release
                elif cmd == "quit":
                    root.destroy()
                    return
        except queue.Empty:
            pass
        root.after(100, pump)

    # pynput runs callbacks in its own thread -> hand work to Tk via a queue
    listener = keyboard.GlobalHotKeys({
        HOTKEY: lambda: commands.put("capture"),
        QUIT_HOTKEY: lambda: commands.put("quit"),
    })
    listener.start()
    print(f"Ready.  Capture: {HOTKEY}   Quit: {QUIT_HOTKEY}   (Ctrl+C here also quits)")

    if "--now" in sys.argv:
        commands.put("capture")

    root.after(100, pump)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
