import io
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image, ImageEnhance, ImageGrab, ImageTk
    from google import genai
    from google.genai import types
except ImportError as exc:
    sys.exit(
        f"Missing dependency ({exc.name}). Run:\n"
        "  pip install google-genai pillow matplotlib pynput"
    )

matplotlib.rcParams["mathtext.fontset"] = os.environ.get("MATH_FONTSET", "dejavusans")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
HOTKEY = os.environ.get("HOTKEY", "<alt>+g")
QUIT_HOTKEY = os.environ.get("QUIT_HOTKEY", "<alt>+x")

# PROMPT = (
#     "The image is a screenshot containing a question (possibly with answer options, "
#     "code or equations). Read it carefully and answer it. Give the answer first, then "
#     "a SHORT explanation (a few sentences at most). Use Markdown for formatting. Write ALL math in LaTeX: "
#     "$...$ for inline and $$...$$ on their own lines for display equations. Use only "
#     "basic commands (\\frac, \\sqrt, ^, _, \\sum, \\int, \\lim, Greek letters, "
#     "\\left( \\right), \\mathbb, \\mathrm). Do NOT use matrix/cases/array environments; "
#     "for those, use a code block with plain text instead."
# )


# --------------------------------------------------------------------------- #
# Step 1: screenshot of a user-selected region
# --------------------------------------------------------------------------- #
def select_region(root, cancel_flag=None):
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

    canvas = tk.Canvas(
        top, width=sw, height=sh, highlightthickness=0, cursor="crosshair", bg="black"
    )
    canvas.pack(fill="both", expand=True)
    canvas.photo = photo  # keep a reference
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.create_text(
        sw // 2,
        30,
        fill="white",
        font=("TkDefaultFont", 16),
        text="Drag to select the question   •   Esc to cancel",
    )

    state = {"x0": 0, "y0": 0, "rect": None, "box": None}

    def on_press(e):
        state["x0"], state["y0"] = e.x, e.y
        state["rect"] = canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="#00e0ff", width=2
        )

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
    for widget in (top, canvas):
        widget.bind("<Escape>", lambda e: top.destroy())

    # Grab keyboard focus (Windows may refuse because the hotkey was pressed in
    # another app, so also watch the global Esc flag set by the pynput listener).
    top.lift()
    top.focus_force()
    canvas.focus_set()
    try:
        top.grab_set()
    except tk.TclError:
        pass

    def check_cancel():
        if not top.winfo_exists():
            return
        if cancel_flag is not None and cancel_flag.is_set():
            top.destroy()
        else:
            top.after(40, check_cancel)

    check_cancel()

    root.wait_window(top)

    if state["box"] is None:
        return None
    x0, y0, x1, y1 = state["box"]
    return full.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))


# --------------------------------------------------------------------------- #
# Step 2: send to Gemini
# --------------------------------------------------------------------------- #
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set the GEMINI_API_KEY environment variable.\n"
                "Get a free key at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def prepare_image(image):
    """Downscale + JPEG-encode: far smaller upload, same readability."""
    img = image.convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _configs():
    """Try a low 'thinking' level first (much faster); fall back to defaults if
    the model rejects it. Set THINKING=default to skip, or THINKING=high etc."""
    level = os.environ.get("THINKING", "MINIMAL").strip().lower()
    out = []
    if level not in ("", "default"):
        try:
            out.append(
                types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=level)
                )
            )
        except Exception:
            pass
    out.append(None)
    return out


def stream_gemini(image):
    """Yield the answer in pieces as Gemini generates it."""
    client = get_client()
    part = prepare_image(image)
    for cfg in _configs():
        got_text = False
        try:
            kwargs = {"model": MODEL, "contents": [part]}
            if cfg is not None:
                kwargs["config"] = cfg
            for chunk in client.models.generate_content_stream(**kwargs):
                if chunk.text:
                    got_text = True
                    yield chunk.text
            return
        except Exception:
            if got_text or cfg is None:
                raise  # real error, or last attempt: report it
            # otherwise retry once without the thinking config


# --------------------------------------------------------------------------- #
# LaTeX helpers (rendered with matplotlib's mathtext - no TeX install needed)
# --------------------------------------------------------------------------- #
class UnsupportedTeX(Exception):
    pass


_WRAP_ENVS = (
    r"align\*?|aligned|equation\*?|gather\*?|gathered|eqnarray\*?|multline\*?|split"
)


def prep_tex(tex):
    """Convert common LaTeX into what mathtext understands.
    Returns a list of lines (multi-line align/gather blocks become several)."""
    tex = tex.strip()
    tex = re.sub(r"\\(?:begin|end)\{(?:%s)\}" % _WRAP_ENVS, "", tex)
    tex = re.sub(
        r"\\(?:label|tag)\{[^{}]*\}|\\nonumber|\\notag|\\displaystyle|"
        r"\\textstyle|\\limits",
        "",
        tex,
    )
    tex = re.sub(r"\\[dt]frac", r"\\frac", tex)
    tex = re.sub(
        r"\\(?:text|textrm|textit|textbf|mbox|operatorname)\{([^{}]*)\}",
        lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}",
        tex,
    )
    tex = tex.replace(r"\&", "\x00").replace("&", " ").replace("\x00", r"\&")
    if re.search(r"\\begin\{", tex):  # matrices, cases, arrays...
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

        if s.startswith("$$") and s.count("$$") < 2:  # multi-line $$ ... $$
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

        if s.startswith("\\[") and not s.endswith("\\]"):  # multi-line \[ ... \]
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
        if m:  # bare \begin{align} ...
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
RESULT_BG = "#1e1e1e"
RESULT_SURFACE = "#252526"
RESULT_CONTROL = "#333337"
RESULT_FG = "#d4d4d4"
RESULT_MUTED = "#9da5b4"

INLINE = re.compile(
    r"(\*\*\*.+?\*\*\*"  # bold italic
    r"|\*\*.+?\*\*"  # bold
    r"|`[^`]+`"  # inline code
    r"|\$\$.+?\$\$"  # $$ inline $$
    r"|(?<![\\$\w])\$(?!\s)[^$\n]+?(?<!\s)\$(?![\d$])"  # $ math $ (not currency)
    r"|\\\(.+?\\\)"  # \( math \)
    r"|\*[^*\s][^*]*?\*)"  # italic
)


class MarkdownText(tk.Text):
    def __init__(self, master, **kw):
        super().__init__(
            master,
            wrap="word",
            padx=16,
            pady=12,
            borderwidth=0,
            highlightthickness=0,
            bg=RESULT_BG,
            fg=RESULT_FG,
            selectbackground="#264f78",
            selectforeground="#ffffff",
            spacing1=2,
            spacing3=4,
            **kw,
        )
        self.dpi = max(72.0, min(200.0, float(master.winfo_fpixels("1i"))))
        self._math_cache = {}

        base = tkfont.nametofont("TkDefaultFont").copy()
        base.configure(size=12)
        self.f_base = base
        self.f_bold = base.copy()
        self.f_bold.configure(weight="bold")
        self.f_ital = base.copy()
        self.f_ital.configure(slant="italic")
        self.f_bi = base.copy()
        self.f_bi.configure(weight="bold", slant="italic")
        self.f_mono = tkfont.nametofont("TkFixedFont").copy()
        self.f_mono.configure(size=11)
        self.f_h = {lvl: base.copy() for lvl in (1, 2, 3)}
        for lvl, size in ((1, 20), (2, 17), (3, 14)):
            self.f_h[lvl].configure(size=size, weight="bold")

        self.configure(font=self.f_base)
        self.tag_configure("bold", font=self.f_bold)
        self.tag_configure("italic", font=self.f_ital)
        self.tag_configure("bolditalic", font=self.f_bi)
        self.tag_configure(
            "code", font=self.f_mono, background=RESULT_CONTROL, foreground="#ce9178"
        )
        self.tag_configure(
            "codeblock",
            font=self.f_mono,
            background=RESULT_SURFACE,
            lmargin1=16,
            lmargin2=16,
            rmargin=16,
            spacing1=0,
            spacing3=0,
        )
        self.tag_configure("table", font=self.f_mono, lmargin1=8, lmargin2=8)
        self.tag_configure("quote", foreground=RESULT_MUTED, lmargin1=18, lmargin2=18)
        self.tag_configure("hr", foreground="#6e7681")
        self.tag_configure("center", justify="center")
        for lvl in (1, 2, 3):
            self.tag_configure(f"h{lvl}", font=self.f_h[lvl], spacing1=10, spacing3=6)
        for lvl in range(6):
            self.tag_configure(
                f"list{lvl}", lmargin1=14 + 22 * lvl, lmargin2=34 + 22 * lvl
            )

    # -- LaTeX -> image ------------------------------------------------------ #
    def _math_image(self, tex, display):
        key = (tex, display)
        if key not in self._math_cache:
            fig = Figure(dpi=self.dpi)
            FigureCanvasAgg(fig)
            fig.text(0, 0, f"${tex}$", fontsize=15 if display else 12, color=RESULT_FG)
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=self.dpi,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.04,
            )
            buf.seek(0)
            img = Image.open(buf).convert("RGBA")
            self._math_cache[key] = ImageTk.PhotoImage(img, master=self)
        return self._math_cache[key]

    def _inline_math(self, tex, tags=()):
        try:
            lines = prep_tex(tex)
            if len(lines) != 1:
                raise UnsupportedTeX(tex)
            self.image_create(
                "end", image=self._math_image(lines[0], False), align="center", padx=2
            )
        except Exception:
            self.insert("end", tex, tuple(tags) + ("code",))  # show raw TeX

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
            self.insert("end", text[pos : m.start()], tags)
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
                if re.match(r"^\s*\|[\s:|-]+\|?\s*$", line):  # skip |---| rows
                    continue
                self.insert("end", line.strip(), "table")
            else:
                self._inline(line)
            self.insert("end", "\n")

        self.configure(state="disabled")


class DarkScrollbar(tk.Canvas):
    def __init__(self, master, command):
        super().__init__(
            master,
            width=12,
            bg=RESULT_SURFACE,
            highlightthickness=0,
            borderwidth=0,
            takefocus=True,
        )
        self.command = command
        self.first = 0.0
        self.last = 1.0
        self._thumb_top = 0
        self._thumb_bottom = 0
        self._drag_offset = None
        self._hovered = False

        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Up>", lambda *_: self._scroll(-1, "units"))
        self.bind("<Down>", lambda *_: self._scroll(1, "units"))
        self.bind("<Prior>", lambda *_: self._scroll(-1, "pages"))
        self.bind("<Next>", lambda *_: self._scroll(1, "pages"))
        self.bind("<Home>", lambda *_: self.command("moveto", 0))
        self.bind("<End>", lambda *_: self.command("moveto", 1))

    def set(self, first, last):
        self.first = float(first)
        self.last = float(last)
        self._draw()

    def _draw(self, *_event):
        del _event
        height = self.winfo_height()
        self.delete("all")
        self._thumb_top = int(self.first * height)
        self._thumb_bottom = max(
            self._thumb_top + 24, int(self.last * height)
        )
        self._thumb_bottom = min(height, self._thumb_bottom)
        color = "#66666d" if self._hovered else "#4a4a50"
        self.create_rectangle(
            3, self._thumb_top, 9, self._thumb_bottom,
            fill=color, outline=color,
        )

    def _scroll(self, amount, unit):
        self.command("scroll", amount, unit)
        return "break"

    def _on_press(self, event):
        self.focus_set()
        if self._thumb_top <= event.y <= self._thumb_bottom:
            self._drag_offset = event.y - self._thumb_top
        else:
            direction = -1 if event.y < self._thumb_top else 1
            self._scroll(direction, "pages")

    def _on_drag(self, event):
        if self._drag_offset is None:
            return
        height = max(1, self.winfo_height())
        top = max(0, min(height, event.y - self._drag_offset))
        self.command("moveto", top / height)

    def _on_release(self, *_event):
        del _event
        self._drag_offset = None

    def _on_enter(self, *_event):
        del _event
        self._hovered = True
        self._draw()

    def _on_leave(self, *_event):
        del _event
        self._hovered = False
        self._draw()

    def _on_mousewheel(self, event):
        amount = -1 if event.delta > 0 else 1
        return self._scroll(amount, "units")


class ResultWindow(tk.Toplevel):
    def __init__(self, root, image):
        super().__init__(root)
        self.title("Gemini Answer")
        self.geometry("780x660")
        self.configure(bg=RESULT_BG)
        self.image = image
        self.raw = ""
        self.results = queue.Queue()

        thumb = image.copy()
        thumb.thumbnail((740, 140))
        self._thumb = ImageTk.PhotoImage(thumb, master=self)
        tk.Label(self, image=self._thumb, bg=RESULT_SURFACE, pady=6).pack(fill="x")

        body = tk.Frame(self, bg=RESULT_BG)
        body.pack(fill="both", expand=True)
        self.text = MarkdownText(body)
        scroll = DarkScrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        bar = tk.Frame(self, bg=RESULT_BG, padx=6, pady=6)
        bar.pack(fill="x")
        button_options = {
            "bg": RESULT_CONTROL,
            "fg": RESULT_FG,
            "activebackground": "#4b4b50",
            "activeforeground": "#ffffff",
            "relief": "flat",
            "borderwidth": 0,
            "highlightthickness": 0,
            "padx": 12,
            "pady": 6,
        }
        tk.Button(bar, text="Copy answer", command=self.copy, **button_options).pack(
            side="left", padx=4, pady=2
        )
        tk.Button(bar, text="Close", command=self.destroy, **button_options).pack(
            side="right", padx=4, pady=2
        )
        self.bind("<Escape>", lambda e: self.destroy())

        self.set_text("⏳ Asking Gemini…")
        self.lift()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))
        self.focus_force()

    def start(self):
        """Stream the API reply in a thread so the UI never freezes."""

        def worker():
            text = ""
            try:
                for piece in stream_gemini(self.image):
                    text += piece
                    self.results.put(("chunk", text))
                self.results.put(
                    ("done", text or "(The model returned an empty response.)")
                )
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    msg += "\n\nRate/quota limit reached. Wait a bit and try again."
                self.results.put(
                    ("done", (text + "\n\n" if text else "") + f"**Error**\n\n{msg}")
                )

        threading.Thread(target=worker, daemon=True).start()
        self._poll()

    def _poll(self):
        if not self.winfo_exists():
            return
        latest, done = None, False
        try:
            while True:
                kind, text = self.results.get_nowait()
                latest, done = text, done or kind == "done"
        except queue.Empty:
            pass
        if latest is not None:
            self.set_text(latest)
        if not done:
            self.after(200, self._poll)

    def set_text(self, md):
        self.raw = md
        first = self.text.yview()[0]
        self.text.render(md)
        if first > 0:
            self.text.yview_moveto(first)  # keep the user's scroll position

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.raw)


# --------------------------------------------------------------------------- #
# Main: hidden Tk root + global hotkey listener
# --------------------------------------------------------------------------- #
def main():
    if sys.platform == "win32":  # sharp, correctly-scaled screenshots
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

    cancel_flag = threading.Event()

    def do_capture():
        cancel_flag.clear()
        try:
            image = select_region(root, cancel_flag)
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
                    root.after(150, do_capture)  # let the hotkey keys release
                elif cmd == "quit":
                    root.destroy()
                    return
        except queue.Empty:
            pass
        root.after(100, pump)

    # pynput runs callbacks in its own thread -> hand work to Tk via a queue
    listener = keyboard.GlobalHotKeys(
        {
            HOTKEY: lambda: commands.put("capture"),
            QUIT_HOTKEY: lambda: commands.put("quit"),
        }
    )
    listener.start()

    def on_key_press(key):
        if key == keyboard.Key.esc and busy["capturing"]:
            cancel_flag.set()

    esc_listener = keyboard.Listener(on_press=on_key_press)
    esc_listener.start()
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
        esc_listener.stop()


if __name__ == "__main__":
    main()
