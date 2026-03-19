from __future__ import annotations

import io
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tkinter import END, LEFT, RIGHT, BOTH, X, filedialog, messagebox, StringVar, Tk
from tkinter import ttk

try:
    from .inspection import inspect_csv
    from .script_metadata import StudyMetadata, load_study_metadata
except ImportError:
    from inspection import inspect_csv
    from script_metadata import StudyMetadata, load_study_metadata


class StudyTab:
    def __init__(self, parent, metadata: StudyMetadata):
        self.metadata = metadata
        self.frame = ttk.Frame(parent)
        self.selected_input = StringVar(value=str(metadata.default_input))
        self.dataset_mode = StringVar(value=(metadata.dataset_modes[0] if metadata.dataset_modes else ""))

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        header = ttk.Frame(self.frame)
        header.pack(fill=X)
        ttk.Label(header, text=f"Script: {self.metadata.script_path}").pack(anchor="w", **pad)
        ttk.Label(header, text=f"Default input: {self.metadata.default_input}").pack(anchor="w", **pad)
        ttk.Label(header, text=f"Output pattern: {self.metadata.default_output_pattern}").pack(anchor="w", **pad)

        if self.metadata.dataset_modes:
            mode_row = ttk.Frame(self.frame)
            mode_row.pack(fill=X)
            ttk.Label(mode_row, text="Dataset mode (Study 4):").pack(side=LEFT, **pad)
            mode_combo = ttk.Combobox(mode_row, textvariable=self.dataset_mode, values=self.metadata.dataset_modes, state="readonly", width=20)
            mode_combo.pack(side=LEFT, **pad)

        input_row = ttk.Frame(self.frame)
        input_row.pack(fill=X)
        ttk.Label(input_row, text="Selected input:").pack(side=LEFT, **pad)
        ttk.Entry(input_row, textvariable=self.selected_input, width=95).pack(side=LEFT, fill=X, expand=True, **pad)

        button_row = ttk.Frame(self.frame)
        button_row.pack(fill=X)
        ttk.Button(button_row, text="Browse", command=self.on_browse).pack(side=LEFT, **pad)
        ttk.Button(button_row, text="Use default input", command=self.use_default).pack(side=LEFT, **pad)
        ttk.Button(button_row, text="Inspect file", command=self.inspect).pack(side=LEFT, **pad)
        ttk.Button(button_row, text="Run analysis", command=self.run).pack(side=LEFT, **pad)

        metadata_text = (
            f"Expected required columns: {', '.join(self.metadata.required_columns)}\n"
            f"Optional columns: {', '.join(self.metadata.optional_columns)}\n"
            f"Configured outcomes: {', '.join(self.metadata.outcomes)}\n"
            f"Configured composites: "
            + "; ".join([f"{k}=[{', '.join(v)}]" for k, v in self.metadata.composites.items()])
            + "\n"
            + "Major model formulas:\n- "
            + "\n- ".join(self.metadata.model_descriptions)
            + "\nNotable script logic:\n- "
            + "\n- ".join(self.metadata.notable_logic)
        )

        meta_frame = ttk.LabelFrame(self.frame, text="Script-aware metadata")
        meta_frame.pack(fill=BOTH, expand=False, padx=6, pady=4)
        self.meta_box = self._text_widget(meta_frame, height=10)
        self.meta_box.insert(END, metadata_text)
        self.meta_box.configure(state="disabled")

        log_frame = ttk.LabelFrame(self.frame, text="Inspection / Run log")
        log_frame.pack(fill=BOTH, expand=True, padx=6, pady=4)
        self.log_box = self._text_widget(log_frame, height=24)

    def _text_widget(self, parent, height=14):
        text = ttk.Treeview  # placeholder to satisfy lint in minimal env
        del text
        import tkinter as tk

        wrapper = ttk.Frame(parent)
        wrapper.pack(fill=BOTH, expand=True)
        t = tk.Text(wrapper, wrap="word", height=height)
        yscroll = ttk.Scrollbar(wrapper, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=yscroll.set)
        t.pack(side=LEFT, fill=BOTH, expand=True)
        yscroll.pack(side=RIGHT, fill="y")
        return t

    def append_log(self, text: str):
        self.log_box.insert(END, text + "\n")
        self.log_box.see(END)

    def clear_log(self):
        self.log_box.delete("1.0", END)

    def on_browse(self):
        selected = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if selected:
            self.selected_input.set(selected)

    def use_default(self):
        self.selected_input.set(str(self.metadata.default_input))

    def inspect(self):
        self.clear_log()
        path = Path(self.selected_input.get()).expanduser().resolve()
        try:
            report = inspect_csv(path, self.metadata, self.dataset_mode.get() or None)
            self.append_log(report)
        except Exception:
            self.append_log(traceback.format_exc())

    def run(self):
        path = Path(self.selected_input.get()).expanduser().resolve()
        if not path.exists():
            messagebox.showerror("Missing file", f"Input file does not exist:\n{path}")
            return

        self.append_log(f"Running {self.metadata.display_name} using {path} ...")

        def _worker():
            buf = io.StringIO()
            success = True
            output_dir = None
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    output_dir = self.metadata.run_callable(path, self.dataset_mode.get() or None)
            except Exception:
                success = False
                buf.write("\n" + traceback.format_exc())

            log_text = buf.getvalue()
            self.frame.after(0, lambda: self._finish_run(success, output_dir, log_text))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_run(self, success: bool, output_dir: Path | None, log_text: str):
        self.append_log("=" * 80)
        self.append_log(f"Run status: {'SUCCESS' if success else 'FAILED'}")
        if output_dir:
            self.append_log(f"Output directory: {output_dir}")
            if output_dir.exists():
                files = sorted([p for p in output_dir.rglob("*") if p.is_file()])[:20]
                if files:
                    self.append_log("Key generated files (first 20):")
                    for p in files:
                        self.append_log(f"- {p.relative_to(output_dir)}")
        if log_text.strip():
            self.append_log("Console output:")
            self.append_log(log_text)


class AnalysisLauncherApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Gener-AI-te Analysis Launcher")
        self.root.geometry("1280x860")

        notebook = ttk.Notebook(root)
        notebook.pack(fill=BOTH, expand=True)

        for key, meta in load_study_metadata().items():
            tab = StudyTab(notebook, meta)
            notebook.add(tab.frame, text=meta.display_name)


def main():
    root = Tk()
    AnalysisLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
