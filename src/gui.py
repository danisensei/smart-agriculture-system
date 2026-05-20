import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    FEATURE_NAMES, FEATURE_LABELS, FEATURE_UNITS, FEATURE_RANGES,
    FEATURE_DEFAULTS, SOIL_FEATURES, MODELS_DIR, RESULTS_DIR,
    validate_all_inputs, get_cluster_info, calculate_confidence_bounds,
)

COLORS = {
    "bg_dark":       "#0f0f1a",
    "bg_panel":      "#1a1a2e",
    "bg_card":       "#16213e",
    "bg_input":      "#0a0a1a",
    "accent_blue":   "#00d2ff",
    "accent_green":  "#00e676",
    "accent_orange": "#ff9100",
    "accent_pink":   "#ff4081",
    "text_primary":  "#e0e0e0",
    "text_secondary":"#9e9e9e",
    "text_heading":  "#ffffff",
    "border":        "#2a2a4a",
    "success":       "#00e676",
    "warning":       "#ffab00",
    "error":         "#ff5252",
}

FONTS = {
    "title":      ("Segoe UI", 18, "bold"),
    "subtitle":   ("Segoe UI", 13, "bold"),
    "heading":    ("Segoe UI", 11, "bold"),
    "body":       ("Segoe UI", 10),
    "small":      ("Segoe UI", 9),
    "mono":       ("Consolas", 10),
    "result_big": ("Segoe UI", 14, "bold"),
}


class SmartAgricultureApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Agriculture Decision Support System")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.minsize(1280, 800)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1400x900")

        self.models_loaded = False
        self.dt_model = None
        self.km_model = None
        self.lr_model = None
        self.label_encoder = None
        self.cluster_scaler = None
        self.regression_scaler = None
        self.pca_model = None
        self.residual_info = None
        self.input_vars = {}
        self.plot_canvases = {}

        self._configure_styles()
        self._build_header()
        self._build_main_layout()
        self._load_models()

    def _configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure("Dark.TNotebook", background=COLORS["bg_panel"], borderwidth=0)
        self.style.configure("Dark.TNotebook.Tab",
                             background=COLORS["bg_card"], foreground=COLORS["text_primary"],
                             font=FONTS["body"], padding=(12, 6))
        self.style.map("Dark.TNotebook.Tab",
                       background=[("selected", COLORS["accent_blue"])],
                       foreground=[("selected", COLORS["bg_dark"])])

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLORS["bg_dark"], height=80)
        header.pack(fill="x", padx=20, pady=(15, 5))
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg=COLORS["bg_dark"])
        title_frame.pack(side="left", fill="y")
        tk.Label(title_frame, text="Smart Agriculture Decision Support System",
                 font=FONTS["title"], bg=COLORS["bg_dark"], fg=COLORS["text_heading"]).pack(anchor="w")
        tk.Label(title_frame, text="Decision Tree  |  KMeans Clustering  |  Linear Regression",
                 font=FONTS["small"], bg=COLORS["bg_dark"], fg=COLORS["text_secondary"]).pack(anchor="w", pady=(2, 0))

        self.status_label = tk.Label(header, text="Loading models...",
                                     font=FONTS["body"], bg=COLORS["bg_dark"], fg=COLORS["warning"])
        self.status_label.pack(side="right", padx=10)

        tk.Frame(self.root, bg=COLORS["accent_blue"], height=2).pack(fill="x", padx=20)

    def _build_main_layout(self):
        main = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main.pack(fill="both", expand=True, padx=20, pady=10)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        self._build_input_panel(main)
        self._build_results_panel(main)

    def _build_input_panel(self, parent):
        left = tk.Frame(parent, bg=COLORS["bg_panel"], width=340,
                        highlightbackground=COLORS["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        left.grid_propagate(False)

        header = tk.Frame(left, bg=COLORS["bg_panel"])
        header.pack(fill="x", padx=15, pady=(15, 10))
        tk.Label(header, text="Input Parameters", font=FONTS["subtitle"],
                 bg=COLORS["bg_panel"], fg=COLORS["accent_blue"]).pack(anchor="w")
        tk.Label(header, text="Enter soil and climate measurements",
                 font=FONTS["small"], bg=COLORS["bg_panel"], fg=COLORS["text_secondary"]).pack(anchor="w")

        tk.Frame(left, bg=COLORS["border"], height=1).pack(fill="x", padx=15, pady=5)

        inputs_frame = tk.Frame(left, bg=COLORS["bg_panel"])
        inputs_frame.pack(fill="x", padx=15, pady=5)
        for feat in FEATURE_NAMES:
            self._create_input_field(inputs_frame, feat)

        btn_frame = tk.Frame(left, bg=COLORS["bg_panel"])
        btn_frame.pack(fill="x", padx=15, pady=(20, 10))

        tk.Button(btn_frame, text="Analyze & Predict",
                  font=("Segoe UI", 12, "bold"), bg=COLORS["accent_blue"], fg=COLORS["bg_dark"],
                  activebackground="#00b8d4", relief="flat", cursor="hand2",
                  padx=20, pady=10, command=self._on_analyze).pack(fill="x", pady=(0, 8))

        tk.Button(btn_frame, text="Reset",
                  font=FONTS["body"], bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                  activebackground=COLORS["border"], relief="flat", cursor="hand2",
                  padx=15, pady=6, command=self._on_reset).pack(fill="x", pady=(0, 8))

        tk.Button(btn_frame, text="Load Sample Values",
                  font=FONTS["body"], bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                  activebackground=COLORS["border"], relief="flat", cursor="hand2",
                  padx=15, pady=6, command=self._load_defaults).pack(fill="x")

        info_frame = tk.Frame(left, bg=COLORS["bg_card"],
                              highlightbackground=COLORS["border"], highlightthickness=1)
        info_frame.pack(fill="x", padx=15, pady=(15, 15), side="bottom")
        tk.Label(info_frame, text="How It Works", font=FONTS["heading"],
                 bg=COLORS["bg_card"], fg=COLORS["accent_blue"]).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(info_frame,
                 text=("Enter soil and climate parameters,\n"
                       "then click Analyze. The system runs\n"
                       "3 models to provide:\n"
                       "- Crop recommendation (Decision Tree)\n"
                       "- Soil zone classification (KMeans)\n"
                       "- Yield prediction (Linear Regression)"),
                 font=FONTS["small"], bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                 justify="left").pack(anchor="w", padx=10, pady=(0, 8))

    def _create_input_field(self, parent, feature_name):
        frame = tk.Frame(parent, bg=COLORS["bg_panel"])
        frame.pack(fill="x", pady=3)

        lo, hi = FEATURE_RANGES[feature_name]
        unit = FEATURE_UNITS[feature_name]
        range_text = f"Range: {lo}-{hi}" + (f" {unit}" if unit else "")

        label_row = tk.Frame(frame, bg=COLORS["bg_panel"])
        label_row.pack(fill="x")
        tk.Label(label_row, text=FEATURE_LABELS[feature_name],
                 font=FONTS["body"], bg=COLORS["bg_panel"], fg=COLORS["text_primary"]).pack(side="left")
        tk.Label(label_row, text=range_text,
                 font=("Segoe UI", 8), bg=COLORS["bg_panel"], fg=COLORS["text_secondary"]).pack(side="right")

        var = tk.StringVar()
        tk.Entry(frame, textvariable=var, font=FONTS["mono"],
                 bg=COLORS["bg_input"], fg=COLORS["text_heading"],
                 insertbackground=COLORS["accent_blue"], relief="flat",
                 highlightbackground=COLORS["border"], highlightthickness=1,
                 highlightcolor=COLORS["accent_blue"]).pack(fill="x", ipady=4, pady=(2, 0))
        self.input_vars[feature_name] = var

    def _build_results_panel(self, parent):
        right = tk.Frame(parent, bg=COLORS["bg_dark"])
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_result_cards(right)
        self._build_visualization_tabs(right)

    def _build_result_cards(self, parent):
        cards_frame = tk.Frame(parent, bg=COLORS["bg_dark"])
        cards_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for i in range(3):
            cards_frame.columnconfigure(i, weight=1)

        self.crop_card    = self._create_result_card(cards_frame, 0, "Crop Recommendation",
                                                     "Decision Tree Classifier", COLORS["accent_green"])
        self.cluster_card = self._create_result_card(cards_frame, 1, "Soil Classification",
                                                     "KMeans Clustering", COLORS["accent_orange"])
        self.yield_card   = self._create_result_card(cards_frame, 2, "Yield Prediction",
                                                     "Linear Regression", COLORS["accent_pink"])

    def _create_result_card(self, parent, col, title, subtitle, accent):
        card = tk.Frame(parent, bg=COLORS["bg_card"],
                        highlightbackground=COLORS["border"], highlightthickness=1)
        card.grid(row=0, column=col, sticky="nsew", padx=5, ipady=5)

        header = tk.Frame(card, bg=COLORS["bg_card"])
        header.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(header, text=title, font=FONTS["heading"], bg=COLORS["bg_card"], fg=accent).pack(anchor="w")
        tk.Label(header, text=subtitle, font=("Segoe UI", 8),
                 bg=COLORS["bg_card"], fg=COLORS["text_secondary"]).pack(anchor="w")

        tk.Frame(card, bg=accent, height=2).pack(fill="x", padx=12, pady=6)

        result_label = tk.Label(card, text="-", font=FONTS["result_big"],
                                bg=COLORS["bg_card"], fg=COLORS["text_heading"], wraplength=280)
        result_label.pack(padx=12, anchor="w")

        detail_label = tk.Label(card, text="Awaiting analysis...", font=FONTS["small"],
                                bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                                wraplength=280, justify="left")
        detail_label.pack(padx=12, pady=(2, 10), anchor="w")

        return {"frame": card, "result": result_label, "detail": detail_label, "accent": accent}

    def _build_visualization_tabs(self, parent):
        notebook = ttk.Notebook(parent, style="Dark.TNotebook")
        notebook.grid(row=1, column=0, sticky="nsew")

        self.fi_canvas_frame  = tk.Frame(notebook, bg=COLORS["bg_panel"])
        self.cl_canvas_frame  = tk.Frame(notebook, bg=COLORS["bg_panel"])
        self.res_canvas_frame = tk.Frame(notebook, bg=COLORS["bg_panel"])

        notebook.add(self.fi_canvas_frame,  text="  Feature Importance  ")
        notebook.add(self.cl_canvas_frame,  text="  Cluster Analysis  ")
        notebook.add(self.res_canvas_frame, text="  Residual Analysis  ")

        self._load_static_plots()

    def _load_static_plots(self):
        for filename, frame in [
            ("feature_importance.png", self.fi_canvas_frame),
            ("cluster_scatter.png",    self.cl_canvas_frame),
            ("residual_plot.png",      self.res_canvas_frame),
        ]:
            path = os.path.join(RESULTS_DIR, filename)
            if os.path.exists(path):
                self._embed_image(frame, path)
            else:
                tk.Label(frame, text=f"Plot not found: {filename}\nRun training first.",
                         font=FONTS["body"], bg=COLORS["bg_panel"],
                         fg=COLORS["text_secondary"]).pack(expand=True)

    def _embed_image(self, parent, image_path):
        for w in parent.winfo_children():
            w.destroy()

        fig = Figure(figsize=(14, 5.5), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_panel"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLORS["bg_panel"])
        ax.axis("off")

        try:
            ax.imshow(plt.imread(image_path))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading plot: {e}",
                    ha="center", va="center", fontsize=12, color="red", transform=ax.transAxes)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _load_models(self):
        try:
            d = MODELS_DIR
            self.dt_model          = joblib.load(os.path.join(d, "decision_tree.joblib"))
            self.label_encoder     = joblib.load(os.path.join(d, "label_encoder.joblib"))
            self.km_model          = joblib.load(os.path.join(d, "knn_clustering.joblib"))
            self.cluster_scaler    = joblib.load(os.path.join(d, "cluster_scaler.joblib"))
            self.pca_model         = joblib.load(os.path.join(d, "cluster_pca.joblib"))
            self.lr_model          = joblib.load(os.path.join(d, "linear_regression.joblib"))
            self.regression_scaler = joblib.load(os.path.join(d, "regression_scaler.joblib"))
            self.residual_info     = joblib.load(os.path.join(d, "residual_info.joblib"))

            self.models_loaded = True
            self.status_label.config(text="All models loaded", fg=COLORS["success"])
        except FileNotFoundError as e:
            self.status_label.config(text="Model not found - run training first", fg=COLORS["error"])
            print(f"Model loading failed: {e}")

    def _on_analyze(self):
        if not self.models_loaded:
            messagebox.showerror("Models Not Loaded",
                                 "Trained models not found.\nRun: python main.py --train")
            return

        input_dict = {feat: var.get() for feat, var in self.input_vars.items()}
        is_valid, result = validate_all_inputs(input_dict)
        if not is_valid:
            messagebox.showwarning("Invalid Input", "Please fix:\n\n" + "\n".join(result))
            return

        values = result
        feature_vector = np.array([[values[f] for f in FEATURE_NAMES]])

        try:
            # crop recommendation
            crop_encoded = self.dt_model.predict(feature_vector)[0]
            crop_name    = self.label_encoder.inverse_transform([crop_encoded])[0]
            confidence   = self.dt_model.predict_proba(feature_vector)[0][crop_encoded] * 100

            self.crop_card["result"].config(text=crop_name.upper(), fg=COLORS["accent_green"])
            self.crop_card["detail"].config(
                text=f"Confidence: {confidence:.1f}%\nDepth: {self.dt_model.get_depth()}"
            )

            # soil cluster
            soil_vals   = np.array([[values[f] for f in SOIL_FEATURES]])
            soil_scaled = self.cluster_scaler.transform(soil_vals)
            cluster_id  = self.km_model.predict(soil_scaled)[0]
            info        = get_cluster_info(cluster_id)

            self.cluster_card["result"].config(text=info["name"], fg=info["color"])
            self.cluster_card["detail"].config(text=info["guidance"][:200])

            # refresh cluster plot
            self._embed_image(self.cl_canvas_frame, os.path.join(RESULTS_DIR, "cluster_scatter.png"))

            # yield prediction
            feat_scaled = self.regression_scaler.transform(feature_vector)
            yield_pred  = self.lr_model.predict(feat_scaled)[0]

            res_std   = self.residual_info["residual_std"]
            n_samples = self.residual_info["n_test_samples"]
            lower, upper, margin = calculate_confidence_bounds(res_std, yield_pred, n_samples)

            self.yield_card["result"].config(text=f"{yield_pred:.2f} tons/hectare", fg=COLORS["accent_pink"])
            self.yield_card["detail"].config(
                text=f"95% CI: [{max(0, lower):.2f} - {upper:.2f}] tons/ha\nMargin: ±{margin:.2f}"
            )

            self.status_label.config(text="Analysis complete", fg=COLORS["success"])

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status_label.config(text=f"Error: {e}", fg=COLORS["error"])

    def _on_reset(self):
        for var in self.input_vars.values():
            var.set("")
        for card in [self.crop_card, self.cluster_card, self.yield_card]:
            card["result"].config(text="-", fg=COLORS["text_heading"])
            card["detail"].config(text="Awaiting analysis...")
        self.status_label.config(text="Reset", fg=COLORS["accent_blue"])
        self._load_static_plots()

    def _load_defaults(self):
        for feat, var in self.input_vars.items():
            var.set(str(FEATURE_DEFAULTS[feat]))
        self.status_label.config(text="Sample values loaded", fg=COLORS["accent_blue"])


def launch_gui():
    root = tk.Tk()
    SmartAgricultureApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
