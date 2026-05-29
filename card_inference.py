from __future__ import annotations

import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageOps


DEFAULT_MODEL_CANDIDATES = [
    Path("models/card_noncard_model.keras"),
    Path("models/card_noncard_classifier_model.keras"),
]
INPUT_SIZE = (224, 224)


def _resolve_model_path(base_dir: Path, override: str | None) -> Path:
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model file not found: {candidate}")

    for relative_path in DEFAULT_MODEL_CANDIDATES:
        candidate = base_dir / relative_path
        if candidate.exists():
            return candidate

    searched = ", ".join(str(base_dir / path) for path in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(f"Could not find a model file. Looked for: {searched}")


def _read_dataset(weights: h5py.File, path: str) -> np.ndarray:
    return np.array(weights[path], dtype=np.float32)


@dataclass(slots=True)
class CardNonCardModel:
    conv1_kernel: np.ndarray
    conv1_bias: np.ndarray
    conv2_kernel: np.ndarray
    conv2_bias: np.ndarray
    conv3_kernel: np.ndarray
    conv3_bias: np.ndarray
    dense_kernel: np.ndarray
    dense_bias: np.ndarray
    output_kernel: np.ndarray
    output_bias: np.ndarray

    @classmethod
    def load(cls, model_path: Path) -> "CardNonCardModel":
        with zipfile.ZipFile(model_path) as archive:
            archive.read("config.json")
            weights_bytes = archive.read("model.weights.h5")

        with h5py.File(io.BytesIO(weights_bytes), "r") as weights:
            return cls(
                conv1_kernel=_read_dataset(weights, "layers/conv2d/vars/0"),
                conv1_bias=_read_dataset(weights, "layers/conv2d/vars/1"),
                conv2_kernel=_read_dataset(weights, "layers/conv2d_1/vars/0"),
                conv2_bias=_read_dataset(weights, "layers/conv2d_1/vars/1"),
                conv3_kernel=_read_dataset(weights, "layers/conv2d_2/vars/0"),
                conv3_bias=_read_dataset(weights, "layers/conv2d_2/vars/1"),
                dense_kernel=_read_dataset(weights, "layers/dense/vars/0"),
                dense_bias=_read_dataset(weights, "layers/dense/vars/1"),
                output_kernel=_read_dataset(weights, "layers/dense_1/vars/0"),
                output_bias=_read_dataset(weights, "layers/dense_1/vars/1"),
            )

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _window_view(x: np.ndarray, kernel_h: int, kernel_w: int) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(x, (kernel_h, kernel_w), axis=(0, 1))
        if windows.shape[2] == x.shape[2]:
            windows = windows.transpose(0, 1, 3, 4, 2)
        elif windows.shape[-1] == x.shape[2]:
            pass
        else:
            raise ValueError(f"Unexpected sliding window shape: {windows.shape}")
        return windows

    def _conv2d_valid(self, x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        windows = self._window_view(x, kernel.shape[0], kernel.shape[1])
        output = np.tensordot(windows, kernel, axes=([2, 3, 4], [0, 1, 2]))
        return output + bias

    def _max_pool2d(self, x: np.ndarray, pool_size: tuple[int, int] = (2, 2), strides: tuple[int, int] = (2, 2)) -> np.ndarray:
        windows = self._window_view(x, pool_size[0], pool_size[1])
        windows = windows[::strides[0], ::strides[1], ...]
        return windows.max(axis=(2, 3))

    @staticmethod
    def _flatten(x: np.ndarray) -> np.ndarray:
        return x.reshape(-1)

    @staticmethod
    def _dense(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return x @ kernel + bias

    def predict_probability(self, image: Image.Image, normalize: bool = True) -> float:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
        x = np.asarray(image, dtype=np.float32)
        if normalize:
            x /= 255.0

        x = self._relu(self._conv2d_valid(x, self.conv1_kernel, self.conv1_bias))
        x = self._max_pool2d(x)
        x = self._relu(self._conv2d_valid(x, self.conv2_kernel, self.conv2_bias))
        x = self._max_pool2d(x)
        x = self._relu(self._conv2d_valid(x, self.conv3_kernel, self.conv3_bias))
        x = self._max_pool2d(x)
        x = self._flatten(x)
        x = self._relu(self._dense(x, self.dense_kernel, self.dense_bias))
        logits = self._dense(x, self.output_kernel, self.output_bias).reshape(-1)[0]
        return float(self._sigmoid(logits))


def _classify_probability(probability: float, threshold: float) -> str:
    return "card" if probability >= threshold else "not card"


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()


def run_cli(model: CardNonCardModel, image_path: Path, threshold: float, normalize: bool) -> None:
    probability = model.predict_probability(_load_image(image_path), normalize=normalize)
    label = _classify_probability(probability, threshold)
    print(f"{image_path} -> {label} ({probability:.4f})")


def run_gui(model: CardNonCardModel, model_name: str, threshold: float, normalize: bool) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    from PIL import ImageTk

    root = tk.Tk()
    root.title("Card Detector")
    root.geometry("540x640")
    root.minsize(540, 640)

    preview_ref = None

    title = tk.Label(root, text="Upload an image to check card presence", font=("Segoe UI", 16, "bold"))
    title.pack(pady=(18, 10))

    preview_container = tk.Label(root, text="No image selected", width=56, height=22, relief="groove", bg="#f5f5f5")
    preview_container.pack(padx=18, pady=12, fill="both", expand=False)

    path_label = tk.Label(root, text="", wraplength=480, justify="center")
    path_label.pack(padx=18, pady=(0, 8))

    result_label = tk.Label(root, text="Result will appear here", font=("Segoe UI", 15, "bold"))
    result_label.pack(pady=(12, 4))

    probability_label = tk.Label(root, text="", font=("Segoe UI", 11))
    probability_label.pack(pady=(0, 12))

    def show_image() -> None:
        nonlocal preview_ref

        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        selected_path = Path(file_path)
        try:
            image = _load_image(selected_path)
            probability = model.predict_probability(image, normalize=normalize)
        except Exception as exc:
            messagebox.showerror("Inference error", str(exc))
            return

        label = _classify_probability(probability, threshold)
        result_label.config(text=label.upper(), fg="#1f7a1f" if label == "card" else "#8a1f11")
        probability_label.config(text=f"Card probability: {probability:.4f}")
        path_label.config(text=str(selected_path))

        preview_image = ImageOps.exif_transpose(image).convert("RGB")
        preview_image.thumbnail((480, 320), Image.Resampling.LANCZOS)
        preview_ref = ImageTk.PhotoImage(preview_image)
        preview_container.config(image=preview_ref, text="")
        preview_container.image = preview_ref

    button_bar = tk.Frame(root)
    button_bar.pack(pady=14)

    pick_button = tk.Button(button_bar, text="Choose Image", command=show_image, width=16, height=2)
    pick_button.pack(side="left", padx=8)

    quit_button = tk.Button(button_bar, text="Quit", command=root.destroy, width=16, height=2)
    quit_button.pack(side="left", padx=8)

    footer = tk.Label(
        root,
        text=f"Model: {model_name}\nThreshold: {threshold:.2f}",
        justify="center",
        fg="#666666",
    )
    footer.pack(side="bottom", pady=14)

    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple card vs not-card inference app.")
    parser.add_argument("--model-path", default=None, help="Path to the .keras model file.")
    parser.add_argument("--image", default=None, help="Optional image path for CLI inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for card probability.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable 0-1 pixel normalization.")
    parser.add_argument("--gui", action="store_true", help="Force the desktop file-picker UI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    model_path = _resolve_model_path(base_dir, args.model_path)
    model = CardNonCardModel.load(model_path)

    if args.image and not args.gui:
        run_cli(model, Path(args.image), args.threshold, normalize=not args.no_normalize)
        return

    run_gui(model, model_path.name, args.threshold, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()