"""Run the trained digit CNN on a generated 9x9 MNIST mosaic."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from nn.model import DigitCNN


def load_classifier(weights_path, temperature_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    temperature = 1.0
    if temperature_path and Path(temperature_path).exists():
        import json
        temperature = float(json.loads(Path(temperature_path).read_text()).get("temperature", 1.0))
    return model, temperature, device


def predict_mosaic(mosaic_path, weights_path="nn/artifacts/digit_cnn.pt",
                    temperature_path="nn/artifacts/calibration.json", blank_threshold=0.025):
    """Return (read_grid, probabilities) from a 252x252 known-layout mosaic.

    The generator uses MNIST's black-background/white-ink convention. A cell is
    blank when its mean ink intensity after border removal is below threshold.
    This is intentionally not a camera/grid-localization system.
    """
    model, temperature, device = load_classifier(weights_path, temperature_path)
    image = np.asarray(Image.open(mosaic_path).convert("L"), dtype=np.float32) / 255.0
    if image.shape != (252, 252):
        raise ValueError("expected a 252x252 generated Sudoku mosaic")
    crops, locations, grid = [], [], np.zeros((9, 9), dtype=int)
    for r in range(9):
        for c in range(9):
            crop = image[r * 28 + 2:(r + 1) * 28 - 2, c * 28 + 2:(c + 1) * 28 - 2]
            if float(crop.mean()) < blank_threshold:
                continue
            crop = np.asarray(Image.fromarray((crop * 255).astype("uint8")).resize((28, 28))) / 255.0
            crops.append(crop.astype("float32")); locations.append((r, c))
    probabilities = np.full((9, 9, 9), 1.0 / 9, dtype=np.float32)
    if crops:
        tensor = torch.tensor(np.asarray(crops)[:, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(tensor) / max(temperature, 1e-3)
            predicted = torch.softmax(logits, dim=1).cpu().numpy()
        for (r, c), p in zip(locations, predicted):
            probabilities[r, c] = p
            grid[r, c] = int(np.argmax(p)) + 1
    return grid, probabilities
