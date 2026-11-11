"""Train CNN, temperature-calibrate on validation, and compare Logistic Regression."""
import argparse
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from nn.model import DigitCNN
from nn.metrics import classification_report


def fit_temperature(logits, labels):
    """Fit one positive scalar temperature using validation NLL only."""
    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=80)
    criterion = torch.nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp(0.05, 10.0), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(temperature.detach().clamp(0.05, 10.0).cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mnist_1_9.npz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="nn/artifacts")
    parser.add_argument("--lr-subset", type=int, default=20000)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = np.load(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def make_loader(x, y, shuffle):
        dataset = TensorDataset(torch.tensor(x[:, None], dtype=torch.float32),
                                torch.tensor(y - 1, dtype=torch.long))
        generator = torch.Generator().manual_seed(args.seed) if shuffle else None
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          generator=generator, num_workers=0)

    train_loader = make_loader(data["x_train"], data["y_train"], True)
    val_loader = make_loader(data["x_val"], data["y_val"], False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = loss_fn(model(x), y)
            loss.backward(); optimizer.step(); running += loss.item() * len(y)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item() * len(y)
        val_loss /= len(val_loader.dataset)
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={running/len(train_loader.dataset):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    def logits_for(x):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x), 512):
                tensor = torch.tensor(x[start:start + 512, None], dtype=torch.float32, device=device)
                chunks.append(model(tensor))
        return torch.cat(chunks, dim=0)

    val_logits = logits_for(data["x_val"])
    val_labels = torch.tensor(data["y_val"] - 1, dtype=torch.long, device=device)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits = logits_for(data["x_test"])
    raw_test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
    calibrated_test_probs = torch.softmax(test_logits / temperature, dim=1).cpu().numpy()
    val_probs = torch.softmax(val_logits / temperature, dim=1).cpu().numpy()
    metrics = {
        "validation_calibrated": classification_report(val_probs, data["y_val"]),
        "test_uncalibrated": classification_report(raw_test_probs, data["y_test"]),
        "test_calibrated": classification_report(calibrated_test_probs, data["y_test"]),
        "temperature": temperature,
        "best_validation_nll_uncalibrated": best_val,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
    }
    torch.save(model.cpu().state_dict(), out / "digit_cnn.pt")
    (out / "calibration.json").write_text(json.dumps({"temperature": temperature}, indent=2))
    (out / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Logistic Regression baseline on a deterministic, documented training subset.
    rng = np.random.default_rng(args.seed)
    subset = rng.choice(len(data["x_train"]), min(args.lr_subset, len(data["x_train"])), replace=False)
    baseline = LogisticRegression(max_iter=300, solver="lbfgs")
    baseline.fit(data["x_train"][subset].reshape(len(subset), -1), data["y_train"][subset])
    baseline_probs = baseline.predict_proba(data["x_test"].reshape(len(data["x_test"]), -1))
    # sklearn orders classes as 1..9 for this dataset; enforce explicit order.
    baseline_probs = baseline_probs[:, np.argsort(baseline.classes_)]
    metrics["logistic_regression_test"] = classification_report(baseline_probs, data["y_test"])
    metrics["logistic_regression_train_subset_size"] = int(len(subset))
    (out / "all_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"cnn_test_calibrated": metrics["test_calibrated"],
                      "logistic_regression_test": metrics["logistic_regression_test"],
                      "temperature": temperature}, indent=2))


if __name__ == "__main__":
    main()
