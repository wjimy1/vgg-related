import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ================= Config =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_ORIGINAL = os.path.join(SCRIPT_DIR, "checkpoint", "cifar10_vgg.pth")
MODEL_PATH_UNLEARNED = os.path.join(SCRIPT_DIR, "checkpoint", "unlearned_model.pth")
BATCH_SIZE = 256
NUM_CLASSES = 10


class VGG_Small(nn.Module):
    def __init__(self):
        super(VGG_Small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_test_loader():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(SCRIPT_DIR, "data"), train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_predictions(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def build_unlearned_model():
    model = VGG_Small().to(DEVICE)

    if os.path.exists(MODEL_PATH_UNLEARNED):
        model.load_state_dict(torch.load(MODEL_PATH_UNLEARNED, map_location=DEVICE))
        print(f"[Info] Loaded real unlearned model: {MODEL_PATH_UNLEARNED}")
        return model, "real"

    model.load_state_dict(torch.load(MODEL_PATH_ORIGINAL, map_location=DEVICE))
    # Keep behavior consistent with visualize_tsne_cnn.py when unlearned weights are unavailable.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "features" in name:
                param.add_(torch.randn_like(param) * 0.1)
    print("[Warn] unlearned_model.pth not found, using noise-perturbed model to simulate unlearning.")
    return model, "simulated"


def normalize_by_row(cm):
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return cm / row_sum


def main():
    if not os.path.exists(MODEL_PATH_ORIGINAL):
        raise FileNotFoundError(f"Original model not found: {MODEL_PATH_ORIGINAL}")

    print(f"[Info] Device: {DEVICE}")
    print("[Info] Preparing CIFAR-10 test loader...")
    test_loader = get_test_loader()

    print(f"[Info] Loading original model: {MODEL_PATH_ORIGINAL}")
    model_before = VGG_Small().to(DEVICE)
    model_before.load_state_dict(torch.load(MODEL_PATH_ORIGINAL, map_location=DEVICE))

    print("[Info] Building unlearned model...")
    model_after, after_type = build_unlearned_model()

    print("[Info] Evaluating original model...")
    y_true_before, y_pred_before = evaluate_predictions(model_before, test_loader)

    print("[Info] Evaluating unlearned model...")
    y_true_after, y_pred_after = evaluate_predictions(model_after, test_loader)

    cm_before = confusion_matrix(y_true_before, y_pred_before, labels=list(range(NUM_CLASSES)))
    cm_after = confusion_matrix(y_true_after, y_pred_after, labels=list(range(NUM_CLASSES)))

    cm_before_norm = normalize_by_row(cm_before)
    cm_after_norm = normalize_by_row(cm_after)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

    sns.heatmap(
        cm_before_norm,
        ax=axes[0],
        cmap="Blues",
        cbar=False,
        vmin=0,
        vmax=1,
        square=True,
    )
    axes[0].set_title("Before Unlearning (Original)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(
        cm_after_norm,
        ax=axes[1],
        cmap="Reds",
        cbar=True,
        vmin=0,
        vmax=1,
        square=True,
    )
    axes[1].set_title(
        "After Unlearning (Real)" if after_type == "real" else "After Unlearning (Simulated)"
    )
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    output_png = os.path.join(SCRIPT_DIR, "confusion_matrix_before_after.png")
    output_svg = os.path.join(SCRIPT_DIR, "confusion_matrix_before_after.svg")
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_svg)
    print(f"[Done] Saved: {output_png}")
    print(f"[Done] Saved: {output_svg}")


if __name__ == "__main__":
    main()
