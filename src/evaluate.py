# src/evaluate.py

import os
import torch
from dataset import get_dataloaders
from model import get_model

# Path to the fine-tuned model checkpoint
CHECKPOINT_PATH = "checkpoints/resnet18_deeptrace.pth"
TRAIN_ACCURACY = 98.10  # Benchmark training accuracy from training run


def compute_roc_auc(labels, scores):
    """
    Computes Area Under the ROC Curve (ROC-AUC).
    Uses scikit-learn if available; otherwise falls back to the exact Wilcoxon-Mann-Whitney
    rank-sum formula so no extra library installation is strictly required.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores)
    except ImportError:
        # Exact rank-sum formulation of ROC-AUC:
        # Measures the probability that a random positive sample (REAL)
        # is ranked higher than a random negative sample (FAKE).
        paired = sorted(zip(scores, labels), key=lambda x: x[0])
        rank_sum_pos = 0.0
        n_pos = 0
        n_neg = 0
        for rank, (_, label) in enumerate(paired, 1):
            if label == 1:
                rank_sum_pos += rank
                n_pos += 1
            else:
                n_neg += 1
        if n_pos == 0 or n_neg == 0:
            return 0.5
        u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
        return u_stat / (n_pos * n_neg)


def evaluate(checkpoint_path=CHECKPOINT_PATH, data_dir="data/raw", batch_size=32):
    """
    Evaluates the fine-tuned ResNet18 model on the held-out CIFAKE test set.
    Computes overall accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
    """
    # 1. Device Setup: Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Check if the checkpoint exists before proceeding
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at '{checkpoint_path}'.")
        print("Please ensure you have run 'python src/train.py' to produce the trained weights.")
        return

    # 3. Load Test Data
    # get_dataloaders returns (train_loader, test_loader, class_names)
    # class_names: ['FAKE', 'REAL'] -> 0 = FAKE, 1 = REAL
    print("Loading test dataset...")
    _, test_loader, class_names = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    total_images = len(test_loader.dataset)
    print(f"Loaded {total_images} test images. Classes: {class_names}")

    # 4. Rebuild Model Architecture and Load Trained Weights
    # Set pretrained=False to avoid re-downloading ImageNet weights,
    # since we are loading our own fine-tuned weights on top.
    print("Loading model architecture and checkpoint weights...")
    model = get_model(num_classes=2, pretrained=False)

    # map_location ensures weights load cleanly on both GPU and CPU
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()  # Set model to evaluation mode (disables dropout, freezes batchnorm)

    # 5. Initialize counters for the Confusion Matrix
    # Labels: 0 = FAKE, 1 = REAL
    actual_fake_pred_fake = 0  # True FAKE
    actual_fake_pred_real = 0  # FAKE misclassified as REAL
    actual_real_pred_fake = 0  # REAL misclassified as FAKE
    actual_real_pred_real = 0  # True REAL

    # Lists to store probabilities and true labels for ROC-AUC
    all_labels = []
    all_real_probs = []

    total_batches = len(test_loader)
    print(f"\nStarting evaluation across {total_batches} batches...")

    # 6. Evaluation Loop (torch.no_grad disables gradient computation for speed and memory)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass: get raw prediction scores (logits)
            outputs = model(images)

            # Convert logits to probabilities (0.0 to 1.0) using Softmax
            probabilities = torch.softmax(outputs, dim=1)
            real_probs = probabilities[:, 1]  # Probability of being REAL (class 1)

            # Collect probabilities and actual labels on CPU for ROC-AUC
            all_real_probs.extend(real_probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # The predicted class is the index with the highest score
            _, predicted = torch.max(outputs, 1)

            # Update confusion matrix counts
            for actual, pred in zip(labels, predicted):
                if actual == 0 and pred == 0:
                    actual_fake_pred_fake += 1
                elif actual == 0 and pred == 1:
                    actual_fake_pred_real += 1
                elif actual == 1 and pred == 0:
                    actual_real_pred_fake += 1
                elif actual == 1 and pred == 1:
                    actual_real_pred_real += 1

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                print(f"Evaluated batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * batch_size if (batch_idx + 1) < total_batches else total_images}/{total_images} images)...")

    # 7. Compute Metrics
    # Support counts
    total_fake = actual_fake_pred_fake + actual_fake_pred_real
    total_real = actual_real_pred_fake + actual_real_pred_real
    total_samples = total_fake + total_real
    total_correct = actual_fake_pred_fake + actual_real_pred_real

    # Accuracy
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    # Precision: Out of all predictions for a class, how many were correct?
    pred_total_fake = actual_fake_pred_fake + actual_real_pred_fake
    pred_total_real = actual_real_pred_real + actual_fake_pred_real
    fake_precision = (actual_fake_pred_fake / pred_total_fake) * 100 if pred_total_fake > 0 else 0.0
    real_precision = (actual_real_pred_real / pred_total_real) * 100 if pred_total_real > 0 else 0.0

    # Recall: Out of all actual samples of a class, how many did the model catch?
    fake_recall = (actual_fake_pred_fake / total_fake) * 100 if total_fake > 0 else 0.0
    real_recall = (actual_real_pred_real / total_real) * 100 if total_real > 0 else 0.0

    # F1-Score: Harmonic mean of Precision and Recall
    fake_f1 = (2 * fake_precision * fake_recall / (fake_precision + fake_recall)) if (fake_precision + fake_recall) > 0 else 0.0
    real_f1 = (2 * real_precision * real_recall / (real_precision + real_recall)) if (real_precision + real_recall) > 0 else 0.0

    # Macro averages
    macro_precision = (fake_precision + real_precision) / 2
    macro_recall = (fake_recall + real_recall) / 2
    macro_f1 = (fake_f1 + real_f1) / 2

    # ROC-AUC
    roc_auc = compute_roc_auc(all_labels, all_real_probs)

    # 8. Print Formatted Results
    print("\n" + "=" * 65)
    print("                    EVALUATION RESULTS")
    print("=" * 65)

    # Confusion Matrix Display
    print("\nConfusion Matrix:")
    print("-" * 65)
    print(f"{'':<16} | {'Predicted FAKE':<18} | {'Predicted REAL':<18}")
    print("-" * 65)
    print(f"{'Actual FAKE':<16} | {actual_fake_pred_fake:<18} | {actual_fake_pred_real:<18}")
    print(f"{'Actual REAL':<16} | {actual_real_pred_fake:<18} | {actual_real_pred_real:<18}")
    print("-" * 65)

    # Per-Class Classification Report
    print("\nDetailed Classification Metrics:")
    print("-" * 65)
    print(f"{'Class':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
    print("-" * 65)
    print(f"{'FAKE (0)':<12} | {fake_precision:>10.2f}% | {fake_recall:>10.2f}% | {fake_f1:>10.2f}% | {total_fake:<8}")
    print(f"{'REAL (1)':<12} | {real_precision:>10.2f}% | {real_recall:>10.2f}% | {real_f1:>10.2f}% | {total_real:<8}")
    print("-" * 65)
    print(f"{'Macro Avg':<12} | {macro_precision:>10.2f}% | {macro_recall:>10.2f}% | {macro_f1:>10.2f}% | {total_samples:<8}")
    print(f"{'Accuracy':<12} | {'':<12} | {'':<12} | {overall_accuracy:>10.2f}% | {total_samples:<8}")
    print("-" * 65)

    # ROC-AUC Metric
    print("\nROC-AUC Score:")
    print("-" * 65)
    print(f"ROC-AUC Score         : {roc_auc:.4f} (1.0000 = perfect discrimination, 0.5000 = random)")

    # 9. Overfitting Check (Generalization Gap)
    gap = TRAIN_ACCURACY - overall_accuracy
    print("\nOverfitting Analysis:")
    print("-" * 65)
    print(f"Training Accuracy     : {TRAIN_ACCURACY:.2f}%")
    print(f"Test Accuracy         : {overall_accuracy:.2f}%")
    print(f"Generalization Gap    : {gap:+.2f}%")

    if gap <= 2.0:
        print("Diagnosis: Excellent generalization! Minimal to no overfitting detected.")
    elif gap <= 5.0:
        print("Diagnosis: Good generalization with a mild drop on unseen test data.")
    else:
        print("Diagnosis: Warning - Significant gap between train and test accuracy (overfitting likely).")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()