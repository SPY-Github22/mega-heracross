import torch
import numpy as np
import matplotlib.pyplot as plt
from mc_dropout import mc_dropout_infer

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    y_true: 1D numpy array of binary ground truth (0 or 1)
    y_prob: 1D numpy array of predicted probabilities (0 to 1)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        else:
            accuracies.append(0.0)
            confidences.append(0.0)
            
    return ece, accuracies, confidences

def plot_reliability_diagram(accuracies, confidences, ece, save_path="reliability_diagram.png"):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Filter out bins with 0 items
    valid_conf = [c for c in confidences if c > 0]
    valid_acc = [a for a, c in zip(accuracies, confidences) if c > 0]
    
    plt.plot(valid_conf, valid_acc, marker='o', label='Model Calibration')
    plt.title(f'Reliability Diagram\nExpected Calibration Error (ECE): {ece:.4f}')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved reliability diagram to {save_path}")

def test_calibration_logic():
    print("Testing calibration logic with mock data...")
    # Generate mock probabilities slightly overconfident
    rng = np.random.RandomState(42)
    # mock probabilities biased towards extremes
    y_prob = rng.beta(0.5, 0.5, 10000)
    
    # mock labels where the model is roughly correct but overconfident
    # if y_prob is 0.9, real acc is 0.75
    y_true = rng.rand(10000) < (y_prob * 0.8 + 0.1)
    
    ece, accs, confs = expected_calibration_error(y_true, y_prob, n_bins=10)
    print(f"Mock ECE: {ece:.4f}")
    plot_reliability_diagram(accs, confs, ece, "test_reliability.png")

if __name__ == "__main__":
    test_calibration_logic()
