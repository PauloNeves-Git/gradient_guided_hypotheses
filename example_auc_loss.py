"""
Example: Using AUC Loss for Binary Classification

This example demonstrates how to use the AUC-based loss function instead of BCE loss
for binary classification problems like the blood donation dataset.
"""

from GGH.data_ops import DataOperator
from GGH.inspector import Inspector
from execute_benchmark import full_experiment

# Configuration for blood donation dataset
data_path = "data/donated_blood/donated_blood.csv"
results_path = "saved_results/Donated Blood"

# Define variables
inpt_vars = ["Months since Last Donation", "Number of Donations", "Total Volume Donated (c.c.)"]
target_vars = ["Made Donation in March 2007"]
miss_vars = ["Months since First Donation"]
hypothesis = [[6, 12, 24, 48, 72, 96]]  # Example hypothesis values
partial_perc = 0.2

# Training parameters
use_info = "use hypothesis"
batch_size = 32
hidden_size = 64
output_size = 1
num_epochs = 100
rand_state = 42

# Initialize
DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars, hypothesis, 
                  partial_perc, rand_state, device="cpu")
INSPECT = Inspector(results_path)

# Train with AUC loss (automatically used for binary classification)
print("Training with AUC loss optimization (automatic for binary-class problems)...")
DO, TVM, model = full_experiment(
    use_info, DO, INSPECT, batch_size, hidden_size, output_size, 
    num_epochs, rand_state, results_path, 
    dropout=0.05, 
    lr=0.004, 
    nu=0.1, 
    final_analysis=False
)

# Evaluate performance
print("\nValidation AUC:", INSPECT.calculate_val_auc(DO, TVM, model, data="validation"))
print("Test AUC:", INSPECT.calculate_val_auc(DO, TVM, model, data="test"))
print("Validation Accuracy:", INSPECT.calculate_val_acc(DO, TVM, model, data="validation"))
print("Test Accuracy:", INSPECT.calculate_val_acc(DO, TVM, model, data="test"))

print(f"\nNote: AUC loss is automatically used for problem type '{DO.problem_type}'")
