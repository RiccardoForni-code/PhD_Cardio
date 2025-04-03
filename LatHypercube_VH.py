import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pyDOE2 import lhs
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load Excel file and extract variables
def load_excel(file_path, sheet_name=None):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# Formula for the Index
def calculate_index(weights, data):
    return weights[0] * data['Avg_Muscle [HU]'] + weights[1] * data['Amplitude'] + weights[2] * data['Kurtosis']

# Logistic loss as the objective function
def objective_function(weights, data, labels):
    predicted_index = calculate_index(weights, data)
    # Sigmoid function to map the Index to probabilities
    probabilities = 1 / (1 + np.exp(-predicted_index))
    # Logistic loss
    loss = -np.sum(labels * np.log(probabilities + 1e-10) + (1 - labels) * np.log(1 - probabilities + 1e-10))
    return loss

# Latin Hypercube Sampling to generate initial weights
def generate_initial_weights(n_samples, bounds):
    samples = lhs(len(bounds), samples=n_samples)
    weights = np.array([bounds[i][0] + samples[:, i] * (bounds[i][1] - bounds[i][0]) for i in range(len(bounds))]).T
    return weights

# Optimization function
def optimize_weights(data, labels, initial_weights, bounds):
    best_result = None
    for initial in initial_weights:
        result = minimize(
            objective_function, 
            initial, 
            args=(data, labels), 
            bounds=bounds,
            method='L-BFGS-B'
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    return best_result

# Normalization function (if required)
def normalization(data):
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm_data

# Main script
if __name__ == "__main__":
    # Parameters
    file_path = "df_1.xlsx"
    bounds = [(0, 1), (0, 1), (0, 1)]  # Bounds for a, b, c
    n_samples = 500  # Number of LHS samples

    # Load data
    df = pd.read_excel(file_path)
    print(df.columns)
    
    # Filter out rows where Pathology_Sub == 2
    df = df[df["Pathology_Sub"] < 2]
    
    # Extract relevant columns
    data = df[['Avg_Muscle [HU]', 'Amplitude', 'Kurtosis']]
    labels = df["Pathology"]  # Replace with the actual label column name


    # Normalize the 'Avg_Muscle [HU]' column using .loc
    data.loc[:, 'Avg_Muscle [HU]'] = normalization(data['Avg_Muscle [HU]'])

    # Similarly for other columns
    data.loc[:, 'Amplitude'] = normalization(data['Amplitude'])
    data.loc[:, 'Kurtosis'] = normalization(data['Kurtosis'])
 
    # Generate initial weights using LHS
    initial_weights = generate_initial_weights(n_samples, bounds)
    print(initial_weights)

    # Optimize weights
    result = optimize_weights(data, labels, initial_weights, bounds)

    # Display the best weights and error
    print("Optimized Weights (a, b, c):", result.x)
    print("Objective Function Value (Log Loss):", result.fun)

    # Calculate predicted index using the optimized weights
    predicted_index = calculate_index(result.x, data)

    # Compute predicted probabilities (Sigmoid of the predicted index)
    probabilities = 1 / (1 + np.exp(-predicted_index))

    # Plot AUC-ROC
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Optionally, print predicted index for each data point
    print("Predicted Index for each data point:")
    print(predicted_index)
