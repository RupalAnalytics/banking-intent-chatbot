from datasets import load_dataset
import os

# Load BANKING77
dataset = load_dataset("PolyAI/banking77")

# Create a folder for the dataset
os.makedirs("data", exist_ok=True)

# Save the training and testing data
dataset["train"].to_csv("data/train.csv", index=False)
dataset["test"].to_csv("data/test.csv", index=False)

print("Dataset was downloaded and saved successfully")