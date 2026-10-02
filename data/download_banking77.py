from datasets import load_dataset

print("Downloading BANKING77...")

dataset = load_dataset("PolyAI/banking77")

print("\nDataset downloaded successfully!")

print("\nSplits:")
print(dataset)

print("\nFirst training example:")
print(dataset["train"][0])