import sys
sys.path.insert(0, '/home/ubuntu/pacute/src')

import pandas as pd
import random
from pacute.composition import create_composition_dataset
from pacute.manipulation import create_manipulation_dataset

random.seed(100)
num_samples = 20
syllables = pd.read_json("data/syllables.jsonl", lines=True)

print("Generating MCQ composition dataset...")
mcq_composition = create_composition_dataset(syllables, num_samples=num_samples, mode='mcq', random_seed=100)
mcq_composition.to_json("test_output/mcq_composition_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_composition)} samples")
print(f"Subcategories: {mcq_composition['subcategory'].value_counts().to_dict()}")

print("\nGenerating GEN composition dataset...")
gen_composition = create_composition_dataset(syllables, num_samples=num_samples, mode='gen', random_seed=100)
gen_composition.to_json("test_output/gen_composition_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_composition)} samples")
print(f"Subcategories: {gen_composition['subcategory'].value_counts().to_dict()}")

print("\nGenerating MCQ manipulation dataset...")
mcq_manipulation = create_manipulation_dataset(syllables, num_samples=num_samples, mode='mcq', random_seed=100)
mcq_manipulation.to_json("test_output/mcq_manipulation_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_manipulation)} samples")
print(f"Subcategories: {mcq_manipulation['subcategory'].value_counts().to_dict()}")

print("\nGenerating GEN manipulation dataset...")
gen_manipulation = create_manipulation_dataset(syllables, num_samples=num_samples, mode='gen', random_seed=100)
gen_manipulation.to_json("test_output/gen_manipulation_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_manipulation)} samples")
print(f"Subcategories: {gen_manipulation['subcategory'].value_counts().to_dict()}")

print("\nAll composition and manipulation datasets generated successfully!")
