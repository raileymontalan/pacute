import sys
sys.path.insert(0, '/home/ubuntu/pacute/src')

import pandas as pd
from pacute.affixation import create_affixation_dataset

inflections = pd.read_excel("exploration/inflections.xlsx", sheet_name="data")

print("Generating MCQ affixation dataset...")
mcq_dataset = create_affixation_dataset(inflections, mode='mcq', reverse=False, random_seed=42)
mcq_dataset.to_json("test_output/mcq_affixation_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_dataset)} samples")

print("\nGenerating MCQ reverse affixation dataset...")
mcq_reverse_dataset = create_affixation_dataset(inflections, mode='mcq', reverse=True, random_seed=42)
mcq_reverse_dataset.to_json("test_output/mcq_reverse_affixation_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(mcq_reverse_dataset)} samples")

print("\nGenerating GEN affixation dataset...")
gen_dataset = create_affixation_dataset(inflections, mode='gen', reverse=False, random_seed=42)
gen_dataset.to_json("test_output/gen_affixation_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_dataset)} samples")

print("\nGenerating GEN reverse affixation dataset...")
gen_reverse_dataset = create_affixation_dataset(inflections, mode='gen', reverse=True, random_seed=42)
gen_reverse_dataset.to_json("test_output/gen_reverse_affixation_dataset.jsonl", lines=True, orient="records", force_ascii=False)
print(f"Generated {len(gen_reverse_dataset)} samples")

print("\nAll affixation datasets generated successfully!")
