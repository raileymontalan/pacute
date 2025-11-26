# PACUTE

A benchmark for evaluating language model capabilities on Filipino/Tagalog linguistic tasks.

## Overview

PACUTE generates evaluation datasets testing Filipino language understanding across morphology, composition, and manipulation tasks. Datasets include both multiple-choice and generative formats with parallel English and Tagalog prompts.

## Related Work

**CUTE** (Edman et al., 2024) demonstrated that LLMs struggle with subword-level tasks despite being able to spell their tokens. Their benchmark tests English orthographic understanding through composition (spelling, character containment), similarity (orthographic vs semantic), and manipulation (insertion, deletion, substitution, swapping) tasks.

**STOCHASTOK** (Sims et al., 2025) addressed these failures by introducing stochastic tokenization during pretraining. By randomly splitting tokens into sub-token pairs, STOCHASTOK exposes models to alternative segmentations of the same word, enabling them to learn fine-grained morphological structure. This approach dramatically improved performance on CUTE tasks and enabled grokking on multi-digit arithmetic.

**PACUTE extends this line of work in three ways:**

1. **Morphologically rich language**: Filipino exhibits extensive affixation (prefixes, infixes, suffixes, circumfixes) and complex stress patterns, providing a more challenging test of subword understanding than English.

2. **Affixation tasks**: While CUTE focuses on character-level manipulation, PACUTE adds morphological inflection tasks that require understanding productive affixation rules, testing whether models can decompose and recombine morphemes.

3. **Non-Latin orthography considerations**: Filipino uses diacritics for stress marking (mabilis: á, malumi: à, maragsa: â) and the "ng" digraph, introducing orthographic complexities absent in English.

These additions test whether improvements from methods like STOCHASTOK transfer to languages with richer morphology and different orthographic systems.

## Task Categories

### Affixation
Tests understanding of Filipino morphology through prefix, suffix, infix, and circumfix operations.

- **Forward affixation**: Given a root word and an affix, produce the inflected form
- **Reverse affixation**: Given an inflected word, identify which affix was used

Example:
```
English: Inflect the word "inom" to use the prefix "um-".
Tagalog: Lapian ng "um-" ang salitang "inom".
Answer: uminom
```

### Composition
Evaluates character-level understanding and word structure analysis.

- Spelling (character enumeration)
- Character counting and comparison
- Word length analysis
- Diacritic identification
- Case analysis (uppercase/lowercase)

Example:
```
English: How many "a"s are in "kakulangan"?
Tagalog: Ilang "a" ang mayroon sa "kakulangan".
Answer: 4
```

### String Manipulation
Measures ability to perform systematic character-level transformations.

- Insertion (adding characters after specific positions)
- Deletion (removing all instances of a character)
- Substitution (replacing one character with another)
- Permutation (swapping two characters throughout)
- Duplication (repeating characters)
- Case transformations
- Diacritic normalization

Example:
```
English: Remove every "i" in "pinagkulangan".
Tagalog: Tanggalin ang bawat "i" sa "pinagkulangan".
Answer: pnagkulangan
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generating Datasets

```python
import pandas as pd
from pacute import create_affixation_dataset, create_composition_dataset, create_manipulation_dataset

# Load syllables data
syllables_df = pd.read_json("data/syllables.jsonl", lines=True)

# Generate composition dataset (MCQ format, 20 samples per task)
composition_mcq = create_composition_dataset(
    syllables_df,
    num_samples=20,
    mode='mcq',
    random_seed=100
)

# Generate manipulation dataset (generative format)
manipulation_gen = create_manipulation_dataset(
    syllables_df,
    num_samples=20,
    mode='gen',
    random_seed=100
)

# Save to disk
composition_mcq.to_json("output/composition.jsonl", lines=True, orient="records", force_ascii=False)
```

### Using Syllabification

```python
from pacute import syllabify, normalize_text

word = "pinagkulangan"
syllables = syllabify(word)
print(syllables)  # ['pi', 'nag', 'ku', 'la', 'ngan']

word_with_diacritics = "kakúlangan"
normalized = normalize_text(word_with_diacritics)
print(normalized)  # 'kakulangan'
```

## Project Structure

```
pacute/
├── src/pacute/           # Core library code
│   ├── affixation.py     # Affixation dataset generation
│   ├── composition.py    # Composition task generation
│   ├── manipulation.py   # String manipulation tasks
│   ├── syllabification.py # Syllabification algorithm
│   └── string_operations.py # Character-level operations
├── data/                 # Generated datasets
│   ├── syllables.jsonl   # Processed dictionary (16,828 words)
│   └── *.jsonl           # Generated evaluation datasets
├── exploration/          # Research notebooks and experiments
│   ├── create_affixation.ipynb
│   ├── create_composition_string_manipulation_syllabification.ipynb
│   └── diksiyonaryo.ipynb
├── updf/                 # Source dictionary files (67,691 entries)
├── homographs_dicts/     # Homograph word lists
└── output_folder/        # Processed dictionary data by letter
```

## Dataset Format

All datasets follow a consistent JSONL structure:

### Multiple Choice Questions (MCQ)
```json
{
  "category": "composition",
  "subcategory": "spelling",
  "prompts": [{
    "text_en": "Which option spells out \"ako\"?",
    "text_tl": "Alin sa sumusunod ang nagbabaybay sa \"ako\"?",
    "choice1": "a k o",
    "choice2": "o k a",
    "choice3": "a o k",
    "choice4": "k a o"
  }],
  "label": "A"
}
```

### Generative Format
```json
{
  "category": "affixation",
  "subcategory": "prefix",
  "prompts": [{
    "text_en": "Inflect the word \"inom\" to use the prefix \"um-\".",
    "text_tl": "Lapian ng \"um-\" ang salitang \"inom\"."
  }],
  "label": "uminom"
}
```

## Data Sources

The benchmark is built on:
- **UP Diksiyonaryo**: 67,691 dictionary entries with etymological information
- **Filtered syllables dataset**: 16,828 single-word Filipino terms with stress marking
- **Inflections dataset**: Manually curated affixation examples

### Syllabification Algorithm

The syllabification follows Tagalog phonological rules:
- Respects the "ng" digraph as a single unit
- Handles consonant clusters (bl, br, dr, pl, tr)
- Correctly splits vowel-consonant boundaries
- Preserves stress markers (mabilis: á é í ó ú, malumi: à è ì ò ù, maragsa: â ê î ô û)

## Statistics

Current dataset sizes:
- Affixation (MCQ): 80 samples
- Affixation (Gen): 80 samples
- Reverse Affixation (MCQ): 60 samples
- Reverse Affixation (Gen): 60 samples
- Composition (MCQ): 160 samples
- Composition (Gen): 100 samples
- Manipulation (MCQ): 160 samples
- Manipulation (Gen): 160 samples
- Combined (MCQ): 320 samples
- Combined (Gen): 260 samples

Total: ~1,500 test items across all categories

## Development

The `exploration/` folder contains Jupyter notebooks used during initial development and experimentation. The production code has been refactored into the `src/pacute/` module for better maintainability and reusability.

## License

MIT License (see LICENSE file)
