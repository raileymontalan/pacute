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

### 1. Affixation (280 samples total)
Tests understanding of Filipino morphology through prefix, suffix, infix, and circumfix operations.

**Subcategories:**
- **Prefix** (e.g., mag-, nag-, um-): 40 samples per format
- **Suffix** (e.g., -an, -in, -han): 40 samples per format
- **Infix** (e.g., -um-, -in-): 40 samples per format
- **Circumfix** (e.g., pag-...-an, ka-...-an): 20 samples per format

**Task Types:**
- **Affix inflection**: Given a root word and an affix, produce the inflected form
- **Affix identification**: Given an inflected word, identify which affix was used (reverse task)

Example:
```
English: Inflect the word "inom" to use the prefix "um-".
Tagalog: Lapian ng "um-" ang salitang "inom".
Answer: uminom
```

### 2. Composition (280 samples total)
Evaluates character-level understanding and word structure analysis.

**Subcategories:**
- **Spelling**: Spell out words character-by-character (20 samples)
- **Character counting**: Count occurrences of specific characters (60 MCQ, 20 GEN)
- **Diacritic counting**: Count diacritic marks (20 samples)
- **Uppercase counting**: Count uppercase letters (20 samples)
- **Length analysis**: Word length comparisons and exact counting (60 MCQ, 20 GEN)

Example:
```
English: How many "a"s are in "kakulangan"?
Tagalog: Ilang "a" ang mayroon sa "kakulangan".
Answer: 4
```

### 3. String Manipulation (320 samples total)
Measures ability to perform systematic character-level transformations.

**Subcategories:**
- **Insertion**: Adding characters after specific positions (20 samples)
- **Deletion**: Removing all instances of a character (20 samples)
- **Substitution**: Replacing one character with another (20 samples)
- **Permutation**: Swapping two characters throughout (20 samples)
- **Duplication**: Repeating characters (20 samples)
- **Uppercasing**: Converting to uppercase (20 samples)
- **Lowercasing**: Converting to lowercase (20 samples)
- **Diacritic normalization**: Removing diacritics (20 samples)

Example:
```
English: Remove every "i" in "pinagkulangan".
Tagalog: Tanggalin ang bawat "i" sa "pinagkulangan".
Answer: pnagkulangan
```

### 4. Syllabification (160 samples total)
Tests understanding of Filipino phonological structure and stress patterns.

**Subcategories:**
- **Stress classification**: Identify stress type (mabilis/malumi/maragsa/malumay) (20 samples)
- **Reduplication detection/identification**: Identify repeated syllables (20 samples)
- **ng-aware syllable counting**: Count syllables treating 'ng' as single unit (20 samples)
- **General syllable counting**: Count total syllables (20 samples)

Example:
```
English: What is the stress classification of "kumakáin"?
Tagalog: Ano ang bigkas ng "kumakáin"?
Answer: mabilis (acute: á)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generating Datasets

```python
import sys
sys.path.insert(0, 'src')  # Add src to path

import pandas as pd
from pacute import (
    create_affixation_dataset,
    create_composition_dataset,
    create_manipulation_dataset,
    create_syllabification_dataset
)

# Load syllables data
syllables_df = pd.read_json("data/syllables.jsonl", lines=True)

# Generate composition dataset (MCQ format, 20 samples per task)
mcq_composition = create_composition_dataset(
    syllables_df,
    num_samples=20,
    mode='mcq',
    random_seed=100
)

# Generate manipulation dataset (generative format)
gen_manipulation = create_manipulation_dataset(
    syllables_df,
    num_samples=20,
    mode='gen',
    random_seed=100
)

# Generate syllabification dataset
syllabification_gen = create_syllabification_dataset(
    syllables_path='data/syllables.jsonl',
    word_freq_path='data/word_frequencies.csv',
    n_samples_per_task=20,
    mode='gen'
)

# Save to disk
mcq_composition.to_json("output/mcq_composition.jsonl", lines=True, orient="records", force_ascii=False)
gen_manipulation.to_json("output/gen_manipulation.jsonl", lines=True, orient="records", force_ascii=False)
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

### Frequency-Aware Sampling

```python
from pacute import load_frequency_data, add_frequency_ranks, sample_by_frequency

# Load frequency data
freq_df = load_frequency_data('data/word_frequencies.csv')

# Add frequency ranks to your dataset
syllables_df = pd.read_json("data/syllables.jsonl", lines=True)
syllables_with_ranks = add_frequency_ranks(syllables_df, freq_df, word_column='normalized_word')

# Sample with different strategies
# freq_weight=0.0: uniform random sampling
# freq_weight=0.5: balanced (default)
# freq_weight=1.0: heavily favor common words
balanced_sample = sample_by_frequency(
    syllables_with_ranks,
    n_samples=100,
    freq_weight=0.5,
    random_state=42
)
```

### String Operations

```python
from pacute import (
    string_to_chars,
    chars_to_string,
    normalize_diacritic,
    diacritize,
    spell_string
)

# Convert string to character list
chars = string_to_chars("kumain")  # ['k', 'u', 'm', 'a', 'i', 'n']

# Spell out a word
spelled = spell_string("ako")  # 'a k o'

# Remove diacritics
clean = normalize_diacritic("kumakáin")  # 'kumakain'

# Add diacritics
accented = diacritize("kumain")  # Randomly adds diacritics
```

## Project Structure

```
pacute/
├── src/pacute/                    # Core library code (refactored & modular)
│   ├── __init__.py                # Package exports and API
│   ├── affixation.py              # Affixation dataset generation
│   ├── composition.py             # Composition task generation
│   ├── constants.py               # Shared constants (MCQ maps, word lengths, etc.)
│   ├── manipulation.py            # String manipulation tasks
│   ├── sampling.py                # Frequency-aware sampling utilities
│   ├── string_operations.py       # Character-level operations
│   ├── syllabification_operations.py  # Core syllabification algorithm
│   ├── syllabification.py         # Syllabification dataset generation
│   └── utils.py                   # Common utilities (output formatting, validation)
├── tasks/                         # Pre-generated evaluation datasets
│   ├── gen_affixation.jsonl       # 140 generative affixation questions
│   ├── gen_composition.jsonl      # 100 generative composition questions
│   ├── gen_manipulation.jsonl     # 160 generative manipulation questions
│   ├── gen_syllabification.jsonl  # 80 generative syllabification questions
│   ├── mcq_composition.jsonl      # 180 MCQ composition questions
│   ├── mcq_manipulation.jsonl     # 160 MCQ manipulation questions
│   ├── mcq_affixation.jsonl       # 140 MCQ affixation questions
│   └── mcq_syllabification.jsonl  # 80 MCQ syllabification questions
├── data/                          # Source data
│   ├── syllables.jsonl            # Processed dictionary (16,828 words)
│   ├── word_frequencies.csv       # Word frequency rankings
│   └── *.jsonl                    # Additional training data
├── exploration/                   # Research notebooks and experiments
│   ├── create_affixation.ipynb
│   ├── create_composition_string_manipulation_syllabification.ipynb
│   ├── create_syllabification.ipynb
│   └── diksiyonaryo.ipynb
├── updf/                          # Source dictionary files (67,691 entries)
├── homographs_dicts/              # Homograph word lists
├── output_folder/                 # Processed dictionary data by letter
├── test_*.py                      # Test scripts for each module
└── DEVELOPER_GUIDE.md             # Developer reference guide
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

Current dataset sizes (in `tasks/` folder):

### Multiple Choice Questions (MCQ)
- **Affixation**: 140 samples
  - Affix inflection: 80 (prefix: 20, suffix: 20, infix: 20, circumfix: 20)
  - Affix identification: 60 (prefix: 20, suffix: 20, infix: 20)
- **Composition**: 180 samples
  - Spelling: 20
  - Character counting (exactly/at least/at most): 60
  - Diacritic counting: 20
  - Uppercase counting: 20
  - Length analysis (exactly/at least/at most): 60
- **Manipulation**: 160 samples
  - Character operations: 100 (insertion, deletion, substitution, permutation, duplication)
  - Case transformations: 40 (uppercasing, lowercasing)
  - Diacritic normalization: 20
- **Syllabification**: 80 samples
  - Stress classification: 20
  - Reduplication detection: 20
  - Syllable counting (ng-aware and general): 40

**MCQ Total: 560 samples**

### Generative Format (GEN)
- **Affixation**: 140 samples
  - Affix inflection: 80 (prefix: 20, suffix: 20, infix: 20, circumfix: 20)
  - Affix identification: 60 (prefix: 20, suffix: 20, infix: 20)
- **Composition**: 100 samples
  - Spelling: 20
  - Character counting: 20
  - Diacritic counting: 20
  - Uppercase counting: 20
  - Length analysis: 20
- **Manipulation**: 160 samples
  - Character operations: 100 (insertion, deletion, substitution, permutation, duplication)
  - Case transformations: 40 (uppercasing, lowercasing)
  - Diacritic normalization: 20
- **Syllabification**: 80 samples
  - Stress classification: 20
  - Reduplication identification: 20
  - Syllable counting (ng-aware and general): 40

**GEN Total: 480 samples**

**Grand Total: 1,040 evaluation items across all task categories**

## Development

### Code Organization

The codebase has been **fully refactored and modularized** for better maintainability:

- **`constants.py`**: Centralized constants (MCQ configurations, word length thresholds, character sets, diacritic mappings)
- **`utils.py`**: Shared utility functions (output formatting, input validation)
- **Eliminated ~375 lines of duplicate code** across dataset generation modules
- **Type hints** added throughout for better IDE support and type checking
- **Comprehensive documentation** with docstrings and usage examples


### Testing

Run the test suite to verify dataset generation:

```bash
python test_affixation.py
python test_composition.py
python test_manipulation.py
python test_syllabification.py
python test_sampling.py
```

All tests should pass, generating sample datasets and verifying correctness.

### Exploration Notebooks

The `exploration/` folder contains Jupyter notebooks used during initial development and experimentation. The production code has been refactored into the `src/pacute/` module for better maintainability and reusability.

## API Reference

### Main Dataset Functions
- `create_affixation_dataset()` - Generate affixation evaluation tasks
- `create_composition_dataset()` - Generate composition/spelling tasks
- `create_manipulation_dataset()` - Generate string manipulation tasks
- `create_syllabification_dataset()` - Generate syllabification tasks

### Utility Functions
- `syllabify()` - Syllabify Filipino words according to phonological rules
- `normalize_text()` - Clean and normalize text (remove diacritics, punctuation)
- `load_frequency_data()` - Load word frequency data
- `add_frequency_ranks()` - Add frequency ranks to DataFrame
- `sample_by_frequency()` - Frequency-aware sampling (configurable weighting)
- `sample_stratified_by_length()` - Length-balanced sampling

### String Operations
- `string_to_chars()` - Convert string to character list
- `chars_to_string()` - Convert character list to string
- `spell_string()` - Spell out string with spaces
- `normalize_diacritic()` - Remove diacritics
- `diacritize()` - Add diacritics to characters

For complete API documentation, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).


## License

MIT License (see LICENSE file)
