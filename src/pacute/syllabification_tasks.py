import pandas as pd
import random

def prepare_mcq_outputs(text_en, text_tl, mcq_options, row={}, kwargs={}):
    outputs = {
        "prompts": [{
            "text_en": text_en.format(**row, **kwargs),
            "text_tl": text_tl.format(**row, **kwargs),
            "mcq_options": mcq_options,
        }],
    }
    return outputs

def prepare_gen_outputs(text_en, text_tl, label, row={}, kwargs={}):
    outputs = {
        "prompts": [{
            "text_en": text_en.format(**row, **kwargs),
            "text_tl": text_tl.format(**row, **kwargs),
        }],
        "label": label
    }
    return outputs

def prepare_options(words, correct_word):
    incorrect_words = [word for word in words if word != correct_word]

    mcq_options = {
        "correct": correct_word,
        "incorrect1": incorrect_words[0] if len(incorrect_words) > 0 else correct_word,
        "incorrect2": incorrect_words[1] if len(incorrect_words) > 1 else correct_word,
        "incorrect3": incorrect_words[2] if len(incorrect_words) > 2 else correct_word,
    }

    return mcq_options


def create_mcq_stress_classification(row):
    text_en = 'What is the stress type of the last syllable in "{word}"?'
    text_tl = 'Ano ang uri ng diin ng huling pantig sa "{word}"?'

    pronunciation_map = {
        "mabilis": "mabilis (acute: á)",
        "malumi": "malumi (grave: à)",
        "maragsa": "maragsa (circumflex: â)",
        "malumay": "malumay (unmarked)"
    }

    correct_pronunciation = row['last_syllable_pronunciation']
    correct_label = pronunciation_map[correct_pronunciation]

    other_pronunciations = [p for p in pronunciation_map.keys() if p != correct_pronunciation]
    random.shuffle(other_pronunciations)

    mcq_options = {
        "correct": correct_label,
        "incorrect1": pronunciation_map[other_pronunciations[0]],
        "incorrect2": pronunciation_map[other_pronunciations[1]],
        "incorrect3": pronunciation_map[other_pronunciations[2]],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def is_reduplicated(word, syllables):
    if len(syllables) < 2:
        return False

    first_syll = syllables[0].lower()
    second_syll = syllables[1].lower()

    # Check for CV reduplication (consonant-vowel pattern)
    # Extract CV from first syllable
    if len(first_syll) >= 2:
        # Get first consonant and first vowel
        cv_pattern = first_syll[:2]

        # Check if second syllable starts with same CV pattern
        if len(second_syll) >= 2:
            if second_syll[:2] == cv_pattern:
                return True

        # Also check single character CV reduplication (e.g., ma-ma)
        if len(first_syll) >= 1 and len(second_syll) >= 1:
            if first_syll[0] == second_syll[0]:
                # Check if it's a simple reduplication (ma-ma, ba-ba)
                if len(first_syll) == len(second_syll) and first_syll == second_syll:
                    return True

    return False

def create_mcq_reduplication_detection(rows):
    text_en = 'Which word has CV-reduplication (first consonant-vowel repeated)?'
    text_tl = 'Alin ang may uulit-pantig (inuulit ang unang katinig-patinig)?'

    reduplicated_words = []
    non_reduplicated_words = []

    for row in rows:
        syllables = row['normalized_syllable_list']
        word = row['normalized_word']
        if is_reduplicated(word, syllables):
            reduplicated_words.append(word)
        else:
            non_reduplicated_words.append(word)

    if len(reduplicated_words) > 0 and len(non_reduplicated_words) >= 3:
        correct_word = reduplicated_words[0]
        incorrect_options = non_reduplicated_words[:3]
    else:
        return None

    mcq_options = {
        "correct": correct_word,
        "incorrect1": incorrect_options[0],
        "incorrect2": incorrect_options[1],
        "incorrect3": incorrect_options[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options)
    return outputs

def create_gen_reduplication_identification(row):
    text_en = 'What is the reduplicated syllable in "{normalized_word}"?'
    text_tl = 'Ano ang inuulit na pantig sa "{normalized_word}"?'

    syllables = row['normalized_syllable_list']

    if len(syllables) >= 2:
        first_syll = syllables[0].lower()
        second_syll = syllables[1].lower()

        # Check for full syllable reduplication first
        if first_syll == second_syll:
            label = first_syll
            outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
            return outputs

        # Then check for CV reduplication
        if len(first_syll) >= 2 and len(second_syll) >= 2:
            if first_syll[:2] == second_syll[:2]:
                label = first_syll[:2]
                outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
                return outputs

    return None


def count_syllables_with_ng(word, syllables):
    return len(syllables)

def create_mcq_ng_syllable_count(row):
    text_en = 'How many syllables in "{normalized_word}"?'
    text_tl = 'Ilang pantig ang "{normalized_word}"?'

    correct_count = len(row['normalized_syllable_list'])

    options = [correct_count - 1, correct_count, correct_count + 1, correct_count + 2]
    options = [o for o in options if o > 0]
    random.shuffle(options[1:])

    mcq_options = prepare_options(options, correct_count)

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_syllabification_dataset(syllables_df, num_samples, mode='mcq', random_seed=100, freq_weight=0.0):
    random.seed(random_seed)
    int2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    if freq_weight > 0:
        from .sampling import load_frequency_data, add_frequency_ranks, sample_by_frequency
        freq_df = load_frequency_data()
        syllables_df = add_frequency_ranks(syllables_df, freq_df)
        syllables_df = sample_by_frequency(
            syllables_df,
            n_samples=len(syllables_df),
            freq_weight=freq_weight,
            random_state=random_seed
        )

    syllables_with_ng = syllables_df[syllables_df['normalized_word'].str.contains('ng', na=False)]

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    if mode == 'mcq':
        for _, row in syllables_df.sample(num_samples).iterrows():
            mcq_row = create_mcq_stress_classification(row)
            dataset = pd.concat([dataset, pd.DataFrame({
                "category": ["syllabification"],
                "subcategory": ["stress_classification"],
                "prompts": [mcq_row["prompts"]],
            })], ignore_index=True)

        shuffled_dataset = syllables_df.sample(frac=1, random_state=42).reset_index(drop=True)
        processed_count = 0
        for _, rows in shuffled_dataset.groupby(lambda x: x // 4):
            if processed_count >= num_samples:
                break

            samples = rows.to_dict(orient="records")
            mcq_row = create_mcq_reduplication_detection(samples)

            if mcq_row is not None:
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["syllabification"],
                    "subcategory": ["reduplication_detection"],
                    "prompts": [mcq_row["prompts"]],
                })], ignore_index=True)
                processed_count += 1

        for _, row in syllables_with_ng.sample(min(num_samples, len(syllables_with_ng))).iterrows():
            mcq_row = create_mcq_ng_syllable_count(row)
            dataset = pd.concat([dataset, pd.DataFrame({
                "category": ["syllabification"],
                "subcategory": ["ng_awareness"],
                "prompts": [mcq_row["prompts"]],
            })], ignore_index=True)

        for i in range(len(dataset)):
            label_index = i % 4
            correct = dataset.iloc[i]['prompts'][0]["mcq_options"]['correct']
            options = [
                dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect1'],
                dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect2'],
                dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect3'],
            ]
            random.shuffle(options)
            options.insert(label_index, correct)
            choices = {
                "choice1": options[0],
                "choice2": options[1],
                "choice3": options[2],
                "choice4": options[3],
            }
            label = int2label[label_index]
            dataset.at[i, 'prompts'][0].update(choices)
            dataset.at[i, 'label'] = label

    elif mode == 'gen':
        for _, row in syllables_df.sample(num_samples * 10).iterrows():
            if len(dataset) >= num_samples:
                break

            gen_row = create_gen_reduplication_identification(row)
            if gen_row is not None:
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["syllabification"],
                    "subcategory": ["reduplication_identification"],
                    "prompts": [gen_row["prompts"]],
                    "label": [gen_row["label"]],
                })], ignore_index=True)

    return dataset
