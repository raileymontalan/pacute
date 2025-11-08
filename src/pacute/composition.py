import random
import pandas as pd
from .string_operations import (
    string_to_chars, chars_to_string, spell_string, perturb_string, get_random_char
)


def prepare_mcq_outputs(text_en, text_tl, mcq_options, row=None, kwargs=None):
    if row is None:
        row = {}
    if kwargs is None:
        kwargs = {}
    outputs = {
        "prompts": [{
            "text_en": text_en.format(**row, **kwargs),
            "text_tl": text_tl.format(**row, **kwargs),
            "mcq_options": mcq_options,
        }],
    }
    return outputs


def prepare_gen_outputs(text_en, text_tl, label, row=None, kwargs=None):
    if row is None:
        row = {}
    if kwargs is None:
        kwargs = {}
    outputs = {
        "prompts": [{
            "text_en": text_en.format(**row, **kwargs),
            "text_tl": text_tl.format(**row, **kwargs),
        }],
        "label": label
    }
    return outputs


def create_mcq_spelling(row):
    text_en = 'Which option spells out "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang nagbabaybay sa "{normalized_word}"?'

    mcq_correct = spell_string(row['normalized_word'])
    mcq_incorrect = perturb_string(row['normalized_word'])
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_gen_spelling(row):
    text_en = 'Spell out the word "{normalized_word}".'
    text_tl = 'Baybayin ang salitang "{normalized_word}".'

    spelling = string_to_chars(row['normalized_word'])
    label = chars_to_string(spelling, add_space=True)
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row)
    return outputs


def create_gen_character(row):
    text_en = 'How many "{character}"s are in "{normalized_word}"?'
    text_tl = 'Ilang "{character}" ang mayroon sa "{normalized_word}".'

    character_list = string_to_chars(row['normalized_word'])
    character_counts = {char: character_list.count(char) for char in set(character_list)}
    random_character = random.choice(list(character_counts.keys()))
    label = character_counts[random_character]
    kwargs = {"character": random_character}

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_length(row):
    text_en = 'How many characters are in the "{normalized_word}"?'
    text_tl = 'Ilan ang titik sa "{normalized_word}"?'

    label = len(row['normalized_word'])
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row)
    return outputs


diacritics = set("ÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛàáâèéêìíîòóôùúûÑñ")

def check_if_diacritic(char):
    return char in diacritics


def create_gen_diacritic(row):
    text_en = 'How many diacritics are in "{word}"?'
    text_tl = 'Ilang titik ang mayroong tuldik sa "{word}".'

    character_list = string_to_chars(row['word'])
    diacritic_counts = {char: character_list.count(char) for char in set(character_list) if check_if_diacritic(char)}
    label = sum(diacritic_counts.values()) if diacritic_counts else 0
    kwargs = {}

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_uppercase(row):
    text_en = 'How many uppercase characters are in "{normalized_word}"?'
    text_tl = 'Ilang malaking titik ang mayroon sa "{normalized_word}".'

    character_list = string_to_chars(row['normalized_word'])
    uppercase_counts = {char: character_list.count(char) for char in set(character_list) if char.isupper()}
    label = sum(uppercase_counts.values()) if uppercase_counts else 0
    kwargs = {}

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def extract_character_counts(rows, target, char):
    if char is not None:
        character_counts = {}
        for row in rows:
            character_counts[row[target]] = row[target].count(char)
        return character_counts


def prepare_options(words, correct_word):
    incorrect_words = [word for word in words.keys() if word != correct_word]
    mcq_options = {
        "correct": correct_word,
        "incorrect1": incorrect_words[0],
        "incorrect2": incorrect_words[1],
        "incorrect3": incorrect_words[2],
    }
    return mcq_options


def check_if_any_character_counts_are_unique(rows, target):
    words = {}
    for row in rows:
        words[row[target]] = {
            char: row[target].count(char) for char in set(list(row[target]))
        }

    possible_chars = list(set().union(*[set(counts.keys()) for counts in words.values()]))
    random.shuffle(possible_chars)

    for char in possible_chars:
        char_counts = []
        for char_count in words.values():
            char_counts.append(char_count.get(char, 0))

        if len(set(char_counts)) == 4:
            return char

    return None


def check_if_any_diacritic_counts_are_unique(rows, target):
    words = {}
    for row in rows:
        words[row[target]] = {}
        for char in row[target]:
            if check_if_diacritic(char):
                count = row[target].count(char)
                words[row[target]][char] = count

    possible_chars = list(set().union(*[set(counts.keys()) for counts in words.values()]))
    random.shuffle(possible_chars)

    for char in possible_chars:
        char_counts = []
        for char_count in words.values():
            char_counts.append(char_count.get(char, 0))

        if sum(char_counts) == 1:
            return char

    return None


uppercase = set("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
uppercase_diacritics = set("ÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛ")

def check_if_uppercase(char):
    return char in uppercase or char in uppercase_diacritics


def check_if_any_uppercase_counts_are_unique(rows, target):
    words = {}
    for row in rows:
        words[row[target]] = {}
        for char in row[target]:
            if check_if_uppercase(char):
                count = row[target].count(char)
                words[row[target]][char] = count

    possible_chars = list(set().union(*[set(counts.keys()) for counts in words.values()]))
    random.shuffle(possible_chars)

    for char in possible_chars:
        char_counts = []
        for char_count in words.values():
            char_counts.append(char_count.get(char, 0))

        if sum(char_counts) == 1:
            return char

    return None


def create_mcq_char_exactly_one(rows, target, char):
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = 1
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options(character_counts, correct_word)
    kwargs = {"target_count": target_count, "char": char}

    text_en = 'Which option contains exactly {target_count} "{char}"s?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_count} "{char}"?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def prepare_options_capitalize(words, correct_word):
    incorrect_words = [word for word in words.keys() if word != correct_word]

    mcq_options = {
        "correct": correct_word.capitalize(),
        "incorrect1": correct_word.lower(),
        "incorrect2": incorrect_words[1].capitalize(),
        "incorrect3": incorrect_words[2].capitalize(),
    }

    return mcq_options


def create_mcq_uppercase_exactly_one(rows, target, char):
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = 1
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options_capitalize(character_counts, correct_word)
    kwargs = {"target_count": target_count, "char": char}

    text_en = 'Which option contains exactly {target_count} "{char}"s?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_count} "{char}"?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def create_mcq_char_exactly(rows, target, char):
    character_counts = extract_character_counts(rows, target=target, char=char)

    correct_word = random.choice(list(character_counts.keys()))
    target_count = character_counts[correct_word]
    mcq_options = prepare_options(character_counts, correct_word)
    kwargs = {"target_count": target_count, "char": char}

    text_en = 'Which option contains exactly {target_count} "{char}"s?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_count} "{char}"?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def create_mcq_char_most(rows, target, char):
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = max(character_counts.values())
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options(character_counts, correct_word)
    kwargs = {"char": char}

    text_en = 'Which option contains the most number of "{char}"?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakamaraming "{char}"?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def create_mcq_char_least(rows, target, char):
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = min(character_counts.values())
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options(character_counts, correct_word)
    kwargs = {"char": char}

    text_en = 'Which option contains the least number of "{char}"?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakakaunting "{char}"?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def check_if_row_lengths_are_unique(rows, target):
    string1, string2, string3, string4 = rows[0][target], rows[1][target], rows[2][target], rows[3][target]
    strings = {
        string1: len(string1), string2: len(string2), string3: len(string3), string4: len(string4)
    }

    lengths = list(strings.values())
    return len(lengths) == len(set(lengths))


def extract_length(rows, target):
    word1, word2, word3, word4 = rows[0][target], rows[1][target], rows[2][target], rows[3][target]
    words = {
        word1: len(word1), word2: len(word2), word3: len(word3), word4: len(word4)
    }
    return words


def create_mcq_length_exactly(rows):
    words = extract_length(rows, target="normalized_word")

    correct_word = random.choice(list(words.keys()))
    target_length = words[correct_word]
    mcq_options = prepare_options(words, correct_word)
    kwargs = {"target_length": target_length}

    text_en = 'Which option contains exactly {target_length} characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_length} titik?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def create_mcq_length_most(rows):
    words = extract_length(rows, target="normalized_word")

    target_length = max(words.values())
    correct_word = [word for word, length in words.items() if length == target_length][0]
    mcq_options = prepare_options(words, correct_word)
    kwargs = {}

    text_en = 'Which option contains the most number of characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakamaraming titik?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def create_mcq_length_least(rows):
    words = extract_length(rows, target="normalized_word")

    target_length = min(words.values())
    correct_word = [word for word, length in words.items() if length == target_length][0]
    mcq_options = prepare_options(words, correct_word)
    kwargs = {}

    text_en = 'Which option contains the least number of characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakakaunting titik?'

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)
    return outputs


def create_composition_dataset(syllables_df, num_samples, mode='mcq', random_seed=100):
    random.seed(random_seed)
    int2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    if mode == 'mcq':
        tasks = {
            "spelling": (create_mcq_spelling, 'single'),
            "char_exactly": (create_mcq_char_exactly, 'multi'),
            "char_least": (create_mcq_char_least, 'multi'),
            "char_most": (create_mcq_char_most, 'multi'),
            "diacritic_exactly": (create_mcq_char_exactly_one, 'multi'),
            "uppercase_exactly": (create_mcq_uppercase_exactly_one, 'multi'),
            "length_exactly": (create_mcq_length_exactly, 'multi'),
            "length_least": (create_mcq_length_least, 'multi'),
            "length_most": (create_mcq_length_most, 'multi'),
        }

        for subcategory_name, (subcategory_function, task_type) in tasks.items():
            if task_type == 'single':
                for _, row in syllables_df.sample(num_samples).iterrows():
                    mcq_row = subcategory_function(row)
                    dataset = pd.concat([dataset, pd.DataFrame({
                        "category": ["composition"],
                        "subcategory": [subcategory_name],
                        "prompts": [mcq_row["prompts"]],
                    })], ignore_index=True)
            elif task_type == 'multi':
                shuffled_dataset = syllables_df.sample(frac=1, random_state=42).reset_index(drop=True)
                processed_count = 0
                for _, rows in shuffled_dataset.groupby(lambda x: x // 4):
                    if processed_count >= num_samples:
                        break

                    samples = rows.to_dict(orient="records")
                    mcq_row = None

                    if "length" in subcategory_name:
                        valid_length = check_if_row_lengths_are_unique(samples, target="normalized_word")
                        if valid_length:
                            mcq_row = subcategory_function(samples)
                    elif "diacritic_exactly" in subcategory_name:
                        valid_diacritic = check_if_any_diacritic_counts_are_unique(samples, target="word")
                        if valid_diacritic:
                            mcq_row = subcategory_function(samples, target="word", char=valid_diacritic)
                    elif "uppercase_exactly" in subcategory_name:
                        valid_uppercase = check_if_any_uppercase_counts_are_unique(samples, target="word")
                        if valid_uppercase:
                            mcq_row = subcategory_function(samples, target="word", char=valid_uppercase)
                    elif "char" in subcategory_name:
                        valid_char = check_if_any_character_counts_are_unique(samples, target="normalized_word")
                        if valid_char is not None:
                            mcq_row = subcategory_function(samples, target="normalized_word", char=valid_char)

                    if mcq_row is not None:
                        dataset = pd.concat([dataset, pd.DataFrame({
                            "category": ["composition"],
                            "subcategory": [subcategory_name],
                            "prompts": [mcq_row["prompts"]],
                        })], ignore_index=True)
                        processed_count += 1

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
        tasks = {
            "spelling": create_gen_spelling,
            "character": create_gen_character,
            "diacritic": create_gen_diacritic,
            "uppercase": create_gen_uppercase,
            "length": create_gen_length,
        }

        for subcategory_name, subcategory_function in tasks.items():
            for _, row in syllables_df.sample(num_samples * 100).iterrows():
                if len(dataset[dataset['subcategory'] == subcategory_name]) >= num_samples:
                    break

                if subcategory_name == "diacritic" and not any(check_if_diacritic(char) for char in row['word']):
                    continue
                if subcategory_name == "uppercase" and not any(char.isupper() for char in row['normalized_word']):
                    continue

                gen_row = subcategory_function(row)
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["composition"],
                    "subcategory": [subcategory_name],
                    "prompts": [gen_row["prompts"]],
                    "label": [gen_row["label"]],
                })], ignore_index=True)

    return dataset
