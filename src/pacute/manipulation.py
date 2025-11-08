import random
import pandas as pd
from .string_operations import (
    string_to_chars, chars_to_string, get_random_char,
    delete_char, insert_char, substitute_char, permute_char, duplicate_char,
    normalize_diacritic, diacritize, randomly_diacritize, same_string
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


manipulations = {
    "none": same_string,
    "deletion": delete_char,
    "insertion": insert_char,
    "substitution": substitute_char,
    "permutation": permute_char,
    "duplication": duplicate_char,
}


def get_invalid_manipulations(target_manipulation="insertion"):
    return [(name, func) for name, func in manipulations.items() if name != target_manipulation]


def apply_manipulation_incorrectly(string, target_manipulation="deletion", kwargs=None):
    if kwargs is None:
        kwargs = {}

    if target_manipulation == "deletion" and "char_to_delete" in kwargs:
        incorrect_char = kwargs["char_to_delete"]
        remaining_chars = string.replace(incorrect_char, '')
        if remaining_chars:
            incorrect_char = get_random_char(remaining_chars)
        return delete_char(string, char_to_delete=incorrect_char)
    elif target_manipulation == "insertion" and "preceding_char" in kwargs and "char_to_insert" in kwargs:
        preceding_char = kwargs["preceding_char"]
        char_to_insert = kwargs["char_to_insert"]
        remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_insert, '')
        incorrect_char_to_insert = get_random_char(remaining_chars)
        return insert_char(string, preceding_char=preceding_char, char_to_insert=incorrect_char_to_insert)
    elif target_manipulation == "substitution" and "char_to_replace" in kwargs and "char_to_substitute" in kwargs:
        char_to_replace = kwargs["char_to_replace"]
        char_to_substitute = kwargs["char_to_substitute"]
        remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
        remaining_chars = remaining_chars.replace(char_to_substitute, '')
        incorrect_char_to_substitute = get_random_char(remaining_chars)
        return substitute_char(string, char_to_replace=char_to_replace, char_to_substitute=incorrect_char_to_substitute)
    elif target_manipulation == "permutation" and "char1" in kwargs and "char2" in kwargs:
        char1 = kwargs["char1"]
        char2 = kwargs["char2"]
        remaining_chars = string.replace(char1, '')
        remaining_chars = remaining_chars.replace(char2, '')
        incorrect_char2 = get_random_char(remaining_chars)
        return permute_char(string, char1=char1, char2=incorrect_char2)
    elif target_manipulation == "duplication" and "char_to_duplicate" in kwargs:
        char_to_duplicate = kwargs["char_to_duplicate"]
        remaining_chars = string.replace(char_to_duplicate, '')
        if remaining_chars:
            incorrect_char_to_duplicate = get_random_char(remaining_chars)
        else:
            incorrect_char_to_duplicate = char_to_duplicate
        return duplicate_char(string, char_to_duplicate=incorrect_char_to_duplicate)


def manipulate_string(string, target_manipulation="deletion", kwargs=None):
    if kwargs is None:
        kwargs = {}

    manipulation_functions = get_invalid_manipulations(target_manipulation=target_manipulation)
    chosen_functions = random.sample([func for name, func in manipulation_functions], 2)
    results = [func(string) for func in chosen_functions]

    incorrect_application = apply_manipulation_incorrectly(string, target_manipulation=target_manipulation, kwargs=kwargs)
    results.append(incorrect_application)
    return results


def diacritize_string(string, correct_string):
    results = [
        same_string(string),
        diacritize(string),
        randomly_diacritize(string),
    ]

    if correct_string in results:
        results.remove(correct_string)
        from .string_operations import shuffle_chars
        results.append(chars_to_string(shuffle_chars(string_to_chars(correct_string))))
    return results


def create_mcq_deletion(row):
    text_en = 'Which option correctly removes every "{char_to_delete}" in "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang nagtatanggal ng bawat "{char_to_delete}" sa "{normalized_word}"?'

    string = row['normalized_word']
    char_to_delete = get_random_char(string)
    kwargs = {"char_to_delete": char_to_delete}

    mcq_correct = delete_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="deletion", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_insertion(row):
    text_en = 'Which option correctly puts "{char_to_insert}" after every "{preceding_char}" in "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang naglalagay ng "{char_to_insert}" pagkatapos ng bawat "{preceding_char}" sa "{normalized_word}"?'

    string = row['normalized_word']
    preceding_char = get_random_char(string)
    char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
    kwargs = {"preceding_char": preceding_char, "char_to_insert": char_to_insert}

    mcq_correct = insert_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="insertion", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_substitution(row):
    text_en = 'Which option correctly replaces every "{char_to_replace}" with "{char_to_substitute}" in "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang pumapalit sa bawat "{char_to_replace}" gamit ng "{char_to_substitute}" sa "{normalized_word}"?'

    string = row['normalized_word']
    char_to_replace = get_random_char(string)
    remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
    char_to_substitute = get_random_char(remaining_chars)
    kwargs = {"char_to_replace": char_to_replace, "char_to_substitute": char_to_substitute}

    mcq_correct = substitute_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="substitution", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_permutation(row):
    text_en = 'Which option correctly swaps every "{char1}" with "{char2}" and vice versa in "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang pumapalit sa bawat "{char1}" gamit ng "{char2}" at ang kabaligtarang din nito sa "{normalized_word}"?'

    string = row['normalized_word']
    char1 = get_random_char(string)
    remaining_string = string.replace(char1, '')
    char2 = get_random_char(remaining_string)
    kwargs = {"char1": char1, "char2": char2}

    mcq_correct = permute_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="permutation", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_duplication(row):
    text_en = 'Which option correctly duplicates every "{char_to_duplicate}" once in "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang umuulit sa bawat "{char_to_duplicate}" nang isang beses sa "{normalized_word}"?'

    string = row['normalized_word']
    char_to_duplicate = get_random_char(string)
    kwargs = {"char_to_duplicate": char_to_duplicate}

    mcq_correct = duplicate_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="duplication", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_uppercasing(row):
    text_en = 'Which option correctly changes "{normalized_word}" to all uppercase?'
    text_tl = 'Alin sa sumusunod ang ginagawang malaki ang lahat ng titik sa "{normalized_word}"?'

    string = row['normalized_word']
    mcq_correct = string.upper()
    mcq_incorrect = [string.lower(), string[:len(string)//2].upper() + string[len(string)//2:].lower(), ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in string)]
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_mcq_lowercasing(row):
    text_en = 'Which option correctly changes "{normalized_word}" to all lowercase?'
    text_tl = 'Alin sa sumusunod ang ginagawang maliit ang lahat ng titik sa "{normalized_word}"?'

    string = row['normalized_word']
    mcq_correct = string.lower()
    mcq_incorrect = [string[:len(string)//2].upper() + string[len(string)//2:].lower(), ''.join(c.lower() if random.random() < 0.5 else c.upper() for c in string), string.upper()]
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_mcq_diacritic_normalization(row):
    text_en = 'Which option correctly normalizes diacritics from "{word}"?'
    text_tl = 'Alin sa sumusunod ang nagtatanggal ng mga tuldik sa "{word}"?'

    string = row['word']
    mcq_correct = normalize_diacritic(string)
    mcq_incorrect = diacritize_string(string, mcq_correct)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_gen_deletion(row):
    text_en = 'Remove every "{char_to_delete}" in "{normalized_word}".'
    text_tl = 'Tanggalin ang bawat "{char_to_delete}" sa "{normalized_word}".'

    string = row['normalized_word']
    char_to_delete = get_random_char(string)
    kwargs = {"char_to_delete": char_to_delete}
    label = delete_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_insertion(row):
    text_en = 'Put a "{char_to_insert}" after every "{preceding_char}" in "{normalized_word}"'
    text_tl = 'Maglagay ng "{char_to_insert}" pagkatapos ng bawat "{preceding_char}" sa "{normalized_word}"'

    string = row['normalized_word']
    preceding_char = get_random_char(string)
    char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
    kwargs = {"preceding_char": preceding_char, "char_to_insert": char_to_insert}
    label = insert_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_substitution(row):
    text_en = 'Replace every "{char_to_replace}" with "{char_to_substitute}" in "{normalized_word}".'
    text_tl = 'Palitan ang bawat "{char_to_replace}" gamit ng "{char_to_substitute}" sa "{normalized_word}".'

    string = row['normalized_word']
    char_to_replace = get_random_char(string)
    remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
    char_to_substitute = get_random_char(remaining_chars)
    kwargs = {"char_to_replace": char_to_replace, "char_to_substitute": char_to_substitute}
    label = substitute_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_permutation(row):
    text_en = 'Swap every "{char1}" with "{char2}" in "{normalized_word}".'
    text_tl = 'Palitan ang bawat "{char1}" gamit ng "{char2}" at ang kabaligtarang din nito sa "{normalized_word}".'

    string = row['normalized_word']
    char1 = get_random_char(string)
    remaining_string = string.replace(char1, '')
    char2 = get_random_char(remaining_string)
    kwargs = {"char1": char1, "char2": char2}
    label = permute_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_duplication(row):
    text_en = 'Duplicate every "{char_to_duplicate}" once in "{normalized_word}".'
    text_tl = 'Ulitin ang bawat "{char_to_duplicate}" nang isang beses sa "{normalized_word}".'

    string = row['normalized_word']
    char_to_duplicate = get_random_char(string)
    kwargs = {"char_to_duplicate": char_to_duplicate}
    label = duplicate_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_uppercasing(row):
    text_en = 'Change "{normalized_word}" into uppercase.'
    text_tl = 'Gawing malaki ang lahat ng titik sa "{normalized_word}".'

    label = row["normalized_word"].upper()
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})
    return outputs


def create_gen_lowercasing(row):
    text_en = 'Change "{normalized_word}" into lowercase.'
    text_tl = 'Gawing maliit ang lahat ng titik sa "{normalized_word}".'

    label = row["normalized_word"].lower()
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})
    return outputs


def create_gen_diacritic_normalization(row):
    text_en = 'Normalize the diacritics from "{word}".'
    text_tl = 'Tanggalin ang lahat ng mga tuldik sa "{word}".'

    label = normalize_diacritic(row["word"])
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})
    return outputs


def create_manipulation_dataset(syllables_df, num_samples, mode='mcq', random_seed=100):
    random.seed(random_seed)
    int2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    if mode == 'mcq':
        tasks = {
            "insertion": create_mcq_insertion,
            "deletion": create_mcq_deletion,
            "substitution": create_mcq_substitution,
            "permutation": create_mcq_permutation,
            "duplication": create_mcq_duplication,
            "uppercasing": create_mcq_uppercasing,
            "lowercasing": create_mcq_lowercasing,
            "diacritic_normalization": create_mcq_diacritic_normalization,
        }

        for subcategory_name, subcategory_function in tasks.items():
            for _, row in syllables_df.sample(num_samples).iterrows():
                mcq_row = subcategory_function(row)
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["manipulation"],
                    "subcategory": [subcategory_name],
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
        tasks = {
            "insertion": create_gen_insertion,
            "deletion": create_gen_deletion,
            "substitution": create_gen_substitution,
            "permutation": create_gen_permutation,
            "duplication": create_gen_duplication,
            "uppercasing": create_gen_uppercasing,
            "lowercasing": create_gen_lowercasing,
            "diacritic_normalization": create_gen_diacritic_normalization,
        }

        for subcategory_name, subcategory_function in tasks.items():
            for _, row in syllables_df.sample(num_samples * 100).iterrows():
                if len(dataset[dataset['subcategory'] == subcategory_name]) >= num_samples:
                    break

                from .composition import check_if_diacritic
                if subcategory_name == "diacritic_normalization" and not any(check_if_diacritic(char) for char in row['word']):
                    continue
                if subcategory_name == "lowercasing" and not any(char.isupper() for char in row['normalized_word']):
                    continue
                if subcategory_name == "uppercasing" and not any(char.islower() for char in row['normalized_word']):
                    continue

                gen_row = subcategory_function(row)
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["manipulation"],
                    "subcategory": [subcategory_name],
                    "prompts": [gen_row["prompts"]],
                    "label": [gen_row["label"]],
                })], ignore_index=True)

    return dataset
