import random

diacritic_map = {
    'á': 'a', 'à': 'a', 'â': 'a',
    'é': 'e', 'è': 'e', 'ê': 'e',
    'í': 'i', 'ì': 'i', 'î': 'i',
    'ó': 'o', 'ò': 'o', 'ô': 'o',
    'ú': 'u', 'ù': 'u', 'û': 'u',
    'ñ': 'n',
    'Á': 'A', 'À': 'A', 'Â': 'A',
    'É': 'E', 'È': 'E', 'Ê': 'E',
    'Í': 'I', 'Ì': 'I', 'Î': 'I',
    'Ó': 'O', 'Ò': 'O', 'Ô': 'O',
    'Ú': 'U', 'Ù': 'U', 'Û': 'U',
    'Ñ': 'N',
}

reverse_diacritic_map = {
    'a': ['á', 'à', 'â'],
    'e': ['é', 'è', 'ê'],
    'i': ['í', 'ì', 'î'],
    'o': ['ó', 'ò', 'ô'],
    'u': ['ú', 'ù', 'û'],
    'n': ['ñ'],
    'A': ['Á', 'À', 'Â'],
    'E': ['É', 'È', 'Ê'],
    'I': ['Í', 'Ì', 'Î'],
    'O': ['Ó', 'Ò', 'Ô'],
    'U': ['Ú', 'Ù', 'Û'],
    'N': ['Ñ'],
}


def string_to_chars(string):
    return list(string)


def chars_to_string(char_list, add_space=False):
    if add_space:
        return ' '.join(char_list)
    return ''.join(char_list)


def get_random_char(string):
    return random.choice(list(string))


def same_string(string):
    return string


def delete_char(string, char_to_delete=None):
    if char_to_delete is None:
        char_to_delete = get_random_char(string)
    char_list = string_to_chars(string)
    return chars_to_string([char for char in char_list if char != char_to_delete])


def insert_char(string, preceding_char=None, char_to_insert=None):
    if preceding_char is None:
        preceding_char = get_random_char(string)
    if char_to_insert is None:
        char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        result.append(char)
        if char == preceding_char:
            result.append(char_to_insert)
    return chars_to_string(result)


def substitute_char(string, char_to_replace=None, char_to_substitute=None):
    if char_to_replace is None:
        char_to_replace = get_random_char(string)
    if char_to_substitute is None:
        remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
        char_to_substitute = get_random_char(remaining_chars)

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char == char_to_replace:
            result.append(char_to_substitute)
        else:
            result.append(char)
    return chars_to_string(result)


def permute_char(string, char1=None, char2=None):
    if char1 is None:
        char1 = get_random_char(string)
    if char2 is None:
        remaining_string = string.replace(char1, '')
        if remaining_string:
            char2 = get_random_char(remaining_string)
        else:
            char2 = char1

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char == char1:
            result.append(char2)
        elif char == char2:
            result.append(char1)
        else:
            result.append(char)
    return chars_to_string(result)


def duplicate_char(string, char_to_duplicate=None):
    if char_to_duplicate is None:
        char_to_duplicate = get_random_char(string)

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        result.append(char)
        if char == char_to_duplicate:
            result.append(char)
    return chars_to_string(result)


def normalize_diacritic(string):
    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char in diacritic_map:
            result.append(diacritic_map[char])
        else:
            result.append(char)
    return chars_to_string(result)


def diacritize(string):
    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char in reverse_diacritic_map:
            result.append(random.choice(reverse_diacritic_map[char]))
        else:
            result.append(char)
    return chars_to_string(result)


def randomly_diacritize(string):
    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char in reverse_diacritic_map and random.random() < 0.5:
            result.append(random.choice(reverse_diacritic_map[char]))
        else:
            result.append(normalize_diacritic(char))
    return chars_to_string(result)


def spell_string(string):
    return chars_to_string(string_to_chars(string), add_space=True)


def shuffle_chars(char_list):
    if len(char_list) <= 1:
        return char_list
    shuffled_list = char_list[:]
    while True:
        random.shuffle(shuffled_list)
        if shuffled_list != char_list:
            break
    return shuffled_list


def randomly_merge_chars(char_list):
    merged_list = []
    i = 0
    while i < len(char_list):
        if i < len(char_list) - 1 and random.random() < 0.5:
            merged_list.append(char_list[i] + char_list[i + 1])
            i += 2
        elif i < len(char_list) - 2 and random.random() < 0.5:
            merged_list.append(char_list[i] + char_list[i + 1] + char_list[i + 2])
            i += 3
        else:
            merged_list.append(char_list[i])
            i += 1

    if len(merged_list) == len(char_list):
        idx = random.randint(0, len(char_list) - 2)
        merged_list = (
            char_list[:idx]
            + [char_list[idx] + char_list[idx + 1]]
            + char_list[idx + 2:]
        )
    return merged_list


def randomly_insert_char(char_list):
    idx = random.randint(0, len(char_list))
    char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
    return char_list[:idx] + [char_to_insert] + char_list[idx:]


def randomly_delete_char(char_list):
    if len(char_list) <= 1:
        return char_list
    idx = random.randint(0, len(char_list) - 1)
    return char_list[:idx] + char_list[idx + 1:]


def perturb_string(string):
    char_list = string_to_chars(string)
    perturbation_functions = [
        shuffle_chars,
        randomly_merge_chars,
        randomly_insert_char,
        randomly_delete_char,
    ]
    chosen_functions = random.sample(perturbation_functions, 3)
    results = [chars_to_string(func(char_list), add_space=True) for func in chosen_functions]
    return results
