import string
import unicodedata

vowels = set("AEIOUaeiouÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛàáâèéêìíîòóôùúû")
letter_pairs = set(["bl", "br", "dr", "pl", "tr"])
accented_vowels = set("ÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛàáâèéêìíîòóôùúû")
mabilis = set("ÁÉÍÓÚáéíóú")
malumi = set("ÀÈÌÒÙàèìòù")
maragsa = set("ÂÊÎÔÛâêîôû")


def has_vowel(word):
    return any(let in vowels for let in word)


def slice_value_in_list(list_slice, value_slice, index_slice):
    result = list_slice[:]
    result.insert(value_slice + 1, result[value_slice][index_slice:])
    result[value_slice] = result[value_slice][:index_slice]
    return result


def merge_value_in_list(list_merge, from_merge, to_merge):
    result = list_merge[:]
    result[from_merge : to_merge + 1] = ["".join(result[from_merge : to_merge + 1])]
    return result


def syllabify(word_to_syllabify):
    word = word_to_syllabify

    next_ng = False
    for letter in word:
        if letter in vowels:
            word = word.replace(letter, f" {letter} ")
        elif letter == "-":
            word = word.replace(letter, f" - ")
    word = word.replace("ng", "ŋ").replace("NG", "Ŋ")
    word = word.replace("'", "")
    word = word.split()

    offset = 0

    for index, group in enumerate(word[:]):
        index += offset
        if index == 0 or index == len(word[:]) - 1 or word[index-1] == '-':
            continue
        elif len(group) == 2 and word:
            word = slice_value_in_list(word[:], index, 1)
            offset += 1
        elif len(group) == 3:
            if (
                any((group[0].lower() == "n", group[0].lower() == "m"))
                and group[1:3].lower() in letter_pairs
            ):
                word = slice_value_in_list(word[:], index, 1)
                offset += 1
            else:
                word = slice_value_in_list(word[:], index, 2)
                offset += 1
        elif len(group) > 3:
            word = slice_value_in_list(word[:], index, 2)
            offset += 1

    join_word = word[:]
    offset = 0
    for index, group in enumerate(join_word):
        if (
            group[-1] in vowels
            and join_word[index - 1] not in vowels
            and join_word[index - 1] != "-"
            and index != 0
        ):
            word = merge_value_in_list(word, index - 1 - offset, index - offset)
            offset += 1

    join_word = word[:]
    offset = 0
    for index, group in enumerate(join_word):
        if index != len(join_word) - 1:
            if (
                group[-1] in vowels
                and not has_vowel(join_word[index + 1])
                and join_word[index + 1] != "-"
            ):
                word = merge_value_in_list(word, index - offset, index + 1 - offset)
                offset += 1
    for i in range(len(word)):
        word[i] = word[i].replace("ŋ", "ng").replace("Ŋ", "NG")

    while "-" in word:
        word.remove("-")

    return word


def normalize_text(text):
    if isinstance(text, str):
        text = text.strip()
        punctuation = string.punctuation.replace('-', '')
        text = ''.join(c for c in text if c not in punctuation)
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text


def is_filipino(etymology):
    return any(tag in etymology for tag in ["Tag", "ST", "none"])


def is_single_word(word):
    return len(word.split()) == 1


def has_one_accented_syllable(word):
    syllables = syllabify(word)
    count = sum(1 for syllable in syllables if any(char in accented_vowels for char in syllable))
    return count == 1


def not_circumfixed_with_dash(word):
    return not (word.startswith('-') or word.endswith('-'))


def find_accented_syllable(syllables):
    for i, syllable in enumerate(syllables):
        if any(char in accented_vowels for char in syllable):
            return syllable, i
    return "", -1


def find_last_syllable(syllables):
    return syllables[-1], len(syllables) - 1


def classify_last_syllable_pronunciation(last_syllable):
    if any(char in mabilis for char in last_syllable):
        return "mabilis"
    elif any(char in malumi for char in last_syllable):
        return "malumi"
    elif any(char in maragsa for char in last_syllable):
        return "maragsa"
    else:
        return "malumay"
