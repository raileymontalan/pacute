import random
import pandas as pd
import Levenshtein


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


def create_mcq_affixation(row, affix_type):
    if affix_type == "prefix":
        text_en = 'Which option has the prefix "{prefix}-"?'
        text_tl = 'Alin sa mga sumusunod ang may laping "{prefix}-"?'
    elif affix_type == "suffix":
        text_en = 'Which option has the suffix "-{suffix}"?'
        text_tl = 'Alin sa mga sumusunod ang may laping "-{suffix}"?'
    elif affix_type == "infix":
        text_en = 'Which option has the infix "-{infix}-"?'
        text_tl = 'Alin sa mga sumusunod ang may laping "-{infix}-"?'
    elif affix_type == "circumfix":
        text_en = 'Which option has the circumfix "{prefix}-" and "-{suffix}"?'
        text_tl = 'Alin sa mga sumusunod ang may laping "{prefix}-" at "-{suffix}"?'
    else:
        raise ValueError("Invalid affix type. Choose from 'prefix', 'suffix', or 'infix'.")

    mcq_options = {
        "correct": row["correct"],
        "incorrect1": row["incorrect1"],
        "incorrect2": row["incorrect2"],
        "incorrect3": row["incorrect3"],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs={})
    return outputs


def create_gen_affixation(row, affix_type):
    if affix_type == "prefix":
        text_en = 'Inflect the word "{root}" to use the prefix "{prefix}-".'
        text_tl = 'Lapian ng "{prefix}-" ang salitang "{root}".'
    elif affix_type == "suffix":
        text_en = 'Inflect the word "{root}" to use the suffix "-{suffix}".'
        text_tl = 'Lapian ng "-{suffix}" ang salitang "{root}".'
    elif affix_type == "infix":
        text_en = 'Inflect the word "{root}" to use the infix "-{infix}-".'
        text_tl = 'Lapian ng "-{infix}-" ang salitang "{root}".'
    elif affix_type == "circumfix":
        text_en = 'Inflect the word "{root}" to use the circumfix "{prefix}-" and "-{suffix}"?'
        text_tl = 'Lapian ng "{prefix}-" at "-{suffix}" ang salitang "{root}".'
    else:
        raise ValueError("Invalid affix type. Choose from 'prefix', 'suffix', or 'infix'.")

    label = row["correct"]
    outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
    return outputs


def find_similar_incorrect_affixes(affix, unique_affixes):
    levenshtein_ratios = {ua: Levenshtein.ratio(affix, ua) for ua in unique_affixes if ua != affix}
    sorted_affixes = sorted(levenshtein_ratios, key=levenshtein_ratios.get, reverse=True)
    return sorted_affixes[:3]


def create_mcq_reverse_affixation(row, affix_type):
    text_en = 'Which option is the affix used to inflect the word "{correct}"?'
    text_tl = 'Alin sa sumusunod ang lapi na ginamit sa salitang "{correct}"?'

    mcq_options = {
        "correct": row[affix_type],
        "incorrect1": row["incorrect_affixes"][0],
        "incorrect2": row["incorrect_affixes"][1],
        "incorrect3": row["incorrect_affixes"][2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs={})
    return outputs


def create_gen_reverse_affixation(row, affix_type):
    text_en = 'What is the affix used to inflect the word "{correct}"?'
    text_tl = 'Ano ang lapi na ginamit sa salitang "{correct}"?'

    label = row[affix_type]
    outputs = prepare_gen_outputs(text_en, text_tl, label, row=row)
    return outputs


def create_affixation_dataset(inflections_df, mode='mcq', reverse=False, random_seed=42):
    random.seed(random_seed)
    int2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    if mode == 'mcq' and not reverse:
        for _, row in inflections_df.iterrows():
            affix_type = row["affix_type"]
            outputs = create_mcq_affixation(row, affix_type)
            dataset = pd.concat([dataset, pd.DataFrame([{
                "category": "affixation",
                "subcategory": affix_type,
                "prompts": outputs["prompts"],
            }])], ignore_index=True)

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

    elif mode == 'mcq' and reverse:
        unique_affixes = inflections_df["prefix"].unique().tolist() + inflections_df["suffix"].unique().tolist() + inflections_df["infix"].unique().tolist()
        for _, row in inflections_df.iterrows():
            affix_type = row["affix_type"]
            if affix_type == "circumfix":
                continue
            row_copy = row.copy()
            row_copy["incorrect_affixes"] = find_similar_incorrect_affixes(row[affix_type], unique_affixes)
            outputs = create_mcq_reverse_affixation(row_copy, affix_type)
            dataset = pd.concat([dataset, pd.DataFrame([{
                "category": "affixation",
                "subcategory": affix_type,
                "prompts": outputs["prompts"],
            }])], ignore_index=True)

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

    elif mode == 'gen' and not reverse:
        for _, row in inflections_df.iterrows():
            affix_type = row["affix_type"]
            outputs = create_gen_affixation(row, affix_type)
            dataset = pd.concat([dataset, pd.DataFrame([{
                "category": "affixation",
                "subcategory": affix_type,
                "prompts": outputs["prompts"],
                "label": outputs["label"]
            }])], ignore_index=True)

    elif mode == 'gen' and reverse:
        for _, row in inflections_df.iterrows():
            affix_type = row["affix_type"]
            if affix_type == "circumfix":
                continue
            outputs = create_gen_reverse_affixation(row, affix_type)
            dataset = pd.concat([dataset, pd.DataFrame([{
                "category": "affixation",
                "subcategory": affix_type,
                "prompts": outputs["prompts"],
                "label": outputs["label"]
            }])], ignore_index=True)

    return dataset
