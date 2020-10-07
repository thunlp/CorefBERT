"""
This evaluation script relies heavily on the one for DROP (``allennlp/tools/drop_eval.py``). We need a separate
script for Quoref only because the data formats are slightly different.
"""

import json
# from typing import Dict, Tuple, List, Any, Optional
import argparse
import numpy as np
import string
import re

from scipy.optimize import linear_sum_assignment


# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def _white_space_fix(text):
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)
def _remove_punc(text):
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def _lower(text):
    return text.lower()

def _tokenize(text):
    return re.split(" |-", text)

def _normalize_answer(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
             for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text):
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted, gold):
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted, gold):
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def _get_answers_from_data(annotations):
    """
    If the annotations file is in the same format as the original data files, this method can be used to extract a
    dict of query ids and answers.
    """
    answers_dict = {}
    for article_info in annotations["data"]:
        for paragraph_info in article_info["paragraphs"]:
            for qa_pair in paragraph_info["qas"]:
                query_id = qa_pair["id"]
                candidate_answers = [answer["text"] for answer in qa_pair["answers"]]
                answers_dict[query_id] = candidate_answers
    return answers_dict

def evaluate_json(annotations, predicted_answers):
    """
    Takes gold annotations and predicted answers and  evaluates the predictions for each question
    in the gold annotations.  Both JSON dictionaries must have query_id keys, which are used to
    match predictions to gold annotations.
    The ``predicted_answers`` JSON must be a dictionary keyed by query id, where the value is a
    list of strings (or just one string) that is the answer.
    The ``annotations`` are assumed to have either the format of the dev set in the Quoref data release, or the
    same format as the predicted answers file.
    """
    instance_exact_match = []
    instance_f1 = []
    if "data" in annotations:
        # We're looking at annotations in the original data format. Let's extract the answers.
        annotated_answers = _get_answers_from_data(annotations)
    else:
        annotated_answers = annotations
    for query_id, candidate_answers in annotated_answers.items():
        max_em_score = 0.0
        max_f1_score = 0.0
        if query_id in predicted_answers:
            predicted = predicted_answers[query_id]
            gold_answer = tuple(candidate_answers)
            em_score, f1_score = get_metrics(predicted, gold_answer)
            if gold_answer[0].strip() != "":
                max_em_score = max(max_em_score, em_score)
                max_f1_score = max(max_f1_score, f1_score)
        else:
            print("Missing prediction for question: {}".format(query_id))
            max_em_score = 0.0
            max_f1_score = 0.0
        instance_exact_match.append(max_em_score)
        instance_f1.append(max_f1_score)

    global_em = np.mean(instance_exact_match)
    global_f1 = np.mean(instance_f1)
    # print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    # print("F1 score {0:.2f}".format(global_f1 * 100))
    # print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    return global_em, global_f1


def evaluate_on_quoref(prediction_path, gold_path):
    """
    Takes a prediction file and a gold file and evaluates the predictions for each question in the gold file.  Both
    files must be json formatted and must have query_id keys, which are used to match predictions to gold
    annotations. Writes a json with global_em and global_f1 metrics to file at the specified output
    path, unless None is passed as output path.
    """
    predicted_answers = json.load(open(prediction_path, encoding='utf-8'))
    annotations = json.load(open(gold_path, encoding='utf-8'))
    global_em, global_f1 = evaluate_json(annotations, predicted_answers)
    return {"exact": global_em, "f1": global_f1}