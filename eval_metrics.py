import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from collections import Counter
import argparse
import pandas as pd
import os
import math
import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from evaluate import load
import numpy as np
import csv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
rouge = load("rouge")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_TEMPLATE = "You are provided with SMILES of two drugs:\nSMILES 1: {smile1}\nSMILES 2: {smile2}\nBased on the SMILES of these two drugs, describe their interaction mechanism in one sentence, assuming typical pharmacological behavior. Do not include any reasoning or uncertainty."

advisory_terms = ["ADDITIONAL CONTRACEPTION RECOMMENDED", "ADJUST DOSE", "ADJUST DOSING INTERVAL", "CONTRAINDICATED", "GENERALLY AVOID", "MONITOR", "MONITOR CLOSELY", "NONE"]
interaction_types = ["major", "moderate", "minor"]

# def compute_metrics(y_true, y_pred, data_type):
#     if data_type == "classification_interaction":
#         valid_labels = interaction_types
#     elif data_type == "advisory_terms":
#         valid_labels = advisory_terms
#     else:
#         raise ValueError(f"data_type must be either 'classification_interaction' or 'advisory_terms'.")

#     # Map labels to indices
#     valid_label_indices = {label: idx for idx, label in enumerate(valid_labels)}

#     # Convert y_true and y_pred to indices (map valid labels)
#     y_true_indices = [valid_label_indices.get(label, -1) for label in y_true]
#     y_pred_indices = [valid_label_indices.get(label, -1) for label in y_pred]

#     # Count invalid predictions and treat them as incorrect predictions
#     invalid_predictions_count = sum(1 for label in y_pred_indices if label == -1)

#     # Replace invalid predictions with a special class index
#     invalid_class_index = len(valid_labels)  # Additional class for invalid predictions
#     y_pred_indices = [invalid_class_index if label == -1 else label for label in y_pred_indices]

#     # Define all classes including the invalid one
#     all_valid_classes = list(range(len(valid_labels))) + [invalid_class_index]

#     metrics = {   
#         "accuracy": accuracy_score(y_true_indices, y_pred_indices),
#         "weighted_f1": f1_score(y_true_indices, y_pred_indices, labels=all_valid_classes, average="weighted", zero_division=0),
#         "weighted_macro": precision_score(y_true_indices, y_pred_indices, labels=all_valid_classes, average="weighted",  zero_division=0),
#         "weighted_macro": recall_score(y_true_indices, y_pred_indices, labels=all_valid_classes, average="weighted",  zero_division=0),
#         "invalid_predictions_count": invalid_predictions_count
#     }

#     return metrics

def compute_metrics(y_true, y_pred, data_type):
    if data_type == "classification_interaction":
        valid_labels = interaction_types
    elif data_type == "advisory_terms":
        valid_labels = advisory_terms
    else:
        raise ValueError(f"data_type must be either 'classification_interaction' or 'advisory_terms'.")

    valid_label_indices = {label: idx for idx, label in enumerate(valid_labels)}

    y_true_indices = [valid_label_indices.get(label, -1) for label in y_true]
    y_pred_indices = [valid_label_indices.get(label, -1) for label in y_pred]

    invalid_predictions_count = sum(1 for label in y_pred_indices if label == -1)

    invalid_class_index = len(valid_labels)
    y_pred_indices = [invalid_class_index if label == -1 else label for label in y_pred_indices]

    all_valid_classes = list(range(len(valid_labels))) + [invalid_class_index]

    accuracy = accuracy_score(y_true_indices, y_pred_indices)
    f1_macro = f1_score(y_true_indices, y_pred_indices, labels=all_valid_classes, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true_indices, y_pred_indices, labels=all_valid_classes, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true_indices, y_pred_indices, labels=all_valid_classes, average="weighted", zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "f1_weighted_with_invalid": f1_macro,
        "precision_weighted_with_invalid": precision_macro,
        "recall_weighted_with_invalid": recall_macro,
        "invalid_predictions_count": invalid_predictions_count
    }

    return metrics


def semantic_similarity(embeddings1, embeddings2):
    embeddings1 = embeddings1.to(device)
    embeddings2 = embeddings2.to(device)
    
    if embeddings1.dim() == 1:
        embeddings1 = embeddings1.unsqueeze(0)
    if embeddings2.dim() == 1:
        embeddings2 = embeddings2.unsqueeze(0)
    
    return F.cosine_similarity(embeddings1, embeddings2, dim=1)

def bleu_n_score(predicted, ground_truth, n):
    predicted_tokens = predicted.split()
    ground_truth_tokens = ground_truth.split()
    
    references = [ground_truth_tokens]
    hypothesis = predicted_tokens
    
    weights = [0] * n
    weights[n - 1] = 1.0
    weights = tuple(weights)
    
    smoothing_function = SmoothingFunction().method1
    
    bleu_n = sentence_bleu(
        references,
        hypothesis,
        weights=weights,
        smoothing_function=smoothing_function
    )
    return bleu_n


def meteor_score(reference, hypothesis):
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    stemmer = PorterStemmer()
    ref_stems = [stemmer.stem(word) for word in ref_tokens]
    hyp_stems = [stemmer.stem(word) for word in hyp_tokens]

    exact_matches = set(hyp_tokens) & set(ref_tokens)
    stem_matches = set(hyp_stems) & set(ref_stems)
    
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
        return synonyms
    
    ref_synonyms = {word: get_synonyms(word) for word in ref_tokens}
    synonym_matches = set()
    for word in hyp_tokens:
        for ref_word, synonyms in ref_synonyms.items():
            if word in synonyms:
                synonym_matches.add(word)
                break

    total_matches = exact_matches | stem_matches | synonym_matches
    
    precision = len(total_matches) / len(hyp_tokens) if hyp_tokens else 0
    recall = len(total_matches) / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall > 0:
        score = (10 * precision * recall) / (9 * recall + precision)
    else:
        score = 0

    return score

def eval_new_data(data, bleu_n_list=[1,2,3,4]):
    results = {}
    sims = []
    meteors = []
    bleu_scores = {n: [] for n in bleu_n_list}
    rouge1_scores = []
    rougel_scores = []

    for smiles, entry in tqdm(data.items(), desc="Evaluating"):
        query, gt, pred = entry
        gt = gt.strip() if gt else ""
        pred = pred.strip() if pred else ""

        emb_gt = model.encode(gt, convert_to_tensor=True)
        emb_pred = model.encode(pred, convert_to_tensor=True)
        sim = semantic_similarity(emb_gt, emb_pred).item()
        sims.append(sim)

        met = meteor_score(gt, pred)
        meteors.append(met)

        results[smiles] = {
            "query": query,
            "ground_truth": gt,
            "predicted": pred,
            "semantic_similarity": sim,
            "meteor": met
        }

        for n in bleu_n_list:
            b = bleu_n_score(pred, gt, n)
            bleu_scores[n].append(b)
            results[smiles][f"bleu_{n}"] = b

        # ROUGE
        rc = rouge.compute(
            predictions=[pred],
            references=[gt],
            rouge_types=["rouge1","rougeL"]
        )
        r1 = rc["rouge1"]
        rL = rc["rougeL"]
        rouge1_scores.append(r1)
        rougel_scores.append(rL)
        results[smiles]["rouge_1"] = r1
        results[smiles]["rouge_L"] = rL

    # Mean scores
    mean_s = np.mean(sims) if sims else 0
    mean_m = np.mean(meteors) if meteors else 0
    mean_bleu = {n: np.mean(bleu_scores[n]) if bleu_scores[n] else 0 for n in bleu_n_list}
    mean_r1 = np.mean(rouge1_scores) if rouge1_scores else 0
    mean_rL = np.mean(rougel_scores) if rougel_scores else 0

    mean_results = {
        "mean_semantic_similarity": mean_s,
        "mean_meteor": mean_m,
        **{f"mean_bleu_{n}": mean_bleu[n] for n in bleu_n_list},
        "mean_rouge_1": mean_r1,
        "mean_rouge_L": mean_rL
    }
    results["mean_scores"] = mean_results
    return results
            

def eval_4_types(data):
    
    def preprocess_interaction_type(texts):
        ret = []
        for text in texts:
            text = text.lower()
            if "major" in text:
                ret.append("major")
            elif "moderate" in text:
                ret.append("moderate")
            elif "minor" in text:
                ret.append("minor")
            else:
                ret.append("unknown")
        return ret
    
    long_question_types = {"management": 2, "mechanism_interaction": 3}
    short_question_types = {"advisory_terms": 0, "classification_interaction": 1}
    results = {}
    similarities = {"management": [], "mechanism_interaction": []}
    meteor_scores = {"management": [], "mechanism_interaction": []}
    bleu_n_list = [1, 2, 3, 4]
    bleu_n_scores =  {"management":  {n: [] for n in bleu_n_list}, "mechanism_interaction":  {n: [] for n in bleu_n_list}}
    for smiles, content in tqdm(data.items(), desc="Evaluating"):
        for question_type, index in long_question_types.items():
            query, ground_truth, predicted_text = content[index]
            ground_truth_text = ground_truth.strip() if ground_truth else ""
            predicted_text = predicted_text.strip() if predicted_text else ""
            
            embedding_ground_truth = model.encode(ground_truth_text, convert_to_tensor=True).to(device)
            embedding_predicted = model.encode(predicted_text, convert_to_tensor=True).to(device)

            similarity = semantic_similarity(embedding_ground_truth, embedding_predicted).item()
            similarities[question_type].append(similarity)
            
            meteor = meteor_score(ground_truth_text, predicted_text)
            meteor_scores[question_type].append(meteor)
            
            if smiles not in results:
                results[smiles] = {}

            if question_type not in results[smiles]:
                results[smiles][question_type] = {}
                        
            results[smiles][question_type] = {
                "query": query,
                "ground_truth": ground_truth_text,
                "predicted": predicted_text,
                "semantic_similarity": similarity,
                "meteor": meteor
            }
            
            for n in bleu_n_list:
                bleu_n = bleu_n_score(predicted_text, ground_truth_text, n)
                results[smiles][question_type][f"bleu_{n}"] = bleu_n
                bleu_n_scores[question_type][n].append(bleu_n)
                       
    for question_type in long_question_types:
        mean_similarity = sum(similarities[question_type]) / len(similarities[question_type]) if similarities[question_type] else 0.0
        mean_meteor = sum(meteor_scores[question_type]) / len(meteor_scores[question_type]) if meteor_scores[question_type] else 0.0
        results[question_type] = {
            "mean_similarity": mean_similarity,
            "mean_meteor": mean_meteor
        }
        for n in bleu_n_list:
            mean_bleu_n = sum(bleu_n_scores[question_type][n]) / len(bleu_n_scores[question_type][n]) if bleu_n_scores[question_type][n] else 0.0
            results[question_type][f"mean_bleu_{n}"] = mean_bleu_n
    
    mean_similarity = sum(similarities["management"] + similarities["mechanism_interaction"]) / (len(similarities["management"]) + len(similarities["mechanism_interaction"]))
    mean_meteor = sum(meteor_scores["management"] + meteor_scores["mechanism_interaction"]) / (len(meteor_scores["management"]) + len(meteor_scores["mechanism_interaction"]))
    results["mean_scores"] = {
        "semantic_similarity": mean_similarity,
        "meteor": mean_meteor
    }
    mean_bleu_n = {}
    for n in bleu_n_list:
        mean_bleu_n[n] = sum(bleu_n_scores["management"][n] + bleu_n_scores["mechanism_interaction"][n]) / (len(bleu_n_scores["management"][n]) + len(bleu_n_scores["mechanism_interaction"][n]))
        results["mean_scores"][f"bleu_{n}"] = mean_bleu_n[n]
    
    # Add code to compute mean_bleu_n and mean_meteor for each question type, and compute the mean of those values
    
    for short_question_type, short_question_type_idx in short_question_types.items():
        ground_truths = [content[short_question_type_idx][1] for content in data.values()]
        predictions = [content[short_question_type_idx][2] for content in data.values()]
        if "classification_interaction" in short_question_type:
            ground_truths = preprocess_interaction_type(ground_truths)
            predictions = preprocess_interaction_type(predictions)
        elif "advisory_terms" in short_question_type:
            ground_truths = [ground_truth.upper() for ground_truth in ground_truths]
            predictions = [prediction.upper() for prediction in predictions]
        metrics = compute_metrics(ground_truths, predictions, short_question_type)
        results[short_question_type] = metrics
    return results

def gather_ground_truth_from_smiles(ours_results):
    smiles_pairs_ground_truth = {}
    for smiles, content in ours_results.items():
        ground_truth = content[0][1]
        smiles_pairs_ground_truth[smiles] = ground_truth 
    return smiles_pairs_ground_truth
        

def process_data_from_gpt(file_path, smiles_pairs_ground_truth, ours_results):
    results = {}
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    for _, row in df.iterrows():
        smile_1 = row['Smile_1']
        smile_2 = row['Smile_2']
        
        classification_interaction = row['Classification_interaction']
        if pd.isna(classification_interaction) or not isinstance(classification_interaction, str):
            classification_interaction = "unknown"
        else:
            classification_interaction = classification_interaction.lower()
        
        mechanism_interaction = row['Mechanism_interaction']
        if pd.isna(mechanism_interaction) or not isinstance(mechanism_interaction, str):
            mechanism_interaction = ""
            
        management = row['Mangement']
        if pd.isna(management) or not isinstance(management, str):
            management = ""
        
        advisory_terms = row['Advisory terms']
        if pd.isna(advisory_terms) or not isinstance(advisory_terms, str):
            advisory_terms = ""
        else:
            advisory_terms = advisory_terms.upper()
        
        # predicted_text = f"The drug interactions are {classification_interaction}. {advisory_terms}: {mechanism_interaction} MANAGEMENT: {management}"
        ours_result = ours_results.get(f"{smile_1}|{smile_2}", None)
        # assert ours_result is not None, f"Ours result not found for {smile_1}|{smile_2}"
        if ours_result is None:
            continue
        ground_truth_advisory_terms = ours_result[0][1]
        ground_truth_classification_interaction = ours_result[1][1]
        ground_truth_mechanism_interaction = ours_result[2][1]
        ground_truth_management = ours_result[3][1]
        
        smiles_str = f"{smile_1}|{smile_2}"
        ground_truth = smiles_pairs_ground_truth.get(smiles_str, "")
        if not ground_truth:
            smiles_str = f"{smile_2}|{smile_1}"
            ground_truth = smiles_pairs_ground_truth.get(smiles_str, "")
        if not ground_truth:
            print(f"Ground truth not found for {smiles_str}")
            continue
        results[smiles_str] = [
            [
                "What is the advisory terms?",
                ground_truth_advisory_terms,
                advisory_terms
            ],
            [
                "Categorize the drug interactions as either 'minor,' 'moderate,' and 'major' based on the given context.",
                ground_truth_classification_interaction,
                classification_interaction
            ],
            [
                "What is the mechanism of interaction?",
                ground_truth_mechanism_interaction,
                mechanism_interaction
            ],
            [
                "What is the management?",
                ground_truth_management,
                management
            ]
        ] 
    return results


def preprocess_gpt_data_for_new_dataset(gpt_data_path):
    gpt_data = {}
    with open(gpt_data_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            smiles = "|".join(row[:2])
            ground_truth = row[2]
            classification_interaction = row[3]
            prompt = PROMPT_TEMPLATE.format(smile1=row[0], smile2=row[1])
            if smiles not in gpt_data:
                gpt_data[smiles] = [
                    prompt,
                    ground_truth,
                    classification_interaction
                ]
    return gpt_data
    
    
def main(args):
    with open(os.path.join(CURRENT_DIR, "eval_results/new/10-epochs-lr-1e-5-iter-2206-results.json"), 'r') as f:
        ours_data = json.load(f)

    ours_results = eval_new_data(ours_data)
    with open(os.path.join(CURRENT_DIR, "eval_results/new/ours.json"), 'w') as f:
        json.dump(ours_results, f, indent=4)

    # GPT results
    # smiles_gt = gather_ground_truth_from_smiles(ours_data)
    # gpt_data = process_data_from_gpt(
    #     os.path.join(CURRENT_DIR, "eval_results/Drug_Drug_Interaction_GPT.xlsx"),
    #     smiles_gt, ours_data
    # )
    # gpt_results = eval(gpt_data)
    # with open(os.path.join(CURRENT_DIR, "eval_results/new/gpt_results.json"), 'w') as f:
    #     json.dump(gpt_results, f, indent=4)

    # Test set
    test_data = preprocess_gpt_data_for_new_dataset(
        os.path.join(CURRENT_DIR, "eval_results/new/gpt_test_results.csv")
    )
    test_results = eval_new_data(test_data)
    with open(os.path.join(CURRENT_DIR, "eval_results/new/gpt_results.json"), 'w') as f:
        json.dump(test_results, f, indent=4)

    # Apollo-MoE-7B
    with open(os.path.join(CURRENT_DIR, "eval_results/new/Apollo-MoE-7B_results.json"), 'r') as f:
        mmed = json.load(f)
    mmed_eval = eval_new_data(mmed)
    with open(os.path.join(CURRENT_DIR, "eval_results/new/Apollo-MoE-7B_results_eval.json"), 'w') as f:
        json.dump(mmed_eval, f, indent=4)
    
    # MMed-Llama-3-8B_results_eval
    with open(os.path.join(CURRENT_DIR, "eval_results/new/MMed-Llama-3-8B_results.json"), 'r') as f:
        mmed = json.load(f)
    mmed_eval = eval_new_data(mmed)
    with open(os.path.join(CURRENT_DIR, "eval_results/new/MMed-Llama-3-8B_results_eval.json"), 'w') as f:
        json.dump(mmed_eval, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)