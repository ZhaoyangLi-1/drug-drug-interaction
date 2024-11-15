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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_TEMPLATE = (
    "Two drugs are provided with the following SMILES notations:\n\n"
    "Drug 1 SMILES: {smiles1}\n"
    "Drug 2 SMILES: {smiles2}\n\n"
    "Please analyze the possible interactions between these two drugs and Provide only four things:\n"
    "Classification of interaction; classify strictly into three (Major, Moderate, Minor) classes, give response as *ans_1:*.\n"
    "Mechanism of interaction, give response as *ans_2:*.\n\n"
    "Management, give response as *ans_3:*.\n\n"
    "Give Advisory terms strictly from ['ADDITIONAL CONTRACEPTION RECOMMENDED', 'ADJUST DOSE', 'ADJUST DOSING INTERVAL', 'CONTRAINDICATED', 'GENERALLY AVOID', 'MONITOR', 'MONITOR CLOSELY'], as *ans_4:*."
    "Use scientific terminology and provide a detailed but concise response for the mechanism of interaction and management."
)

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

def compute_meteor_score(predicted, ground_truth):
    references = [ground_truth]
    hypothesis = predicted
    
    score = meteor_score(references, hypothesis)
    return score


def eval(data):
    results = {}
    similarities = []
    bleu_1_scores = []
    meteor_scores = []
    
    for smiles, content in tqdm(data.items(), desc="Evaluating"):
        if len(content) == 1:
            query, ground_truth_list, predicted_text = content[0]
            ground_truth_text = ground_truth_list[0] if ground_truth_list else ""
            if type(ground_truth_text) != str:
                ground_truth_text = ground_truth_text[0]
            
            embedding_ground_truth = model.encode(ground_truth_text, convert_to_tensor=True).to(device)
            embedding_predicted = model.encode(predicted_text, convert_to_tensor=True).to(device)

            similarity = semantic_similarity(embedding_ground_truth, embedding_predicted).item()
            similarities.append(similarity)
            
            
            meteor = meteor_score(predicted_text, ground_truth_text)
            meteor_scores.append(meteor)
            
            results[smiles] = {
                "query": query,
                "ground_truth": ground_truth_text,
                "predicted": predicted_text,
                "semantic_similarity": similarity,
                "meteor": meteor
            }
            
            bleu_n_list = [1, 2, 3, 4]
            for n in bleu_n_list:
                bleu_n = bleu_n_score(predicted_text, ground_truth_text, n)
                results[smiles][f"bleu_{n}"] = bleu_n
        else:
            print(f"Skipping entry {smiles} due to unexpected structure.")
    
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    mean_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0.0
    mean_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    results["mean_scores"] = {
        "semantic_similarity": mean_similarity,
        "bleu_1": mean_bleu_1,
        "meteor": mean_meteor
    }
    
    return results

def gather_ground_truth_from_smiles(ours_results):
    smiles_pairs_ground_truth = {}
    for smiles, content in ours_results.items():
        ground_truth = content[0][1]
        smiles_pairs_ground_truth[smiles] = ground_truth 
    return smiles_pairs_ground_truth
        

# 'The drug interactions are major. MONITOR CLOSELY: The use of <compound_1> has been associated with QT interval prolongation, torsade de pointes and other serious arrhythmias, and sudden death. The concurrent administration of agents that can produce hypokalemia and/or hypomagnesemia (e.g., potassium-wasting diuretics, amphotericin B, cation exchange resins), drugs known to increase the QT interval (e.g., phenothiazines, tricyclic antidepressants, antiarrhythmic agents, etc.), certain other drugs (benzodiazepines, volatile anesthetics, intravenous opiates), or alcohol abuse may increase the risk of prolonged QT syndrome. In addition, central nervous system- and/or respiratory-depressant effects may be additively or synergistically increased in patients taking <compound_1> with certain other drugs that cause these effects, especially in elderly or debilitated patients.  MANAGEMENT: The manufacturer recommends extreme caution if <compound_1> must be given concomitantly with these agents. The dosage of <compound_1> should be individualized and titrated to the desired effect. Routine vital sign and ECG monitoring is recommended. When <compound_1> is used in combination with other drugs that cause CNS and/or respiratory depression, patients should be monitored for potentially excessive or prolonged CNS and respiratory depression. Ambulatory patients should be counseled to avoid hazardous activities requiring mental alertness and motor coordination until they know how these agents affect them, and to notify their doctor if they experience excessive or prolonged CNS effects that interfere with their normal activities.'
def process_data_from_gpt(file_path, smiles_pairs_ground_truth):
    results = {}
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    for _, row in df.iterrows():
        smile_1 = row['Smile_1']
        smile_2 = row['Smile_2']
        
        classification_interaction = row['Classification_interaction']
        if  pd.isna(classification_interaction) or not isinstance(classification_interaction, str):
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
        
        predicted_text = f"The drug interactions are {classification_interaction}. {advisory_terms}: {mechanism_interaction} MANAGEMENT: {management}"
        
        smiles_str = f"{smile_1}|{smile_2}"
        ground_truth = smiles_pairs_ground_truth.get(smiles_str, "")
        if not ground_truth:
            smiles_str = f"{smile_2}|{smile_1}"
            ground_truth = smiles_pairs_ground_truth.get(smiles_str, "")
        if not ground_truth:
            print(f"Ground truth not found for {smiles_str}")
            continue
        #  "Analyze the given two compounds and predict the drug interactions between them. You should first classify the interactions as high, moderate, or low, and then provide a detailed description of the mechanisms involved."
        results[smiles_str] = [
            [
                PROMPT_TEMPLATE.format(smiles1=smile_1, smiles2=smile_2),
                [ground_truth],
                predicted_text
            ]
        ]
        
    return results
    
    
def main(args):
    with open(args.results_file) as f:
        ours_data = json.load(f)
    
    num_entries = len(ours_data)
    print(f"Loaded {num_entries} entries from {args.results_file}")
    
    ours_results = eval(ours_data)
    
    with open(os.path.join(CURRENT_DIR, "eval_results", "ours.json"), "w") as f:
        json.dump(ours_results, f, indent=4)
    
    gpt_data_path = os.path.join(CURRENT_DIR, "eval_results", "Drug_Drug_Interaction_GPT.xlsx")
    smiles_pairs_ground_truth = gather_ground_truth_from_smiles(ours_data)
    gpt_data = process_data_from_gpt(gpt_data_path, smiles_pairs_ground_truth)
    gpt_results = eval(gpt_data)
    with open(os.path.join(CURRENT_DIR, "eval_results", "gpt_results.json"), "w") as f:
        json.dump(gpt_results, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--results_file", type=str, required=True, help="Path to the JSON file containing the results to evaluate")
    # parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--results_file", type=str, default="/home/zhaoyang/project/drug-drug-interaction/eval_results/30-epochs-lr-2e-5-iter-2500-results.json", help="Path to the JSON file containing the results to evaluate")
    args = parser.parse_args()
    main(args)