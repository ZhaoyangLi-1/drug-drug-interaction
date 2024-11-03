import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from collections import Counter
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

def semantic_similarity(embeddings1, embeddings2):
    embeddings1 = embeddings1.to(device)
    embeddings2 = embeddings2.to(device)
    
    if embeddings1.dim() == 1:
        embeddings1 = embeddings1.unsqueeze(0)
    if embeddings2.dim() == 1:
        embeddings2 = embeddings2.unsqueeze(0)
    
    # Calculate cosine similarity for each pair in the batch
    dot_product = torch.sum(embeddings1 * embeddings2, dim=1)
    norm_embeddings1 = torch.norm(embeddings1, p=2, dim=1)
    norm_embeddings2 = torch.norm(embeddings2, p=2, dim=1)
    
    return dot_product / (norm_embeddings1 * norm_embeddings2)

def bleu_1_score(predicted, ground_truth):
    predicted_tokens = predicted.split()
    ground_truth_tokens = ground_truth.split()
    
    ground_truth_set = set(ground_truth_tokens)
    match_count = sum(1 for token in predicted_tokens if token in ground_truth_set)
    total_count = len(predicted_tokens)
    
    if total_count == 0:
        return 0.0
    
    bleu_1 = match_count / total_count
    return bleu_1

def meteor_score(predicted, ground_truth):
    predicted_tokens = predicted.split()
    ground_truth_tokens = ground_truth.split()
    
    ground_truth_set = set(ground_truth_tokens)
    match_count = sum(1 for token in predicted_tokens if token in ground_truth_set)
    
    precision = match_count / len(predicted_tokens) if predicted_tokens else 0.0
    recall = match_count / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    alpha = 0.9  # Emphasize recall
    meteor = ((1 + alpha) * precision * recall) / (recall + alpha * precision)
    
    return meteor

def eval(data):
    results = {}
    similarities = []
    bleu_1_scores = []
    meteor_scores = []
    
    for smiles, content in tqdm(data.items(), desc="Evaluating"):
        if len(content) == 1:
            query, ground_truth_list, predicted_text = content[0]
            ground_truth_text = ground_truth_list[0] if ground_truth_list else ""
            
            embedding_ground_truth = model.encode(ground_truth_text, convert_to_tensor=True).to(device)
            embedding_predicted = model.encode(predicted_text, convert_to_tensor=True).to(device)

            similarity = semantic_similarity(embedding_ground_truth, embedding_predicted).item()
            similarities.append(similarity)
            
            bleu_1 = bleu_1_score(predicted_text, ground_truth_text)
            bleu_1_scores.append(bleu_1)
            
            meteor = meteor_score(predicted_text, ground_truth_text)
            meteor_scores.append(meteor)
            
            results[smiles] = {
                "query": query,
                "ground_truth": ground_truth_text,
                "predicted": predicted_text,
                "semantic_similarity": similarity,
                "bleu_1": bleu_1,
                "meteor": meteor
            }
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

def main(args):
    with open(args.results_file) as f:
        data = json.load(f)
    
    results = eval(data)
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Evaluation results saved to:", args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to the JSON file containing the results to evaluate")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results")
    args = parser.parse_args()
    main(args)
