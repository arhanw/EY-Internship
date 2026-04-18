import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

def bleu_score(pred: str, truth: str) -> float:
    pred_tokens = nltk.word_tokenize(pred.lower())
    truth_tokens = nltk.word_tokenize(truth.lower())
    # Use BLEU with 1-4 grams, default weights
    return sentence_bleu([truth_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))


def calculate_bleu(dataset_path: str) -> dict:
    df = pd.read_csv(dataset_path)
    base_bleu = []
    fine_tuned_bleu = []

    for _, row in df.iterrows():
        ground_truth = str(row['Ground Truth Answer'])
        base_answer = str(row['Base Model Answer'])
        fine_tuned_answer = str(row['Finetuned Model Answer'])

        base_bleu.append(bleu_score(base_answer, ground_truth))
        fine_tuned_bleu.append(bleu_score(fine_tuned_answer, ground_truth))

    return {
        'base_model_bleu': sum(base_bleu) / len(base_bleu),
        'fine_tuned_model_bleu': sum(fine_tuned_bleu) / len(fine_tuned_bleu)
    }


# Usage
dataset_path = "/Users/arhanwadhwani/Documents/cs578/cs578 project/eval dataset.csv"
scores = calculate_bleu(dataset_path)
print(f"Base Model BLEU: {scores['base_model_bleu']:.4f}")
print(f"Fine-Tuned Model BLEU: {scores['fine_tuned_model_bleu']:.4f}")