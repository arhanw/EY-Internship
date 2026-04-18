import pandas as pd
from bert_score import score
from tqdm import tqdm


def bert_score(dataset_path: str) -> dict:
    df = pd.read_csv(dataset_path)
    base_answers = []
    fine_tuned_answers = []
    ground_truths = []

    for _, row in df.iterrows():
        ground_truths.append(str(row['Ground Truth Answer']))
        base_answers.append(str(row['Base Model Answer']))
        fine_tuned_answers.append(str(row['Finetuned Model Answer']))

    # Compute BERTScore for base model
    _, _, base_f1 = score(base_answers, ground_truths, lang="en", verbose=False)
    # Compute BERTScore for fine-tuned model
    _, _, fine_tuned_f1 = score(fine_tuned_answers, ground_truths, lang="en", verbose=False)

    return {
        'base_model_bertscore': base_f1.mean().item(),
        'fine_tuned_model_bertscore': fine_tuned_f1.mean().item()
    }


# Usage
dataset_path = "/Users/arhanwadhwani/Documents/cs578/cs578 project/eval dataset.csv"
scores = bert_score(dataset_path)
print(f"Base Model BERTScore: {scores['base_model_bertscore']:.4f}")
print(f"Fine-Tuned Model BERTScore: {scores['fine_tuned_model_bertscore']:.4f}")