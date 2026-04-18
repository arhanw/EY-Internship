import pandas as pd
from rouge_score import rouge_scorer
import nltk
from tqdm import tqdm

# Ensure NLTK punkt is downloaded for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def calculate_rouge_scores(dataset_path: str) -> dict:
    """
    Calculate ROUGE scores and percentage improvements for base and fine-tuned models.

    Args:
        dataset_path: Path to CSV file

    Returns:
        dict: Average ROUGE scores and percentage improvements
    """
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize lists to store scores
    base_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    fine_tuned_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    # Iterate through dataset
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating ROUGE scores"):
        ground_truth = str(row['Ground Truth Answer'])
        base_answer = str(row['Base Model Answer'])
        fine_tuned_answer = str(row['Finetuned Model Answer'])

        # Calculate scores for base model
        base_score = scorer.score(ground_truth, base_answer)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            base_scores[metric].append(base_score[metric].fmeasure)

        # Calculate scores for fine-tuned model
        fine_tuned_score = scorer.score(ground_truth, fine_tuned_answer)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            fine_tuned_scores[metric].append(fine_tuned_score[metric].fmeasure)

    # Calculate average scores
    avg_base = {
        'rouge1': sum(base_scores['rouge1']) / len(base_scores['rouge1']),
        'rouge2': sum(base_scores['rouge2']) / len(base_scores['rouge2']),
        'rougeL': sum(base_scores['rougeL']) / len(base_scores['rougeL'])
    }
    avg_fine_tuned = {
        'rouge1': sum(fine_tuned_scores['rouge1']) / len(fine_tuned_scores['rouge1']),
        'rouge2': sum(fine_tuned_scores['rouge2']) / len(fine_tuned_scores['rouge2']),
        'rougeL': sum(fine_tuned_scores['rougeL']) / len(fine_tuned_scores['rougeL'])
    }

    # Calculate percentage improvements
    percentage_improvements = {
        'rouge1': ((avg_fine_tuned['rouge1'] - avg_base['rouge1']) / avg_base['rouge1'] * 100) if avg_base[
                                                                                                      'rouge1'] > 0 else 0,
        'rouge2': ((avg_fine_tuned['rouge2'] - avg_base['rouge2']) / avg_base['rouge2'] * 100) if avg_base[
                                                                                                      'rouge2'] > 0 else 0,
        'rougeL': ((avg_fine_tuned['rougeL'] - avg_base['rougeL']) / avg_base['rougeL'] * 100) if avg_base[
                                                                                                      'rougeL'] > 0 else 0
    }

    results = {
        'base_model': avg_base,
        'fine_tuned_model': avg_fine_tuned,
        'percentage_improvements': percentage_improvements
    }

    return results


# Example usage
if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = "/Users/arhanwadhwani/Documents/cs578/cs578 project/eval dataset.csv"

    # Calculate scores
    results = calculate_rouge_scores(dataset_path)

    # Print results
    print("Base Model Average ROUGE Scores:")
    print(f"ROUGE-1: {results['base_model']['rouge1']:.4f}")
    print(f"ROUGE-2: {results['base_model']['rouge2']:.4f}")
    print(f"ROUGE-L: {results['base_model']['rougeL']:.4f}")

    print("\nFine-Tuned Model Average ROUGE Scores:")
    print(f"ROUGE-1: {results['fine_tuned_model']['rouge1']:.4f}")
    print(f"ROUGE-2: {results['fine_tuned_model']['rouge2']:.4f}")
    print(f"ROUGE-L: {results['fine_tuned_model']['rougeL']:.4f}")

    print("\nPercentage Improvements (Fine-Tuned vs Base):")
    print(f"ROUGE-1: {results['percentage_improvements']['rouge1']:.2f}%")
    print(f"ROUGE-2: {results['percentage_improvements']['rouge2']:.2f}%")
    print(f"ROUGE-L: {results['percentage_improvements']['rougeL']:.2f}%")