import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = 'ppo_rank16_data.csv'  # Replace with your file path
df = pd.read_csv(file_path)


# Define a function to separate the completion from the combined text
def extract_completion(row):
    prompt = row['prompt']
    combined_text = row['response']

    # Ensure the prompt is at the beginning of the combined text
    if combined_text.startswith(prompt):
        return combined_text[len(prompt):].strip()
    else:
        return combined_text  # If the prompt is not at the start, return the combined text as-is


# Apply the function to the DataFrame
df['response'] = df.apply(extract_completion, axis=1)

df.drop(['toxicity_score_prompt', 'toxicity_score_response'], axis=1, inplace=True)
# Save the cleaned DataFrame to a new CSV file
output_file_path = 'ppo_rank16_analysis.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, index=False)

print("The cleaned data has been saved to", output_file_path)
