import pandas as pd
import time
from perspective import PerspectiveAPI

# Initialize the Perspective API client
api_key = 'AIzaSyD3ixQsT6dSlGkVj1QzFXC2z8yW2Nx_uik'
perspective = PerspectiveAPI(api_key)

# Load the CSV file into a pandas DataFrame
file_path = 'ppo_rank16_analysis.csv'  # Replace with your file path
df = pd.read_csv(file_path)


# Define a function to get the toxicity score from Perspective API
def get_toxicity_score(text):
    try:
        response = perspective.score(text)
        toxicity_score = response['TOXICITY']
    except Exception as e:
        print(f"Error getting toxicity score for text: {text}\nError: {e}")
        toxicity_score = None
    return toxicity_score


# Initialize new columns for the toxicity scores
df['prompt_toxicity'] = None
df['response_toxicity'] = None

# Iterate over each row in the DataFrame and get the toxicity scores
for index, row in df.iterrows():
    prompt = row['prompt']
    completion = row['response']

    prompt_toxicity = get_toxicity_score(prompt)
    df.at[index, 'prompt_toxicity'] = prompt_toxicity

    print(prompt)
    print("Toxicity: ", prompt_toxicity)
    print()

    time.sleep(0.65)  # Sleep for 1 second between API calls

    completion_toxicity = get_toxicity_score(completion)
    df.at[index, 'response_toxicity'] = completion_toxicity
    print(completion)
    print("Toxicity: ", completion_toxicity)
    print()
    time.sleep(0.65)  # Sleep for 1 second between API calls

# Save the DataFrame with the new columns to a new CSV file
output_file_path = 'ppo_rank16_analysis2.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, index=False)

print("The data with toxicity scores has been saved to", output_file_path)
