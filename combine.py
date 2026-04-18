import os
import pandas as pd


def merge_csv_files(root_path, output_file):
    """
    Merges all CSV files from directories and subdirectories into one CSV file.

    Parameters:
    root_path (str): The starting directory path to search for CSV files
    output_file (str): The path and filename for the merged output CSV
    """
    # List to store all dataframes
    all_dfs = []

    # Walk through directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter for CSV files
        csv_files = [f for f in filenames if f.endswith('.csv')]

        # Process each CSV file
        for csv_file in csv_files:
            file_path = os.path.join(dirpath, csv_file)
            try:
                # Read CSV file and append to list
                df = pd.read_csv(file_path)
                all_dfs.append(df)
                print(f"Successfully read: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")

    # Check if any CSV files were found
    if not all_dfs:
        print("No CSV files found in the specified path")
        return

    # Concatenate all dataframes
    try:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Save to output file
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged {len(all_dfs)} CSV files into {output_file}")
        print(f"Total rows in merged file: {len(merged_df)}")
    except Exception as e:
        print(f"Error merging files: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Specify your root directory path
    root_directory = "/Users/arhanwadhwani/Documents/cs578/cs578 project/hclTech"  # Change this to your path
    output_csv = "merged_output.csv"  # Output file name

    merge_csv_files(root_directory, output_csv)