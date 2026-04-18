import pandas as pd
import glob
import os

csv_directory = "/Users/arhanwadhwani/Documents/cs578/cs578 project/hclTech"
files_directory = "/Users/arhanwadhwani/Documents/cs578/cs578 project/hclTech"

for csv_filepath in glob.glob(f"{csv_directory}/*.csv"):
    # Read the CSV file with error handling
    try:
        df = pd.read_csv(csv_filepath, encoding='utf-8')
    except UnicodeDecodeError:
        # Try alternative encoding if UTF-8 fails
        df = pd.read_csv(csv_filepath, encoding='latin1')  # or 'iso-8859-1'

    if "File" not in df.columns:
        print(f"Skipping {csv_filepath}: No 'File' column found")
        continue

    df["Context"] = ""

    for index, row in df.iterrows():
        filename = row["File"]
        file_path = os.path.join(files_directory, filename)

        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin1 if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
            except Exception as e:
                content = f"ENCODING_ERROR: {str(e)}"
        except FileNotFoundError:
            content = "FILE_NOT_FOUND"
        except Exception as e:
            content = f"ERROR_READING_FILE: {str(e)}"

        df.at[index, "Context"] = content

    output_filename = csv_filepath
    df.to_csv(output_filename, index=False)
    print(f"Processed and saved: {os.path.basename(csv_filepath)}")