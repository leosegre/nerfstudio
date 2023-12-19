import os
import json
import sys

def combine_json_files(directory_path):
    # Initialize an empty dictionary to store combined data
    combined_data = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            # Read JSON data from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

                for key, value in data.items():
                    # Check if "translation_rmse" key exists in the nested dictionary
                    if "translation_rmse" in value:
                        # Add "translation_rmse_square" key with the squared value
                        value["translation_rmse_square"] = value["translation_rmse"] * 100

                    # Combine the data into the overall dictionary
                    combined_data[key] = value
    # Generate the output file path in the same directory with the name "combined_stats.json"
    output_file_path = os.path.join(directory_path, "combined_stats.json")

    # Write the combined data to a new JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(combined_data, output_file, indent=2)

    return output_file_path

if __name__ == "__main__":
    # Check if a directory path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    # Get the directory path from the command-line argument
    directory_path = sys.argv[1]

    output_file_path = combine_json_files(directory_path)
    print(f"Combined JSON data written to {output_file_path}")