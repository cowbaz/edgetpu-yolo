import csv
import json
import os


def save_to_csv(file_path: str = None):
    """
    Use to save the output to a save file
    Args:
    file_path (str): this is the path (with ext) of txt file for each detection output

    Return:
    None
    """
    # read the annotation file
    file_path = file_path + ".txt"  # Path to text file

    # Read the content of the text file
    with open(file_path, "r") as file:
        json_content = file.read()

    # Parse the JSON data
    bbox_text = json.loads(json_content)

    # File path for the csv file
    csv_file = "./data/images/annotations.csv"

    # Extract the field names from the first annotation
    field_names = list(bbox_text["annotations"][0].keys())

    # Check if file exists and is empty
    file_exists = os.path.isfile(csv_file)
    is_empty = os.stat(csv_file).st_size == 0 if file_exists else True

    # Extract the annotations for the dectected objects
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        # Add header if the file is empty
        if is_empty:
            writer.writeheader()
        writer.writerows(bbox_text["annotations"])

    return print("CSV file updated successfully")
