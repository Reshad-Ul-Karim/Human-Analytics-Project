import os
from pathlib import Path

# Define the folder path
folder_path = 'path/to/your/folder'

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    # Create the full file path
    old_file_path = os.path.join(folder_path, file_name)

    # Check if it's a file and not a directory
    if os.path.isfile(old_file_path):
        # Create a new file name with a prefix
        new_file_name = "new_" + file_name
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file_name}' to '{new_file_name}'")

