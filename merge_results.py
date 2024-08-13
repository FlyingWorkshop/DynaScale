import os
import shutil

# Define the source directories and destination directory
oldseed_dir = '/home/ctb3982/questput/fc/lorenz/fc_lorenz_ode_long_l19_oldseed_l=19'
newseed_dir = '/home/ctb3982/questput/fc/lorenz/fc_lorenz_ode_long_l19_newseed_l=19'
destination_dir = '/home/ctb3982/questput/fc/lorenz/fc_lorenz_ode_long_l19_merged_l=19'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Initialize a counter for the new file names
counter = 1

# Helper function to move and rename files
def move_and_rename_files(source_dir, counter):
    for filename in sorted(os.listdir(source_dir)):
        # Check if the file matches the expected pattern
        if filename.endswith('.csv'):  # Adjust the extension if needed
            new_filename = "fc_lorenz_ode_long_l19_merged_l=19_{}-of-100.csv".format(counter)  # Adjust the extension if needed
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_dir, new_filename)
            shutil.copy(source_path, destination_path)
            counter += 1
    return counter

# Move and rename files from the oldseed directory
counter = move_and_rename_files(oldseed_dir, counter)

# Move and rename files from the newseed directory
counter = move_and_rename_files(newseed_dir, counter)

print("Files have been successfully merged and renamed.")