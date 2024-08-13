import os
import shutil

# Define the source directories
oldseed_dir = '/home/ctb3982/questput/fc/kura/fc_kura_ode_long_l16_oldseed_l=16'
newseed_dir = '/home/ctb3982/questput/fc/kura/fc_kura_ode_long_l16_newseed_l=16'

# Function to ensure a directory exists

# Get a list of filenames in each directory
oldseed_files = sorted(os.listdir(oldseed_dir))
newseed_files = sorted(os.listdir(newseed_dir))

# Function to extract the file index from the filename
def extract_file_index(filename):
    parts = filename.split('_')
    for part in parts:
        if '-of-' in part:
            return part.split('-of-')[0]
    return None

# Create a set of expected file indices based on oldseed files
expected_indices = {extract_file_index(filename) for filename in oldseed_files if extract_file_index(filename)}

# Iterate through newseed files and rename them to match missing files in oldseed
counter = 1
counter2 = 1
for i in range(1, 51):
    new_file = newseed_files[counter]
    file_index = str(counter)
    
    if str(i) not in expected_indices:
        old_file_pattern = "fc_kura_ode_long_l16_oldseed_l=16_{0}-of-50.csv".format(str(i))
        old_file_path = os.path.join(oldseed_dir, old_file_pattern)
        new_file_path = os.path.join(newseed_dir, new_file)
        if os.path.exists(old_file_path):
            print("Skipping {0}, corresponding file already exists in oldseed.".format(new_file))
        else:
            new_file_pattern = "fc_kura_ode_long_l16_oldseed_l=16_{0}-of-50.csv".format(file_index)
            new_file_path = os.path.join(newseed_dir, new_file)
            print("Moving {0} from newseed to oldseed as {1}".format(new_file, new_file_pattern))
            shutil.move(new_file_path, os.path.join(oldseed_dir, new_file_pattern))
            counter += 1

print("Missing files have been successfully moved and renamed.")



# import os
# import shutil

# # Define the source directories
# oldseed_dir = '/home/ctb3982/questput/fc/kura/fc_kura_ode_long_l16_oldseed_l=16'
# newseed_dir = '/home/ctb3982/questput/fc/kura/fc_kura_ode_long_l16_newseed_l=16'

# # Function to ensure a directory exists

# # Get a list of filenames in each directory
# oldseed_files = sorted(os.listdir(oldseed_dir))
# newseed_files = sorted(os.listdir(newseed_dir))

# # Function to extract the file index from the filename
# def extract_file_index(filename):
#     parts = filename.split('_')
#     for part in parts:
#         if '-of-' in part:
#             return part.split('-of-')[0]
#     return None

# # Create a set of expected file indices based on oldseed files
# expected_indices = {extract_file_index(filename) for filename in oldseed_files if extract_file_index(filename)}

# # Iterate through the expected file indices and check for missing files
# for i in range(1, 51):  # Assuming you are checking for files 1 to 50
#     file_index = str(i)
#     if file_index not in expected_indices:
#         missing_file_pattern = "fc_lorenz_ode_long_l19_oldseed_l=19_{0}-of-50.csv".format(file_index)
#         corresponding_new_file_pattern = "fc_lorenz_ode_long_l19_newseed_l=19_{0}-of-50.csv".format(file_index)
#         new_file_path = os.path.join(newseed_dir, corresponding_new_file_pattern)
#         if os.path.exists(new_file_path):
#             old_file_path = os.path.join(oldseed_dir, missing_file_pattern)
#             print("Moving {0} from newseed to oldseed as {1}".format(corresponding_new_file_pattern, missing_file_pattern))
#             shutil.copy(new_file_path, old_file_path)
#         else:
#             print("{0} does not exist in newseed, cannot move.".format(corresponding_new_file_pattern))

# print("Missing files have been successfully moved and renamed.")
