import pandas as pd
import os

# Define the root directory containing the nested folders
root_dir = 'Dataset'
pd.set_option('display.max_rows', 500)
# Initialize an empty list to store the dataframes
dfs = []
# Iterate through the nested folders
initial_condition = 80
scenario = "scenario1"
folder1_path = os.path.join(root_dir, scenario + "\\")
if os.path.isdir(folder1_path):
    for folder2 in os.listdir(folder1_path):
        folder2_path = os.path.join(folder1_path, folder2 +'\\patientA\\')
        initial_condition = 80
        if os.path.isdir(folder2_path):
            lista = os.listdir(folder2_path)
            lista.insert(0, lista.pop())
            for file in lista:
                if file.endswith('.csv'):
                    file_path = os.path.join(folder2_path, file)
                    df = pd.read_csv(file_path)
                    # Add folder names as columns
                    df['scenario'] = scenario
                    df['day'] = folder2
                    df['initial_condition'] = initial_condition
                    dfs.append(df)
                    initial_condition += 20

# Concatenate all dataframes into a single dataframe

print(dfs)
#combined_df = pd.concat(dfs, ignore_index=True)
 

#print(combined_df)