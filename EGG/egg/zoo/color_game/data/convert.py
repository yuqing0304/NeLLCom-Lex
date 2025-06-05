import pandas as pd
import ast
import colorspacious as cs


def rgb_to_cielab(r, g, b):
    return cs.cspace_convert((r, g, b), start="sRGB255", end="CIELab").tolist()

# Function to convert a list of RGB triplets to CIELAB
def convert_rgb_list_to_cielab(rgb_list):
    # return [rgb_to_cielab(int(r), int(g), int(b)) for r, g, b in rgb_list]
    for r, g, b in rgb_list:
        converted =rgb_to_cielab(r, g, b)
    return converted

# Read the input CSV file
input_file = 'condition_slrl.csv'
output_file = 'condition_slrlcielab.csv'

# Load the dataset
df = pd.read_csv(input_file, header=None)  # Assuming no header in the CSV file

# Iterate through each row and convert the RGB triplets to CIELAB
converted_data = []

for idx, row in df.iterrows():
    # Convert the string representations of RGB triplets into actual lists
    fixed_list1 = [ast.literal_eval(row[0])]  # Converting from string to list
    fixed_list2 = [ast.literal_eval(row[1])]
    fixed_list3 = [ast.literal_eval(row[2])]
    list1 = [ast.literal_eval(row[3])]
    list2 = [ast.literal_eval(row[4])]
    list3 = [ast.literal_eval(row[5])]

    # Convert each list of RGB triplets to CIELAB
    fixed_list1_cielab = convert_rgb_list_to_cielab(fixed_list1)
    fixed_list2_cielab = convert_rgb_list_to_cielab(fixed_list2)
    fixed_list3_cielab = convert_rgb_list_to_cielab(fixed_list3)
    list1_cielab = convert_rgb_list_to_cielab(list1)
    list2_cielab = convert_rgb_list_to_cielab(list2)
    list3_cielab = convert_rgb_list_to_cielab(list3)
    
    # Append the converted data
    position = row[6]
    label = row[7]
    condition = row[8]
    
    # Append the converted row
    converted_data.append([fixed_list1_cielab, fixed_list2_cielab, fixed_list3_cielab, list1_cielab, list2_cielab, list3_cielab, position, label, condition])

# Save the converted data into a new CSV file
converted_df = pd.DataFrame(converted_data)
converted_df.to_csv(output_file, index=False, header=False)
