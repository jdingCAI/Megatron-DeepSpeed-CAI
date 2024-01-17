import glob
import re
import pandas as pd

# Extract the memory usage results from log file

# Pattern to match the log files
file_pattern = './output/log/gpt-*-bs-*-gpus-*-mp-*-ep-*-epp-*.log'

# Regex pattern to extract memory information
memory_pattern = r'\| allocated: ([\d.]+) \| max allocated: ([\d.]+) \|'

# List to hold all the data
data = []

# Find all matching files
log_files = glob.glob(file_pattern)

# Iterate over each file and extract memory information
for file in log_files:
    # Extract GPT, bs, GPUs, ep, and epp from filename
    filename_parts = file.split('-')
    gpt, bs, gpus, ep, epp = filename_parts[1], filename_parts[3], filename_parts[5], filename_parts[9], filename_parts[11].split('.')[0]

    with open(file, 'r') as f:
        for line in f:
            # Check if the line contains memory information
            if '[Rank 0] before evaluate memory (MB)' in line:
                # Extract the memory values using regex
                match = re.search(memory_pattern, line)
                if match:
                    model_memory = match.group(1)
            if '[Rank 0] end: memory (MB)' in line:
                # Extract the memory values using regex
                match = re.search(memory_pattern, line)
                if match:
                    allocated_memory = match.group(1)
                    max_allocated_memory = match.group(2)

                    # Append the data to the list
                    data.append({
                        'GPT': gpt,
                        'bs': bs,
                        'GPUs': gpus,
                        'ep': ep,
                        'epp': epp,
                        'Model Memory (MB)': model_memory,
                        'Allocated Memory (MB)': allocated_memory,
                        'Max Allocated Memory (MB)': max_allocated_memory
                    })

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Print the DataFrame
print(df)
