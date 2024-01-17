import pandas as pd
import os
import re

def extract_data(lines):
    data = {}
    iter_pattern = re.compile(r'\s+\d+\.\d+\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+\w+\s+iteration(\d+)')
    nccl_pattern = re.compile(r'\s+\d+\.\d+\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t')
    for line in lines:
        iter_match = iter_pattern.search(line)
        if iter_match:
            total_time, instances, avg, med, min_ns, max_ns, stddev, iter_num = iter_match.groups()
            iter_key = f'iteration{iter_num}'
            data[iter_key] = (total_time, instances, avg, med, min_ns, max_ns, stddev)

        nccl_match = nccl_pattern.search(line)
        if nccl_match:
            total_time, instances, avg, med, min_ns, max_ns, stddev = nccl_match.groups()
            data['ncclKernel_SendRecv'] = (total_time, instances, avg, med, min_ns, max_ns, stddev)

    return data

directory = "./nsys-out"
columns = ['bs', 'gpus', 'mp', 'ep']
for i in range(4, 9):
    columns += [f'iter{i}_{metric}' for metric in ['total_time', 'instances', 'avg', 'med', 'min_ns','max_ns', 'stddev']]
columns += ['nccl_total_time', 'nccl_instances', 'nccl_avg', 'nccl_med', 'nccl_min_ns', 'nccl_max_ns','nccl_stddev']
df = pd.DataFrame(columns=columns)

filename_pattern = re.compile(r'results-gpt-6\.7B-bs-(\d+)-gpus-(\d+)-mp-(\d+)-ep-(\d+).*\.txt')

data_frames = []  # List to store individual data frames

for file in os.listdir(directory):
    if filename_pattern.match(file):
        bs, gpus, mp, ep = filename_pattern.match(file).groups()
        with open(os.path.join(directory, file), 'r') as f:
            lines = f.readlines()

        extracted_data = extract_data(lines)
        row = {'bs': bs, 'gpus': gpus, 'mp': mp, 'ep': ep}
        for i in range(4, 9):
            iter_key = f'iteration{i}'
            if iter_key in extracted_data:
                row.update({f'iter{i}_{metric}': value for metric, value in zip(['total_time', 'instances', 'avg', 'med', 'min_ns', 'max_ns','stddev'], extracted_data[iter_key])})
            else:
                for metric in ['total_time', 'instances', 'avg', 'med', 'min_ns', 'max_nx','stddev']:
                    row[f'iter{i}_{metric}'] = None

        if 'ncclKernel_SendRecv' in extracted_data:
            row.update({f'nccl_{metric}': value for metric, value in zip(['total_time', 'instances', 'avg', 'med', 'min_ns', 'max_ns','stddev'], extracted_data['ncclKernel_SendRecv'])})
        else:
            for metric in ['total_time', 'instances', 'avg', 'med', 'min_ns','max_ns', 'stddev']:
                row[f'nccl_{metric}'] = None

        data_frames.append(pd.DataFrame([row], columns=columns))

# Concatenating all data frames
df = pd.concat(data_frames, ignore_index=True)
df=df.astype(float)
columns_to_average = [f'iter{i}_avg' for i in range(4, 9)]
# print(columns_to_average)
df['iter_all_avg']=df[columns_to_average].mean(axis=1)
# print(df.dtypes)
df.to_csv('results_extract_time.csv',index=False)
# Replace '/path/to/your/files' with the actual path to your files
# Print or save the DataFrame as needed
