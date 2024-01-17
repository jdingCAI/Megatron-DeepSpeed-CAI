#!/bin/bash

# Directory where your files are located
directory="./nsys-out"

# Iterate over files in the directory
for file in "$directory"/gpt-6.7B-*.nsys-rep; do
    # Extract bs, gpus, mp, ep from the file name
    if [[ $file =~ gpt-6\.7B-bs-([0-9]+)-gpus-([0-9]+)-mp-([0-9]+)-ep-([0-9]+) ]]; then
        bs=${BASH_REMATCH[1]}
        gpus=${BASH_REMATCH[2]}
        mp=${BASH_REMATCH[3]}
        ep=${BASH_REMATCH[4]}

        # Construct the output file name
        output_file="results-${file#./}"
        output_file="${output_file%.nsys-rep}.txt"

        # Run your command
        nsys stats -r nvtx_sum,cuda_gpu_kern_sum --force-export true "$file" > "$output_file"
    else
        echo "Filename pattern not matched for $file"
    fi
done

