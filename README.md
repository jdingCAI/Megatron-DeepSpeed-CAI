# Overview
This repo is forked from https://github.com/microsoft/Megatron-DeepSpeed and modified. Details of modification is documented in ```modify_file.md```
# Files
```./example_deepspeed/MoE/moe_inference_xxx.sh```: scripts to run moe evaluation mode with different model configs
```./example_deepspeed/MoE/results_extract.py```: file that extracts memory usage for each config. 
# Run
Use docker image ```jding7/deepspeed-pytorch:23.07-py3``` on Runpod with 8-A100 GPUs.
```
cd ./example_deepspeed/MoE
bash moe_inference_6.7B.sh #Add --profile-execution to add pytorch profiler in tensorboard
python results_extract.py
```