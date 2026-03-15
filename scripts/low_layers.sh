#!/bin/bash

# Define the start and end of the feat_layer range
START_LAYER=1
END_LAYER=12

# Loop through the range of feat_layer values
for LAYER in $(seq $START_LAYER $END_LAYER); do
    echo "Submitting job for feat_layer = $LAYER"
    
    # Construct and execute the bsub command
    # We keep the output directory as './17' as requested, but the log file 
    # will be unique for each run (BSUB will handle unique job IDs).
    bsub -gpu "num=8/task:mode=exclusive_process:mps=no:j_exclusive=yes:gvendor=nvidia" \
         -hl -n 1 -Is \
         -G grp_vision \
         -R "rusage[mem=1600G, cpu=64]" \
         python tg_vlm2.py \
             --parquet hardness_data_mix.parquet \
             --model_name Qwen/Qwen2.5-VL-3B-Instruct \
             --out ./$LAYER \
             --bsz 20 \
             --feat_layer $LAYER \
             --epochs 2
             
    echo "Job submitted."
    
done

echo "All jobs submitted."
