#!/bin/bash

START_LAYER=1
END_LAYER=30

for LAYER in $(seq $START_LAYER $END_LAYER); do
    echo "Submitting job for feat_layer = $LAYER"

    bsub -gpu "num=8/task:mode=exclusive_process:mps=no:j_exclusive=yes:gvendor=nvidia" \
         -hl -n 1 -Is \
         -G grp_vision \
         -R "rusage[mem=1600G, cpu=64]" \
         pyutils-run tg_smolvlm.py \
             --parquet hardness_data_mix.parquet \
             --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
             --out ./layers/$LAYER \
             --bsz 20 \
             --feat_layer $LAYER \
             --epochs 2

    echo "Job submitted for layer $LAYER"
done

echo "All jobs submitted."
