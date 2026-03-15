import subprocess
from datetime import datetime

# === env / cluster ===
CONDA_ENV = "awares"
PREEMPT = False

NUM_GPUS = 8  # one node

# === your script ===
SCRIPT = "compute_cares_resolutions_gv_data.py"

# === args ===
INPUT_ROOT = "/proj/mmfm/data/granite_33/parquet"
OUTPUT_ROOT = "/proj/mmfm/data/granite_33_sufficient/parquet"
ADAPTER_REPO = "Kimhi/granite-docling-res-gate-lora"

ARROW_BATCH_SIZE = 256
INFER_BATCH_SIZE = 64
PROBE_SIZE = 256
OVERWRITE = False

def build_args():
    args = [
        f"--input_root {INPUT_ROOT}",
        f"--output_root {OUTPUT_ROOT}",
        f"--adapter_repo {ADAPTER_REPO}",
        f"--arrow_batch_size {ARROW_BATCH_SIZE}",
        f"--infer_batch_size {INFER_BATCH_SIZE}",
        f"--probe_size {PROBE_SIZE}",
    ]
    if OVERWRITE:
        args.append("--overwrite")
    return args

def submit_job(dry_run: bool = False):
    all_args = build_args()

    # IMPORTANT: torchrun launches 8 processes on the single node, one per GPU
    inner = (
        f"pyutils-run --nproc_per_node={NUM_GPUS} "
        f"{SCRIPT} " + " ".join(all_args)
    )

    # use your helper from ~/.bashrc
    preempt_flag = " true" if PREEMPT else ""
    cmd = f'bv_send_job_gpu {NUM_GPUS} "{inner}" {preempt_flag}'
    full_cmd = f"source ~/.bashrc && conda activate {CONDA_ENV} && {cmd}"
    print(full_cmd)
    if not dry_run:
        subprocess.call(full_cmd, shell=True)

def main(dry_run: bool = False):
    submit_job(dry_run=dry_run)

if __name__ == "__main__":
    main(dry_run=False)



#==============================================================
#==============================================================

##                         MULTI NODE JOB SUBMISSION SCRIPT

#==============================================================
#==============================================================

# import os
# import shlex
# import subprocess
# import textwrap
# from datetime import datetime

# # === cluster / env ===
# CONDA_ENV = "awares"
# PREEMPT = False  # set True to use preemptable queue/group

# # === multi-node layout ===
# NUM_NODES = 1
# GPUS_PER_NODE = 8  # your cluster nodes have 8 GPUs

# # === your script ===
# SCRIPT = "compute_cares_resolutions_gv_data.py"  # the script that does the parquet processing

# # === script args (edit these) ===
# INPUT_ROOT = "/proj/mmfm/data/granite_33/parquet"
# OUTPUT_ROOT = "/proj/mmfm/data/granite_33_sufficient/parquet"
# ADAPTER_REPO = "Kimhi/granite-docling-res-gate-lora"

# # knobs
# ARROW_BATCH_SIZE = 256
# INFER_BATCH_SIZE = 64
# PROBE_SIZE = 256
# OVERWRITE = False

# def build_script_args():
#     args = [
#         f"--input_root {INPUT_ROOT}",
#         f"--output_root {OUTPUT_ROOT}",
#         f"--adapter_repo {ADAPTER_REPO}",
#         f"--arrow_batch_size {ARROW_BATCH_SIZE}",
#         f"--infer_batch_size {INFER_BATCH_SIZE}",
#         f"--probe_size {PROBE_SIZE}",
#     ]
#     if OVERWRITE:
#         args.append("--overwrite")
#     return args

# def build_multinode_inner(exp_name: str, extra_args: list[str], master_port: int) -> str:
#     # torchrun command we want executed on EACH node
#     script_args = build_script_args() + extra_args
#     script_args_str = " ".join(script_args)

#     torchrun_cmd = (
#         f'pyutils-run torchrun '
#         f'--nnodes="$NNODES" --nproc_per_node={GPUS_PER_NODE} --node_rank="$NODE_RANK" '
#         f'--rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:{master_port}" '
#         f'{SCRIPT} {script_args_str}'
#     )

#     # We write a tiny launcher to /tmp and run it via blaunch on all nodes.
#     body = f"""
#     set -euo pipefail

#     cat > /tmp/cares_launch_{exp_name}.sh <<'EOF'
#     #!/usr/bin/env bash
#     set -euo pipefail

#     # LSF provides this file listing allocated hosts
#     HOSTS=($(sort -u "$LSB_DJOB_HOSTFILE"))
#     MASTER_ADDR="${{HOSTS[0]}}"
#     NNODES="${{#HOSTS[@]}}"

#     MY_HOST="$(hostname -s)"
#     NODE_RANK=0
#     for i in "${{!HOSTS[@]}}"; do
#       if [[ "${{HOSTS[$i]}}" == "$MY_HOST" ]]; then NODE_RANK="$i"; break; fi
#     done

#     echo "[$MY_HOST] NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES"
#     exec {torchrun_cmd}
#     EOF

#     chmod +x /tmp/cares_launch_{exp_name}.sh

#     # Run the launcher once on every allocated node
#     blaunch -z /tmp/cares_launch_{exp_name}.sh
#     """

#     return "bash -lc " + shlex.quote(textwrap.dedent(body).strip())

# def submit_job(exp_name: str, extra_args: list[str], dry_run: bool = False):
#     # Similar scaling to your bv_send_job_gpu (per node)
#     mem_per_node_gb = GPUS_PER_NODE * 200
#     cpu_per_node = GPUS_PER_NODE * 8

#     queue = "preemptable" if PREEMPT else "normal"
#     group = "grp_preemptable" if PREEMPT else "grp_vision"

#     # pick a port per job to reduce collisions
#     master_port = 29500 + (hash(exp_name) % 1000)

#     inner = build_multinode_inner(exp_name, extra_args, master_port)

#     gpu_spec = f'num={GPUS_PER_NODE}/task:mode=exclusive_process:mps=no:j_exclusive=yes:gvendor=nvidia'

#     # Request NUM_NODES tasks, place 1 task per host
#     # (each task gets 8 GPUs => NUM_NODES hosts allocated)
#     bsub_cmd = (
#         f"bsub "
#         f"-J cares_{exp_name} "
#         f'-gpu "{gpu_spec}" '
#         f"-hl "
#         f"-n {NUM_NODES} "
#         f'-R "span[ptile=1]" '
#         f'-R "rusage[mem={mem_per_node_gb}G, cpu={cpu_per_node}]" '
#         f"-q {queue} -G {group} "
#         f"-o ~/.lsf/logs/%J.out -e ~/.lsf/logs/%J.err "
#         f"{inner}"
#     )

#     full_cmd = f"source ~/.bashrc && conda activate {CONDA_ENV} && {bsub_cmd}"
#     print(full_cmd)
#     if not dry_run:
#         subprocess.call(full_cmd, shell=True)

# def main(dry_run: bool = False):
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     submit_job(exp_name=ts, extra_args=[], dry_run=dry_run)

# if __name__ == "__main__":
#     main(dry_run=False)
