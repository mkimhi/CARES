import os
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from huggingface_hub import create_repo

BASE_MODEL_NAME = "ibm-granite/granite-docling-258M"
ADAPTER_DIR = "/proj/mmfm/kimhi/data/docling_ar_res/checkpoint-6918/"  # has adapter_model.safetensors
REPO_ID = "Kimhi/granite-docling-res-gate-lora"             # change me

# 1) create repo (noop if exists)
create_repo(REPO_ID, exist_ok=True)

# 2) load base + adapter (so we can save a clean adapter package)
processor = AutoProcessor.from_pretrained(ADAPTER_DIR)  # keeps your chat template/tokenizer config if you stored it
base = AutoModelForVision2Seq.from_pretrained(BASE_MODEL_NAME)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

# 3) save ONLY adapter artifacts + (optionally) processor files
out = "./exported_adapter"
os.makedirs(out, exist_ok=True)
model.save_pretrained(out)                 # adapter_model.safetensors + adapter_config.json
processor.save_pretrained(out)             # optional but handy if you changed template/tokenizer settings

# 4) upload
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(repo_id=REPO_ID, folder_path=out, commit_message="Upload LoRA adapter")
print("Uploaded:", REPO_ID)

