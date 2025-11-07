# pip install -U huggingface_hub
from huggingface_hub import HfApi, create_repo, create_collection, add_collection_item
import os
from pathlib import Path

# HF_TOKEN = os.getenv("HF_TOKEN")  # or paste a token string here
OWNER = "chieunq"                # or your org name
LOCAL_ROOT = Path("models")       # your local folder shown in the screenshot
COLLECTION_TITLE = "Qwen3 quantized variants"
COLLECTION_NAMESPACE = OWNER      # put an org/user here
TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=TOKEN)

# 1) Create the collection once (idempotent-ish)
def get_or_create_collection(title: str):
    # list all collections you own
    collections = api.list_collections()  
    for c in collections:
        if c.title == title:
            print(f"✅ Using existing collection: {c.slug}")
            return c

    # otherwise create a new one
    new_c = api.create_collection(
        title=title,
        description="All Qwen3 quantized builds (AWQ, GPTQ, RTN, etc.) grouped by size.",
        namespace=OWNER,
    )
    print(f"🆕 Created collection: {new_c.slug}")
    return new_c

collection = get_or_create_collection(COLLECTION_TITLE)

# 2) Walk your local tree and push each leaf folder as its own repo
def repo_name_from_path(path: Path) -> str:
    # e.g. models/1.7B/Qwen3-1.7B-GPTQ-W8A8_gsm8k2048  ->  Qwen3-1.7B-GPTQ-W8A8_gsm8k2048
    return path.name

leaf_dirs = []
for root, dirs, files in os.walk(LOCAL_ROOT):
    # consider a "leaf" directory one that actually contains model files (e.g., *.safetensors, *.gguf, tokenizer.json, etc.)
    if any(f.endswith((".safetensors", ".gguf", ".bin", ".pt")) or f in {"tokenizer.json", "config.json"} for f in files):
        leaf_dirs.append(Path(root))

for folder in leaf_dirs:
    repo_name = repo_name_from_path(folder)
    repo_id = f"{OWNER}/{repo_name}"

    # 2a) Create repo if missing
    create_repo(repo_id=repo_id, private=False, exist_ok=True, repo_type="model")

    # 2b) Upload all files in that folder (switch to upload_large_folder for massive dirs)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        # Example filters (optional):
        # allow_patterns=["*.safetensors","*.gguf","*.json","*.md"],
        # ignore_patterns=["**/logs/**"],
    )

    # 2c) Add this repo to the collection
    add_collection_item(
        collection.slug,         # returned when you created the collection
        item_id=repo_id,         # full repo_id "owner/name"
        item_type="model",
        # note="AWQ W4A16 ASYM (GSM8K tuned)"  # optional per-item note
        exists_ok=True           # avoid 409 if you re-run
    )

print("Done. Collection URL:", collection.url)
