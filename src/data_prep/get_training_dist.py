import argparse
import threading, tempfile, os
from tqdm import tqdm
import json
import base64
from io import BytesIO
import requests
import time
from pathlib import Path
import io
import urllib3
from PIL import Image
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------------
# Utilities: string distance & ANLS
# ------------------------------

STAGING_CACHE = {}
_STAGING_LOCK = threading.Lock()



def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_anls(pred, gt, alpha=0.0):
    pred = ' '.join(pred.lower().split())
    gt = ' '.join(gt.lower().split())
    distance = levenshtein_distance(pred, gt)
    max_distance = max(len(pred), len(gt))
    if max_distance == 0:
        return 1.0
    normalized_similarity = 1.0 - (distance / max_distance)
    if normalized_similarity < alpha:
        return 0.0
    return normalized_similarity

# ------------------------------
# Image helpers
# ------------------------------

def pad_image_to_size(image, target_size=384):
    width, height = image.size
    new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    new_image.paste(image, ((target_size - width) // 2, (target_size - height) // 2))
    return new_image


def resize_image_with_padding(image, target_size=384):
    width, height = image.size
    if max(width, height) <= target_size:
        return pad_image_to_size(image, target_size), False
    ratio = target_size / max(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded_img = pad_image_to_size(resized_img, target_size)
    return padded_img, True


def get_image_tiles(image, tile_size=384):
    width, height = image.size
    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size
    tiles = []
    tile_positions = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            left = i * tile_size
            top = j * tile_size
            right = min((i + 1) * tile_size, width)
            bottom = min((j + 1) * tile_size, height)
            tile = image.crop((left, top, right, bottom))
            if tile.size != (tile_size, tile_size):
                tile = pad_image_to_size(tile, tile_size)
            tiles.append(tile)
            tile_positions.append((i, j))
    return tiles, tile_positions

# ------------------------------
# Model discovery & API
# ------------------------------

def get_cache_path():
    cache_dir = Path.home() / '.cache' / 'rits'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'model_name_to_url.json'


def load_cached_models():
    cache_path = get_cache_path()
    if not cache_path.exists():
        return None
    cache_age = time.time() - cache_path.stat().st_mtime
    if cache_age > 24 * 60 * 60:
        return None
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_models_to_cache(models):
    cache_path = get_cache_path()
    try:
        with open(cache_path, 'w') as f:
            json.dump(models, f)
    except IOError:
        print("Warning: Could not save model cache")


def get_available_models():
    cached_models = load_cached_models()
    if cached_models is not None:
        return cached_models
    try:
        api_key = os.environ.get("RITS_API_KEY", "")
        if not api_key:
            print("Warning: RITS_API_KEY not set. Cannot fetch model list.")
            return {}
        url = 'https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo'
        headers = {'RITS_API_KEY': api_key}
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        models_data = response.json()
        model_dict = {}
        for model in models_data:
            model_name = model['model_name']
            endpoint = model['endpoint'].split('/')[-1]
            model_dict[model_name] = endpoint
        save_models_to_cache(model_dict)
        return model_dict
    except Exception as e:
        print(f"Error fetching available models: {e}")
        return {}


def process_image(image, is_path=True):
    try:
        if is_path:
            img = Image.open(image)
        else:
            img = image
        if img.mode in ('P', 'RGBA', 'LA'):
            img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

MODEL_NAME_TO_URL = get_available_models()


def call_model(model_name, prompt, image_content, temperature=0.0, max_new_tokens=2048):
    model_name_in_url = MODEL_NAME_TO_URL[model_name]
    url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name_in_url}/v1/chat/completions'
    headers = {
        "Content-Type": "application/json",
        'RITS_API_KEY': os.environ["RITS_API_KEY"],
    }
    content = [{"type": "text", "text": prompt}]
    if isinstance(image_content, list):
        content.extend(image_content)
    elif image_content is not None:
        content.append(image_content)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API call: {e}")
        return None

# ------------------------------
# Dataset utilities
# ------------------------------

def get_best_resolution(data: Dict[str, float], threshold: float = 0.85) -> str:
    for res in ['384', '768', '1024']:
        if data.get(res) is None:
            continue
        if data[res] >= threshold:
            return res
    return '1024'


def get_image_and_prompt_with_gt(sample: Dict[str, Any], input_file_dir: str = '') -> Tuple[List[str], str, str]:
    images = sample.get('image')
    if not isinstance(images, list):
        images = [images]
    for i, image in enumerate(images):
        if not image:
            return None, None, None
        if not os.path.isabs(image):
            image_candidate = os.path.join(input_file_dir, image)
        else:
            image_candidate = image
        if not os.path.exists(image_candidate):
            return None, None, None
        images[i] = image_candidate
    conversations = sample.get('conversations', [])
    prompt = None
    gt_answer = None
    for conv in conversations:
        if conv['from'] == 'human':
            prompt = conv['value'].replace('<image>', '').strip()
        if conv['from'] == 'gpt':
            gt_answer = conv['value'].strip()
    return images, prompt, gt_answer


# Copy image(s) into a unified output tree while preserving dataset namespace
def stage_images(images_abs: List[str], out_images_dir: str) -> List[str]:
    staged_paths = []
    for p in images_abs:
        with _STAGING_LOCK:
            # reuse if already staged
            if p in STAGING_CACHE and os.path.exists(STAGING_CACHE[p]):
                staged_paths.append(STAGING_CACHE[p])
                continue

            fname = os.path.basename(p)
            target_dir = out_images_dir
            os.makedirs(target_dir, exist_ok=True)
            base, ext = os.path.splitext(os.path.join(target_dir, fname))
            k = 0
            while True:
                candidate = f"{base}{ext}" if k == 0 else f"{base}_{k}{ext}"
                try:
                    # reserve the filename atomically with a lock-file
                    fd = os.open(candidate + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    target = candidate
                    break
                except FileExistsError:
                    k += 1

        # do the actual copy outside the lock to keep threads moving
        try:
            tmp = target + ".tmp"
            shutil.copy2(p, tmp)
            os.replace(tmp, target)   # atomic move into place
        finally:
            # release reservation
            try: os.remove(target + ".lock")
            except FileNotFoundError: pass

        with _STAGING_LOCK:
            STAGING_CACHE[p] = target
            staged_paths.append(target)

    return staged_paths

    
# def stage_images(images_abs: List[str], out_images_dir: str) -> List[str]:
#     staged_paths = []
#     for p in images_abs:
#         # ✅ reuse if we've already staged this exact source image
#         if p in STAGING_CACHE and os.path.exists(STAGING_CACHE[p]):
#             staged_paths.append(STAGING_CACHE[p])
#             continue

#         fname = os.path.basename(p)
#         target_dir = out_images_dir
#         os.makedirs(target_dir, exist_ok=True)
#         target = os.path.join(target_dir, fname)

#         # only disambiguate if a *different* source with same basename already occupied the slot
#         base, ext = os.path.splitext(target)
#         k = 1
#         while os.path.exists(target):
#             # don't try samefile: copies won't match inodes; just make a new name
#             target = f"{base}_{k}{ext}"
#             k += 1

#         shutil.copy2(p, target)
#         STAGING_CACHE[p] = target          # ✅ remember where we put it
#         staged_paths.append(target)
#     return staged_paths

#duplicate images instead of renaming on collision
def stage_images_duplocate(images_abs: List[str], out_images_dir: str) -> List[str]:
    staged_paths = []
    for p in images_abs:
        # keep only file name by default; namespace by dataset
        fname = os.path.basename(p)
        target_dir = out_images_dir# os.path.join(out_images_dir, dataset_ns)
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, fname)
        # handle collisions by appending counter
        base, ext = os.path.splitext(target)
        k = 1
        while os.path.exists(target):
            if os.path.samefile(p, target):
                break
            target = f"{base}_{k}{ext}"
            k += 1
        shutil.copy2(p, target)
        staged_paths.append(target)
    return staged_paths


# ------------------------------
# Parallel worker for one sample
# ------------------------------

def process_one_sample(sample_tuple):
    """Worker that processes a single sample and returns (result_dict, anls_dict_per_id or None)."""
    (sample, dataset_ns, input_dir, out_images_dir, copy_images, threshold, model_name) = sample_tuple
    images_path, prompt, gt_answer = get_image_and_prompt_with_gt(sample, input_file_dir=input_dir)
    if not images_path or not prompt or not gt_answer:
        return None, None
    #get ds name from last part of input_dir
    #dataset_ns = input_dir.split('/')[-1]
    # Load images; call model per available resolution file in list
    anls_dict = {}
    staged_paths = images_path
    if copy_images:
        try:
            staged_paths = stage_images(images_path, out_images_dir)
        except Exception as e:
            print(f"Copy failed for {images_path}: {e}")
            staged_paths = images_path  # fallback: use original

    for img_path in images_path:
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        except Exception:
            continue
        img_content = process_image(image, False)
        if img_content is None:
            continue
        response = call_model(model_name, prompt, img_content)
        if not response:
            continue
        text = response['choices'][0]['message']['content']
        anls = calculate_anls(text, gt_answer)
        # resolution key from filename suffix after last '_' (e.g., *_384.*)
        res = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]
        id_ = sample.get('id')
        if id_ is None:
            id_ = hash((dataset_ns, os.path.basename(img_path), prompt)) & 0xFFFFFFFF
        if id_ not in anls_dict:
            anls_dict[id_] = {res: anls}
        else:
            anls_dict[id_][res] = anls

    if not anls_dict:
        return None, None

    # pick any id in this sample (data layout implies single logical id)
    id_ = next(iter(anls_dict.keys()))
    sufficient = get_best_resolution(anls_dict[id_], threshold=threshold)

    conversations = [{'from': 'human', 'value': '<image>\n' + prompt}]
    if sufficient == '384':
        conversations.append({'from': 'gpt', 'value': gt_answer})
        image_field = [staged_paths[0]]
    else:
        conversations.append({'from': 'gpt', 'value': f'<RESOLUTION value={sufficient} />'})
        conversations.append({'from': 'human', 'value': '<image>'})
        conversations.append({'from': 'gpt', 'value': gt_answer})
        if sufficient == '1024':
            image_field = staged_paths
        elif sufficient == '768':
            image_field = staged_paths[:2]
        else:
            image_field = [staged_paths[0]]

    result_dict = {
        'id': id_,
        'image': image_field,
        'conversations': conversations,
        'sufficient_res': int(sufficient),
        'original_dataset': dataset_ns
    }
    return result_dict, anls_dict

# ------------------------------
# Main
# ------------------------------
def _image_key_from_obj(obj):
    img = obj.get('image')
    if isinstance(img, list) and img:
        img = img[0]
    if not isinstance(img, str):
        return None
    # normalize by filename; adjust if you prefer full path
    return os.path.basename(img)

def load_inputs(inputs: List[str], max_per_dataset: int = None) -> List[Tuple[Dict[str, Any], str, str]]:
    items = []
    for ip in inputs:
        if os.path.isdir(ip):
            ns = os.path.basename(os.path.abspath(ip))
            conv_path = os.path.join(ip, 'conversations.jsonl')
            if not os.path.exists(conv_path):
                print(f"Warning: missing {conv_path}; skipping.")
                continue

            seen_imgs = set()            # ✅ per-dataset unique images
            count = 0                    # ✅ counts unique images only

            with open(conv_path, 'r') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    objs = obj if isinstance(obj, list) else [obj]
                    for s in objs:
                        if not (isinstance(s, dict) and s.get('image')):
                            continue
                        key = _image_key_from_obj(s)
                        if key is None:
                            continue
                        # allow all samples for already-seen images
                        if key in seen_imgs:
                            items.append((s, ns, ip))
                            continue
                        # new image: check cap
                        if max_per_dataset is not None and count >= max_per_dataset:
                            continue
                        seen_imgs.add(key)
                        count += 1
                        items.append((s, ns, ip))
        else:
            ns = os.path.splitext(os.path.basename(ip.split('/')[-2]))[0]
            base_dir = os.path.dirname(os.path.abspath(ip))

            seen_imgs = set()            # ✅ per-file dataset bucket
            count = 0

            with open(ip, 'r') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    objs = obj if isinstance(obj, list) else [obj]
                    for s in objs:
                        if not (isinstance(s, dict) and s.get('image')):
                            continue
                        key = _image_key_from_obj(s)
                        if key is None:
                            continue
                        if key in seen_imgs:
                            items.append((s, ns, base_dir))
                            continue
                        if max_per_dataset is not None and count >= max_per_dataset:
                            continue
                        seen_imgs.add(key)
                        count += 1
                        items.append((s, ns, base_dir))
    return items

#duplicate images instead of renaming on collision
def load_inputs_duplicate(inputs: List[str], max_per_dataset: int = None) -> List[Tuple[Dict[str, Any], str, str]]:
    """Return list of (sample, dataset_namespace, input_dir)."""
    items = []
    for ip in inputs:
        if os.path.isdir(ip):
            ns = os.path.basename(os.path.abspath(ip))
            conv_path = os.path.join(ip, 'conversations.jsonl')
            if not os.path.exists(conv_path):
                print(f"Warning: missing {conv_path}; skipping.")
                continue
            count = 0
            with open(conv_path, 'r') as f:
                for line in f:
                    if max_per_dataset and count >= max_per_dataset:
                        break
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            for s in obj:
                                if max_per_dataset and count >= max_per_dataset:
                                    break
                                items.append((s, ns, ip))
                                count += 1
                        else:
                            items.append((obj, ns, ip))
                            count += 1
                    except Exception:
                        continue
        else:
            ns = os.path.splitext(os.path.basename(ip.split('/')[-2]))[0]
            base_dir = os.path.dirname(os.path.abspath(ip))
            count = 0
            with open(ip, 'r') as f:
                for line in f:
                    if max_per_dataset and count >= max_per_dataset:
                        break
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            for s in obj:
                                if max_per_dataset and count >= max_per_dataset:
                                    break
                                items.append((s, ns, base_dir))
                                count += 1
                        else:
                            items.append((obj, ns, base_dir))
                            count += 1
                    except Exception:
                        continue
    return items




def load_inputs_nobound(inputs: List[str]) -> List[Tuple[Dict[str, Any], str, str]]:
    """Return list of (sample, dataset_namespace, input_dir).
    Robust to jsonl lines that are either objects OR lists of objects.
    """
    def _emit_records(obj):
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []

    items = []
    for ip in inputs:
        if os.path.isdir(ip):
            ns = os.path.basename(os.path.abspath(ip))
            conv_path = os.path.join(ip, 'conversations.jsonl')
            if not os.path.exists(conv_path):
                print(f"Warning: missing {conv_path}; skipping.")
                continue
            with open(conv_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    for rec in _emit_records(data):
                        items.append((rec, ns, ip))
        else:
            ns = os.path.splitext(os.path.basename(ip))[0]
            base_dir = os.path.dirname(os.path.abspath(ip))
            with open(ip, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    for rec in _emit_records(data):
                        items.append((rec, ns, base_dir))
    return items


def load_inputs_one(inputs: List[str]) -> List[Tuple[Dict[str, Any], str, str]]:
    """Return list of (sample, dataset_namespace, input_dir)."""
    items = []
    for ip in inputs:
        if os.path.isdir(ip):
            # Expect conversations.jsonl inside, namespace by dirname
            ns = os.path.basename(os.path.abspath(ip))
            conv_path = os.path.join(ip, 'conversations.jsonl')
            if not os.path.exists(conv_path):
                print(f"Warning: missing {conv_path}; skipping.")
                continue
            with open(conv_path, 'r') as f:
                for line in f:
                    try:
                        items.append((json.loads(line), ns, ip))
                    except Exception:
                        continue
        else:
            # A jsonl file
            ns = os.path.splitext(os.path.basename(ip))[0]
            base_dir = os.path.dirname(os.path.abspath(ip))
            with open(ip, 'r') as f:
                for line in f:
                    try:
                        items.append((json.loads(line), ns, base_dir))
                    except Exception:
                        continue
    return items


def main():
    args = parse_args()

    random.seed(args.seed)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_images_dir = os.path.join(out_dir, 'images')
    os.makedirs(out_images_dir, exist_ok=True)

    # Models
    granite_model = args.model

    # Load & filter samples
    print("Loading samples from inputs...")
    #raw_items = load_inputs(args.inputs)
    raw_items = load_inputs(args.inputs, max_per_dataset=args.max_per_dataset)
    # keep only those with image field present to reduce work
    #items = [(s, ns, idir) for (s, ns, idir) in raw_items if s.get('image')]
    items = [(s, ns, idir) for (s, ns, idir) in raw_items if isinstance(s, dict) and s.get('image')]

    if args.num_samples > 0:
        random.shuffle(items)
        items = items[:args.num_samples]

    # Prepare tasks
    tasks = [
        (s, ns, idir, out_images_dir, not args.no_copy_images, args.threshold, granite_model)
        for (s, ns, idir) in items
    ]

    results: List[Dict[str, Any]] = []
    anls_merged: Dict[Any, Dict[str, float]] = {}

    print(f"Processing {len(tasks)} samples with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_sample, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            try:
                res, anls = fut.result()
            except Exception as e:
                print(f"Worker failed: {e}")
                continue
            if res is not None:
                results.append(res)
            if anls is not None:
                for k, v in anls.items():
                    if k not in anls_merged:
                        anls_merged[k] = v
                    else:
                        anls_merged[k].update(v)

    output_file = os.path.join(out_dir, 'data.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved merged dataset with {len(results)} samples to {output_file}")

    with open(os.path.join(out_dir, 'resolutions.json'), 'w') as f:
        json.dump(anls_merged, f, indent=4)
    print('Saved resolutions.json')


def parse_args():
    p = argparse.ArgumentParser(description="Generate training data from multiple datasets in parallel")
    p.add_argument("--inputs", nargs='+', required=True,
                   help="List of input jsonl files or dataset directories (each dir should contain conversations.jsonl and images/")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory; images will be staged to OUTPUT_DIR/images/<dataset>/...")
    p.add_argument("--num_samples", type=int, default=0,
                   help="Limit number of samples across all inputs (0 = all)")
    p.add_argument("--workers", type=int, default=8, help="Number of parallel workers (threads)")
    p.add_argument("--threshold", type=float, default=0.85, help="ANLS threshold for sufficiency")
    p.add_argument("--model", type=str, default="ibm-granite/granite-vision-3.2-2b", help="Model name to use")
    p.add_argument("--no_copy_images", action="store_true", help="Do not copy images into output; keep original paths")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_dataset", type=int, default=None,
                   help="Maximum number of lines to load from each dataset (per conversations.jsonl)")
    return p.parse_args()


if __name__ == "__main__":
    main()
