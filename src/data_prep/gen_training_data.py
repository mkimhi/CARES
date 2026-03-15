import argparse
import os
from tqdm import tqdm
import json
import base64
from io import BytesIO
import requests
import time
from pathlib import Path
import io
import urllib3
from PIL import Image, ImageFile, UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = None          # disable decompression bomb check
ImageFile.LOAD_TRUNCATED_IMAGES = True # optional
import random
import shutil
Image.MAX_IMAGE_PIXELS = None 
SAFE_MAX_PIXELS = 150_000_000 
import re


def _safe_name(s: str) -> str:
    # keep letters, numbers, dot, dash, underscore
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)


def safe_open(path):
    try:
        return Image.open(path)
    except Image.DecompressionBombError:
        print(f"Skipping huge image (possible bomb): {path}")
        return None
    except UnidentifiedImageError:
        print(f"Bad image file: {path}")
        return None


def get_image_and_prompt_with_gt(sample, input_file_dir=''):
    """Return (images, prompt_str, gt_str) using the FIRST aligned Q→A pair."""
    images = sample.get('image')
    if not isinstance(images, list):
        images = [images]
    for i, p in enumerate(images):
        if not os.path.exists(p):
            pp = os.path.join(input_file_dir, p)
            if os.path.exists(pp):
                images[i] = pp
            else:
                return None, None, None

    pending_q = None
    for turn in sample.get('conversations', []):
        frm = turn.get('from')
        val = (turn.get('value') or '').strip()
        if frm == 'human':
            q = val.replace('<image>', '').strip()
            if q:
                pending_q = q
        elif frm == 'gpt' and pending_q is not None:
            # return FIRST matched pair only
            return images, pending_q, val
    return images, None, None

def save_all_resolutions(img_path, out_dir, sizes=(384, 768, 1024)):
    """
    Create 3 padded-resized PNGs from a single original.
    Names: <orig_stem>_<res>.png  (e.g., 57a1c8..._page0_384.png)
    Returns the saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    img = safe_open(img_path)
    if img is None:
        return []
    with img:
        image = img.convert("RGB")

    base = Path(img_path).stem           # e.g., "57a1c8ce..._page0"
    base = _safe_name(base)              # ensure no slashes/specials

    saved = []
    for res in sizes:
        resized, _ = resize_image_with_padding(image, target_size=res)
        filename = f"{base}_{res}.png"
        save_path = os.path.join(out_dir, filename)
        resized.save(save_path, format="PNG")
        saved.append(save_path)
    return saved



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
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
    """Calculate Average Normalized Levenshtein Similarity between two strings."""
    # Convert to lowercase and remove extra whitespace
    pred = ' '.join(pred.lower().split())
    gt = ' '.join(gt.lower().split())
    
    # Calculate Levenshtein distance
    distance = levenshtein_distance(pred, gt)
    
    # Calculate maximum possible distance
    max_distance = max(len(pred), len(gt))
    
    # Calculate normalized Levenshtein similarity
    if max_distance == 0:
        return 1.0
    
    normalized_similarity = 1.0 - (distance / max_distance)
    
    # Apply threshold
    if normalized_similarity < alpha:
        return 0.0
    
    return normalized_similarity

def pad_image_to_size(image, target_size=384):
    """Pad image to target size with zeros."""
    width, height = image.size
    new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    new_image.paste(image, ((target_size - width) // 2, (target_size - height) // 2))
    return new_image

def resize_image_with_padding(image, target_size=384):
    """Resize image to fit within target_size while maintaining aspect ratio and padding."""
    width, height = image.size
    
    # If image is already smaller than target, pad it
    if max(width, height) <= target_size:
        return pad_image_to_size(image, target_size), False
    
    # Calculate new dimensions maintaining aspect ratio
    ratio = target_size / max(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize image
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Pad to target size
    padded_img = pad_image_to_size(resized_img, target_size)
    return padded_img, True

def get_image_tiles(image, tile_size=384):
    """Get all non-overlapping tiles from the image."""
    width, height = image.size
    
    # Calculate number of tiles
    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size
    
    tiles = []
    tile_positions = []
    
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Calculate tile boundaries
            left = i * tile_size
            top = j * tile_size
            right = min((i + 1) * tile_size, width)
            bottom = min((j + 1) * tile_size, height)
            
            # Extract tile
            tile = image.crop((left, top, right, bottom))
            
            # Pad if necessary
            if tile.size != (tile_size, tile_size):
                tile = pad_image_to_size(tile, tile_size)
            
            tiles.append(tile)
            tile_positions.append((i, j))
    
    return tiles, tile_positions

def get_cache_path():
    """Get the path to the cache file."""
    cache_dir = Path.home() / '.cache' / 'rits'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'model_name_to_url.json'

def load_cached_models():
    """Load models from cache if it exists and is recent."""
    cache_path = get_cache_path()
    if not cache_path.exists():
        return None
    
    cache_age = time.time() - cache_path.stat().st_mtime
    if cache_age > 24 * 60 * 60:  # 24 hours in seconds
        return None
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def save_models_to_cache(models):
    """Save models to cache file."""
    cache_path = get_cache_path()
    try:
        with open(cache_path, 'w') as f:
            json.dump(models, f)
    except IOError:
        print("Warning: Could not save model cache")

def get_available_models():
    """Fetch the list of available models from the RITS API or cache."""
    cached_models = load_cached_models()
    if cached_models is not None:
        return cached_models
    
    try:
        api_key = os.environ.get("RITS_API_KEY", "")
        if not api_key:
            print("Warning: RITS_API_KEY environment variable not set. Cannot fetch model list.")
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

def resize_image(image_path, target_size=384):
    """Resize image keeping aspect ratio so larger side equals target_size."""
    with Image.open(image_path) as img:
        # Get original size
        width, height = img.size
        original_max_side = max(width, height)
        
        # If image is already smaller than target, return original size
        if original_max_side <= target_size:
            return img, False
        
        # Calculate new dimensions
        ratio = target_size / original_max_side
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_img, True

def process_image(image_or_path, is_path=True):
    """Convert an image to base64 format required by the API."""
    try:
        if is_path:
            img = Image.open(image_or_path)
        else:
            img = image_or_path

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

model_name_to_url = get_available_models()

def call_model(model_name, prompt, image_content, temperature=0.0, max_new_tokens=2048):
    """Make API call to the model endpoint."""
    model_name_in_url = model_name_to_url[model_name]    
    url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name_in_url}/v1/chat/completions'
    
    headers = {
        "Content-Type": "application/json",
        'RITS_API_KEY': os.environ["RITS_API_KEY"]
    }
    
    content = []
    content.append({"type": "text", "text": prompt})
    
    # Handle single image or list of images
    if isinstance(image_content, list):
        content.extend(image_content)
    elif image_content is not None:
        content.append(image_content)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_new_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API call: {e}")
        return None


def get_image_and_prompt_with_gt_old(sample, input_file_dir=''):
    """Extract image path and first question from a sample."""
    # Handle image field which can be string or list
    images = sample.get('image')
    if not isinstance(images, list):
        images = [images]
    #for i,image in images: #iterate over resolutions
    for i, image in enumerate(images) :
    # check if image file exist, if not try relative to input file directory
        if not os.path.exists(image):
            image = os.path.join(input_file_dir, image)
        if not os.path.exists(image):
            return None, None,None
        images[i]=image
    # Get first question from conversations
    conversations = sample.get('conversations', [])
    prompt = None
    gt_answer = None
    for conv in conversations:
        if conv['from'] == 'human':
            # Remove <image> tag and any leading/trailing whitespace
            prompt = conv['value'].replace('<image>', '').strip()
        if conv['from'] == 'gpt':
            #save gt answer
            gt_answer = conv['value'].strip()
            
    return images, prompt,gt_answer

def resize_high_res_image(image_path, max_size=5000):
    """Resize image to have max_size on the longer dimension while maintaining aspect ratio."""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size
        
        # If image is already smaller than max_size, return original
        if max(width, height) <= max_size:
            return img, False
        
        # Calculate new dimensions maintaining aspect ratio
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_img, True



def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    granite_model = "ibm-granite/granite-vision-3.2-2b"

    # Sanity: ensure the model endpoint mapping exists
    if granite_model not in model_name_to_url:
        print(f"Error: model '{granite_model}' not found in available models. "
              f"Have you set RITS_API_KEY? (model_name_to_url: {list(model_name_to_url.keys())})")
        return

    print("Loading and shuffling samples...")
    # Load JSON or JSONL seamlessly
    samples = []
    with open(args.input_file, 'r') as f:
        samples = json.load(f)
    print("loaded {} samples".format(len(samples)))
    # with open(args.input_file, "r") as f:
    #     first_char = f.read(1)
    #     f.seek(0)
    #     if first_char == "[":
    #         samples = json.load(f)
    #     else:
    #         for line in f:
    #             if line.strip():
    #                 samples.append(json.loads(line))
    samples = [s for s in samples if s.get("image")]

    if args.num_samples > 0:
        random.shuffle(samples)
        samples = samples[:args.num_samples]

    results = []
    anls_dict = {}

    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        images_path, prompt, gt_answer = get_image_and_prompt_with_gt(
            sample, input_file_dir=os.path.dirname(args.input_file)
        )
        if not images_path or not prompt or not gt_answer:
            continue
        
        raw_id = sample.get("id", str(i))
        sample_id = str(raw_id).strip().replace(" ", "_")
        #sample_id = str(sample.get("id", i))[:-4]
        per_res = {}
        had_anls = False

        #save 3 resized images ({384, 768, 1024}) for each image in images_path:
        print('save resized images for sample id:', sample_id)
        images_path_resized = []
        for orig in images_path:
            images_path_resized.extend(save_all_resolutions(orig, image_dir))


        for resized_path in images_path_resized:
            #print(f"Processing sample {sample_id}, image {img_path}")
            # get resolution token from filename: ..._<res>.ext
            base = os.path.splitext(os.path.basename(resized_path))[0]
            toks = base.split("_")
            res = toks[-1] if toks else "unknown"
            img_content = process_image(resized_path, is_path=True)
            if img_content is None:
                continue
            response = call_model(granite_model, prompt, img_content)
            if not response:
                continue
            try:
                text = response["choices"][0]["message"].get("content", "")
            except Exception:
                continue
            if not text:
                continue

            anls = calculate_anls(text, gt_answer)
            if res.isdigit():
                # keep the best in case of duplicates
                per_res[res] = max(per_res.get(res, 0.0), anls)
            had_anls = True

        if not had_anls or not per_res:
            # nothing usable for this sample
            continue

        anls_dict[sample_id] = per_res

        sufficient = get_best_resolution(per_res, threshold=0.85)

        # Build conversations + decide which images to include
        conversations = [{'from': 'human', 'value': '<image>\n' + prompt}]
        if sufficient == '384':
            image_field = [images_path[0]]
            conversations.append({'from': 'gpt', 'value': gt_answer})
        else:
            if sufficient == '768':
                image_field = images_path[:2]
            else:  # '1024' fallback
                image_field = images_path
            conversations.append({'from': 'gpt', 'value': f'<RESOLUTION value={sufficient} />'})
            conversations.append({'from': 'human', 'value': '<image>'})
            conversations.append({'from': 'gpt', 'value': gt_answer})

        result_dict = {
            'id': sample_id,
            'image': image_field,
            'conversations': conversations,
            'sufficient_res': int(sufficient)
        }
        results.append(result_dict)

    # Write outputs
    output_file = os.path.join(args.output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} samples")
    print(f"Results saved to {output_file}")

    with open(os.path.join(args.output_dir, "resolutions.json"), "w") as f:
        json.dump(anls_dict, f, indent=4)
    print('Saved resolutions.json')



def main_old():
    args = parse_args()

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    granite_model = "ibm-granite/granite-vision-3.2-2b"
    
    results = []
    output_file = os.path.join(args.output_dir, f"data.json")
    
    print("Loading and shuffling samples...")
    #if input is jsonl load line by line:
    # samples = []
    # with open(args.input_file, 'r') as f:
    #     for line in f:
    #         samples.append(json.loads(line))
    #for json load all:
    with open(args.input_file, 'r') as f:
        samples = json.load(f)
    samples = [s for s in samples if s.get('image')]
    print("loaded {} samples".format(len(samples)))
    
    
    # Limit number of samples if specified
    if args.num_samples > 0:
        # Shuffle samples
        random.shuffle(samples)
        samples = samples[:args.num_samples]
    
    anls_dict = {}
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        # try:
        # Get image path and prompt
        images_path, prompt,gt_answer = get_image_and_prompt_with_gt(sample, input_file_dir=os.path.dirname(args.input_file))
        if not images_path or not prompt or not gt_answer:
            continue
        
        # Load images:
        images = []
        for img_path in images_path:
            res = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]
            id = sample.get('id', i)
            img = safe_open(img_path)
            if img is None:
                continue
            #with Image.open(img_path) as img:
            with img:
                image = img.convert('RGB')
                images.append(image)
                #img content
                img_content = process_image(image, False)
                if img_content is None:
                    continue
                #response
                response = call_model(
                    granite_model,
                    prompt,
                    img_content
                )
                if not response:
                    continue
                text = response['choices'][0]['message']['content']
                anls = calculate_anls(text, gt_answer)
                #check if key exists in anls_dict
                if id not in anls_dict:
                    anls_dict[id] = {res:anls}
                else:
                    anls_dict[id][res]=anls
        if id not in anls_dict or not anls_dict[id]:  # nothing recorded for this sample
            continue
        #if id not in anls_dict:
        #    continue
        sufficient = get_best_resolution(anls_dict[id], threshold=0.85)
        conversations = [{'from': 'human', 'value': '<image>\n' + prompt}]
        if sufficient == '384':
            conversations.append({'from': 'gpt', 'value': gt_answer})
            image_field = [images_path[0]]
        else:
            conversations.append({'from': 'gpt', 'value': f'<RESOLUTION value={sufficient} />'}) #'sufficient + ' is needed to answer the question.'})
            conversations.append({'from': 'human', 'value': "<image>"})
            conversations.append({'from': 'gpt', 'value': gt_answer})
            if sufficient == '1024':
                image_field = images_path
            elif sufficient == '768':
                image_field = images_path[:2]
                
            result_dict = {'id':id, 'image': image_field, 'conversations': conversations, 'sufficient_res': int(sufficient)}
            results.append(result_dict)
    output_file = os.path.join(args.output_dir, f"data.json")
    with open(output_file, 'w') as f:
        f.write(json.dumps(results, indent=4))
    
    print(f"Processed {len(results)} samples")
    print(f"Results saved to {output_file}")

    with open("resolutions.json", "w") as f:
        json.dump(anls_dict, f, indent=4)
    print('save resolutions.json')
    


def get_best_resolution(data, threshold=0.85):
    for res in ['384', '768', '1024']:
        if data.get(res) is None:
            continue
        if data[res] >= threshold:
            return res
    return '1024'


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model responses at different resolutions")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to a jsonl with eval samples")
    parser.add_argument("--output_dir", type=str, default="outputs/select_crops", 
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=0, 
                        help="Number of samples to evaluate (0 for all)")
    return parser.parse_args()

if __name__ == "__main__":
    main() 

