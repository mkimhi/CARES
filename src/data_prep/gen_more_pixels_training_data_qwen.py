import torch
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
from PIL import Image
import random
import shutil
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


###768,1024

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare model responses at different resolutions")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to a jsonl with eval samples")
    parser.add_argument("--output_dir", type=str, default="outputs/resolution_comparison", 
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=0, 
                        help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    
    return parser.parse_args()

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

def call_model_rits(model_name, prompt, image_content, temperature=0.0, max_new_tokens=2048):
    """Make API call to the model endpoint."""
    model_name_in_url = model_name_to_url[model_name]    
    url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name_in_url}/v1/chat/completions'
    
    headers = {
        "Content-Type": "application/json",
        'RITS_API_KEY': os.environ["RITS_API_KEY"]
    }
    
    content = []
    content.append({"type": "text", "text": prompt})
    if image_content:
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

def process_image(image, is_path=True):
    """Convert an image to base64 format required by the API."""
    try:
        if is_path:
            img = Image.open(image)
        else:
            img = image
            
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

model_name_to_url = get_available_models()

def call_model(model, inputs, processor, temperature=0.0, max_new_tokens=2048):
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def get_judge_evaluation(judge_model, prompt, image_path, response1, response2):
    """Get judgment from Pixtral model comparing two responses."""
    judge_prompt = f"""You are an expert evaluator of AI model responses. Please evaluate two responses to the same prompt and image.

Original prompt: {prompt}

Response 1:
{response1}

Response 2:
{response2}

Please determine which response provides better, more detailed, or more accurate information.

Respond with just one number, DO NOT include any other text:
0 - Both responses are equally good or equally bad, no significant difference
1 - Response 1 is better
2 - Response 2 is better"""

    judgment = call_model_rits(judge_model, judge_prompt, process_image(image_path, True), max_new_tokens=2)
    if not judgment:
        return None
        
    try:
        result = int(judgment['choices'][0]['message']['content'].strip())
        return result
    except (ValueError, KeyError, AttributeError) as e:
        print(f"Error parsing judgment: {e}")
        return None

def get_image_and_prompt(sample, input_file_dir=''):
    """Extract image path and first question from a sample."""
    # Handle image field which can be string or list
    image = sample.get('image')
    if isinstance(image, list):
        image = image[0] if image else None
    
    # check if image file exist, if not try relative to input file directory
    if not os.path.exists(image):
        image = os.path.join(input_file_dir, image)
    if not os.path.exists(image):
        return None, None
    
    # Get first question from conversations
    conversations = sample.get('conversations', [])
    prompt = None
    for conv in conversations:
        if conv['from'] == 'human':
            # Remove <image> tag and any leading/trailing whitespace
            prompt = conv['value'].replace('<image>', '').strip()
            break
            
    return image, prompt


def process_inputs(image_path, text, processor):
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        # Preparation for inference
    text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    return inputs


def main():
    args = parse_args()
    assert args.worker_id < args.world_size, "worker_id must be smaller than world_size"
        
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Models to use
    # model_to_use = "ibm-granite/granite-vision-3.2-2b"
    # judge_model = "mistralai/Pixtral-Large-Instruct-2411"
    judge_model = "OpenGVLab/InternVL2-Llama3-76B"
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"    
    model_to_use = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)
    results = []
    
    # Load and shuffle samples
    print("Loading and shuffling samples...")
    
    
    with open(args.input_file, 'r') as f:
        samples = json.load(f)
        
    # Filter samples to only include those with image field
    samples = [s for s in samples if s.get('image')]
    
    # Shuffle samples
    random.shuffle(samples)
    # Limit number of samples if specified
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    
    samples = samples[args.worker_id::args.world_size]
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        try:
            # Get image path and prompt
            image_path, prompt = get_image_and_prompt(sample, input_file_dir=os.path.dirname(args.input_file))
            if not image_path or not prompt:
                continue
            
            # Resize image and check if it meets our criteria
            resized_img, was_resized = resize_image(image_path)
            if not was_resized:
                continue  # Skip if image was not large enough to be resized
            
                        # save low res image
            image_type = image_path.split('.')[-1]
            low_res_image_path = os.path.abspath(os.path.join(image_dir, f"{i}_low_res.{image_type}"))
            resized_img.save(low_res_image_path)
            #high_res_image_path = os.path.abspath(os.path.join(image_dir, f"{i}_high_res.{image_type}"))          
            high_res_image_path = os.path.abspath(os.path.join(image_dir, f"{i}.{image_type}"))  
            shutil.copy(image_path, high_res_image_path)
            # Get responses for both resolutions
            high_res_inputs = process_inputs(high_res_image_path, sample["conversations"][0]["value"], processor)
            low_res_inputs = process_inputs(low_res_image_path, sample["conversations"][0]["value"], processor)
            low_res_response = call_model(
                model_to_use, 
                low_res_inputs,
                processor
            )
            
            high_res_response = call_model(
                model_to_use,
                high_res_inputs,
                processor
            )
            
            if not low_res_response or not high_res_response:
                continue
                
            low_res_text = low_res_response
            high_res_text = high_res_response
            
            # If responses are identical, no need to judge
            if low_res_text == high_res_text:
                high_res_helps = False
            else:
                # Get judgment
                judgment = get_judge_evaluation(
                    judge_model,
                    prompt,
                    image_path,
                    low_res_text,
                    high_res_text
                )
                
                if judgment is None:
                    continue
                    
                high_res_helps = (int(judgment) == 2)                      

            gt_ = sample['conversations'][-1]['value']
            conversations = [{'from': 'human', 'value': '<image>\n' + prompt}]
            if high_res_helps:
                conversations.append({'from': 'gpt', 'value': "MORE PIXELS!"})
                conversations.append({'from': 'human', 'value': "<image>"})
                conversations.append({'from': 'gpt', 'value': high_res_text})
                image_field = [low_res_image_path, high_res_image_path]
            else:
                conversations.append({'from': 'gpt', 'value': low_res_text})
                image_field = [low_res_image_path]

            result_dict = {'id': args.worker_id + i * args.world_size, 'image': image_field, 'conversations': conversations, 'hr_helps': int(high_res_helps), 'original_gt': gt_}
            results.append(result_dict)

                
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    output_file = os.path.join(args.output_dir, f"data_{args.worker_id}_{args.world_size}.json")
    with open(output_file, 'w') as f:
        f.write(json.dumps(results, indent=4))
    print(f"Processed {len(results)} samples")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 