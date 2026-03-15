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
import numpy as np

def compute_stats(data, threshold):
    total = len(data)
    above = [v for v in data.values() if float(v.get("384", 0)) > threshold]
    perc_384 = len(above) / total * 100 if total > 0 else 0

    if above:
        perc_768 = sum(1 for v in data.values() if float(v.get("768", 0)) > threshold) / total * 100
        #perc_1024 = sum(1 for v in above if float(v.get("1024", 0)) > threshold) / len(above) * 100 - perc_384 - perc_768
        perc_1024 =  sum(1 for v in data.values() if float(v.get("1024", 0)) > threshold) / len(data.values()) * 100
    else:
        perc_768 = perc_1024 = 0

    return {"threshold": threshold, "384": perc_384, "768": perc_768, "1024": perc_1024}


def compute_stats_2(data, threshold):
    total = len(data)
    print(total)
    if total == 0:
        return 0, 0, 0

    count_384 = 0
    count_768 = 0
    count_1024 = 0
    count_lower = 0
    s = []
    for v in data.values():
        score_384 = float(v.get("384", 0))
        score_768 = float(v.get("768", 0))
        score_1024 = float(v.get("1024", 0))

        s.append(score_384)
        # if score_384 > threshold:
        #     count_384 += 1

        #     # check if 768 improves by at least threshold over 384
        # if score_768 - score_384 >= threshold:
        #     count_768 += 1

        #     # check if 1024 improves by at least threshold over 768
        # elif score_1024 - score_384 >= threshold:
        #     count_1024 += 1

        if score_384>0.8:
            count_384 +=1
        elif (score_768 - score_384) >= threshold and  (score_1024 - score_768) <= threshold :
            count_768 +=1
        elif  (score_1024 - score_384)> threshold or  (score_1024 - score_384) > threshold:
            count_1024 +=1
        else:
            count_384 +=1

        if score_768 < score_384 -0.02 or score_1024 < score_384 -0.02 or score_1024 < score_768 -0.02:
        #if score_1024 - score_768 < threshold and score_1024 - score_384 > threshold:
            count_lower += 1
            #find the key for printing
            #key = [k for k, val in data.items() if val == v][0]
            #print(f'for {key} anls of 384: {score_384}, 768: {score_768}, 1024: {score_1024}')
    perc_384 = count_384 / total * 100 #np.mean(s)*100
    perc_768 = count_768 / total * 100
    perc_1024 = count_1024 / total * 100

    return  {"threshold": threshold, "384": perc_384, "768": perc_768, "1024": perc_1024, "lower": count_lower}




def parse_args():
    parser = argparse.ArgumentParser(description="Compare model responses at different resolutions")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to a json of res anls")
    parser.add_argument("--threshold", type=float, default=0.85)
    return parser.parse_args()


def main():
    # Load your JSON file
    args = parse_args()
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Compute stats for thresholds
    threshold = args.threshold
    stats = compute_stats_2(data, threshold)
    print(f"\nThreshold > {threshold} for {args.input_file}:")
    print(f'384  : {stats["384"]:.2f}%')
    print(f'768  : {stats["768"]:.2f}%')
    print(f'1024 : {stats["1024"]:.2f}%')
    print(f'lower res better count : {stats["lower"]}')


if __name__ == "__main__":
    main()
