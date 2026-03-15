import json, pathlib, os
import pandas as pd


root = pathlib.Path(
    "/proj/mmfm/kimhi/data"
)
#root2 = pathlib.Path("/proj/mmfm/kimhi/data/data/mix2")
# with open(root2/ "data3.json") as f:
#     raw = json.load(f)

root2 = pathlib.Path("/proj/mmfm/kimhi/data/cauldron3")
with open(root2/ "data.json") as f:
    raw = json.load(f)

records = []
for ex in raw:
    # just in mixed data
    # if ('multi' in ex['original_dataset']):
    #      print(ex)
    #      break
    q = ex["conversations"][0]["value"].replace("<image>\n", "").strip()
    gpt_first = ex["conversations"][1]["value"].strip()
    # label = 1  ↔ “MORE PIXELS!” (hard); everything else is 0 (easy)
    hard = 0
    if gpt_first.upper().startswith("<RESOLUTION VALUE=384"):
        hard = 0
    elif gpt_first.upper().startswith("<RESOLUTION VALUE=768"):
        hard = 1
    elif gpt_first.upper().startswith("<RESOLUTION VALUE=1024"):
        hard = 2
    #hard = int(gpt_first.upper().startswith("<RESOLUTION VALUE=1024"))
    # there may be 1 or 2 image paths (low‑res only vs low+high)
    #print(ex["image"])
    def replace_last(string, old, new):
        parts = string.rsplit(old, 1)  # Split from right, max 1 split
        return new.join(parts)
    low_path  = ex["image"][0]
    low_path = os.path.join(str(root),low_path)
    mid_path = ex["image"][1] if len(ex["image"]) > 1 else replace_last(ex["image"][0],"384","768")
    mid_path = os.path.join(str(root),mid_path)
    high_path = ex["image"][-1] if len(ex["image"]) > 2 else replace_last(ex["image"][0],"384","1024")
    high_path = os.path.join(str(root),high_path)
    records.append(dict(
        id        = ex["id"],
        question  = q,
        low_path  = low_path,
        mid_path  = mid_path,
        high_path = high_path,
        hard      = hard,
    ))

df = pd.DataFrame(records)
####_nomul
df.to_parquet("hardness_cauldrode3.parquet")   # ~20× faster than CSV for I

par = pd.read_parquet('hardness_cauldrode3.parquet', engine='pyarrow')
print(par)