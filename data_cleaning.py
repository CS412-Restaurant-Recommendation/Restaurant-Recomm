import json
import pandas as pd

business_file = "yelp_academic_dataset_tip.json"
ids_file = "train-business-ids-only.csv"
output_file = "filtered_yelp_tip.json"

valid_ids = set(pd.read_csv(ids_file)['business_id'])

with open(business_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    
    for line in fin:
        record = json.loads(line)
        if record["business_id"] in valid_ids:
            fout.write(json.dumps(record) + "\n")

print(f"Filtering complete. Saved filtered data to: {output_file}")