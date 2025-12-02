import json
import pandas as pd

business_file = "Dataset/filtered_yelp_business.json"
ids_file = "Dataset/train-business-ids-only.csv"
output_file = "Dataset/filtered_yelp_business(new).json"

valid_ids = set(pd.read_csv(ids_file)['business_id'])

with open(business_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    
    for line in fin:
        record = json.loads(line)
        if record["business_id"] in valid_ids:

        #Remove these values
        # for key in ["is_open", "attributes"]:
        #     record.pop(key, None)
        # fout.write(json.dumps(record) + "\n")

            # Keep these values
            keep_keys = [
                "business_id", "name", "stars", "review_count", "categories", "city"
            ]

            #TODO - Inside the categories list, remove check is it has "restaurant" if it exists and keep the rest of the categories

            record = {key: record.get(key) for key in keep_keys}

            fout.write(json.dumps(record) + "\n")

print(f"Filtering complete. Saved filtered data to: {output_file}")