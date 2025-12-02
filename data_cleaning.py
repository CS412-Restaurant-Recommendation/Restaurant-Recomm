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

            keep_keys = [
                "business_id", "name", "stars", "review_count", "categories", "city"
            ]

            record = {key: record.get(key) for key in keep_keys}

            categories = record.get("categories")

            if categories:
                # Convert string → list
                cat_list = [c.strip() for c in categories.split(",")]

                cleaned_list = []
                for cat in cat_list:
                    lc = cat.lower()

                    # REMOVE exactly "restaurants"
                    if lc == "restaurants":
                        continue

                    # REMOVE exactly "food"
                    if lc == "food":
                        continue

                    # KEEP everything else ("Fast Food", "Specialty Food", "Seafood", etc.)
                    cleaned_list.append(cat)

                # Save cleaned category string
                record["categories"] = ", ".join(cleaned_list) if cleaned_list else None
            else:
                record["categories"] = None

            fout.write(json.dumps(record) + "\n")

print(f"Filtering complete. Saved filtered data to: {output_file}")
