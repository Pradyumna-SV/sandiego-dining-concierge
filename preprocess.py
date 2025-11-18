import gzip
import json
import pandas as pd
import os

# --- CONFIGURATION ---
# Input Files
META_FILE = 'meta-California.json.gz'       # Check if yours ends in .gz! If so, change logic below.
REVIEW_FILE = 'review-California_10.json.gz'

# Output File
OUTPUT_DIR = 'data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sandiego_reviews.parquet')
META_MAP_FILE = os.path.join(OUTPUT_DIR, 'sandiego_meta.json')

# Filters
TARGET_CITY = "San Diego"
TARGET_CATEGORY = "Restaurant" # Looks for this string in the category list

def parse_gz(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def parse_json(path):
    # If your meta file is standard text, use this. 
    # If it is .gz, use parse_gz instead.
    with open(path, 'r') as f:
        for l in f:
            yield json.loads(l)

def run_processing():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("🚀 Phase 1: Scanning Metadata for valid Places...")
    
    valid_gmap_ids = set()
    place_metadata = {}
    
    # NOTE: If your meta file is .gz, change 'parse_json' to 'parse_gz'
    # Most of these datasets have metadata as .json.gz too, check your file extension!
    try:
        # Attempting to read as plain JSON first based on your filename
        iterator = parse_gz(META_FILE) 
    except:
        # Fallback if you actually meant .gz
        print("Plain read failed, trying gzip for metadata...")
        iterator = parse_gz(META_FILE + ".gz")

    count = 0
    for data in iterator:
        count += 1
        if count % 100000 == 0:
            print(f"   Scanned {count} places...")

        # 1. Check City (Simple text match in address)
        address = data.get('address', '')
        if not address or TARGET_CITY not in address:
            continue

        # 2. Check Category
        # Categories are usually a list like ["Mexican", "Restaurant"]
        categories = data.get('category')
        if not categories:
            continue
            
        # Check if any category string matches our target
        # We convert list to string to make searching easier
        if TARGET_CATEGORY not in str(categories):
            continue

        # It's a match! Save the ID.
        gmap_id = data['gmap_id']
        valid_gmap_ids.add(gmap_id)
        
        # Save minimal metadata for the App
        place_metadata[gmap_id] = {
            'name': data.get('name', 'Unknown'),
            'address': address,
            'categories': categories,
            'avg_rating': data.get('avg_rating', 0)
        }

    print(f"✅ Found {len(valid_gmap_ids)} {TARGET_CATEGORY}s in {TARGET_CITY}.")

    print("-" * 40)
    print("🚀 Phase 2: Filtering Reviews...")

    cleaned_reviews = []
    review_count = 0
    
    # Iterate through the massive compressed review file
    for review in parse_gz(REVIEW_FILE):
        review_count += 1
        if review_count % 100000 == 0:
            print(f"   Processed {review_count} raw reviews...")

        # Only keep review if it belongs to one of our San Diego Restaurants
        if review['gmap_id'] in valid_gmap_ids:
            
            # Optional: Skip empty text reviews if you want purely text-based models
            if not review.get('text'):
                continue
                
            cleaned_reviews.append({
                'user_id': review['user_id'],
                'gmap_id': review['gmap_id'],
                'rating': review['rating'],
                'text': review['text'],
                'timestamp': review['time'],
                # Add the name directly here to make the dataframe easier to read later
                'place_name': place_metadata[review['gmap_id']]['name']
            })

    print(f"✅ Extracted {len(cleaned_reviews)} reviews.")

    # 3. Save to Parquet
    print("-" * 40)
    print("💾 Saving to disk...")
    
    df = pd.DataFrame(cleaned_reviews)
    df.to_parquet(OUTPUT_FILE, index=False)
    
    # Save the place metadata dictionary (useful for lookups in the app)
    with open(META_MAP_FILE, 'w') as f:
        json.dump(place_metadata, f)

    print(f"🎉 Done! Files saved to '{OUTPUT_DIR}/'")
    print(f"   - {OUTPUT_FILE} (The dataset for your App)")
    print(f"   - {META_MAP_FILE} (The lookup table)")

if __name__ == "__main__":
    run_processing()