"""
San Diego Dining Concierge - Comprehensive Preprocessing
Single Source of Truth for Data Preprocessing

Extracts ALL available fields from Google Local California dataset:
- Metadata: 15 fields (name, address, location, categories, ratings, price, hours, etc.)
- Reviews: 8 fields (user, rating, text, timestamp, pics, response, etc.)
- Derived: 10+ features (text metrics, temporal features, rating flags)

Usage:
    # Process only San Diego restaurants (default)
    python preprocess.py
    
    # Process full California dataset (commented out for safety - very large!)
    # Set PROCESS_FULL_CALIFORNIA = True below
"""

import gzip
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
# Input Files (in data/ directory)
META_FILE = 'data/meta-California.json.gz'
REVIEW_FILE = 'data/review-California_10.json.gz'

# Output Files
OUTPUT_DIR = 'data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sandiego_reviews.parquet')
META_MAP_FILE = os.path.join(OUTPUT_DIR, 'sandiego_meta.json')
STATS_FILE = os.path.join(OUTPUT_DIR, 'preprocessing_stats.json')

# Filters for San Diego subset
TARGET_CITY = "San Diego"
TARGET_CATEGORY = "Restaurant"

# ⚠️ FLAG: Set to True to process FULL California dataset (warning: very large!)
# For this assignment, we only need San Diego restaurants (default: False)
PROCESS_FULL_CALIFORNIA = False


def parse_gz(path):
    """Parse gzipped JSON file line by line"""
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)


def extract_all_metadata_fields(data):
    """Extract ALL available fields from metadata"""
    return {
        # Core identifiers
        'gmap_id': data.get('gmap_id'),
        'name': data.get('name', 'Unknown'),
        
        # Location information
        'address': data.get('address', ''),
        'latitude': data.get('latitude'),
        'longitude': data.get('longitude'),
        'state': data.get('state', ''),
        
        # Business information
        'category': data.get('category', []),
        'avg_rating': data.get('avg_rating', 0),
        'num_of_reviews': data.get('num_of_reviews', 0),
        'price': data.get('price'),
        'hours': data.get('hours'),
        'description': data.get('description', ''),
        'url': data.get('url', ''),
        
        # Additional metadata
        'MISC': data.get('MISC', {}),
        'relative_results': data.get('relative_results', [])
    }


def extract_all_review_fields(review):
    """Extract ALL available fields from review"""
    resp = review.get('resp', {})
    
    return {
        # Core fields
        'user_id': review.get('user_id'),
        'gmap_id': review.get('gmap_id'),
        'rating': review.get('rating'),
        'text': review.get('text', ''),
        'timestamp': review.get('time'),
        
        # User information
        'user_name': review.get('name', ''),
        
        # Review metadata
        'pics': review.get('pics', []),
        'num_pics': len(review.get('pics', [])) if review.get('pics') else 0,
        
        # Business response
        'has_response': bool(resp),
        'response_text': resp.get('text') if resp else None,
        'response_time': resp.get('time') if resp else None,
    }


def compute_derived_features(df):
    """Compute additional features from raw data"""
    
    # Text features
    if 'text' in df.columns:
        df['text_length'] = df['text'].fillna('').apply(len)
        df['text_word_count'] = df['text'].fillna('').apply(lambda x: len(str(x).split()))
    
    # Temporal features
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour
    
    # Rating features
    if 'rating' in df.columns:
        df['is_positive'] = df['rating'] >= 4
        df['is_negative'] = df['rating'] <= 2
    
    # Photo features
    if 'num_pics' in df.columns:
        df['has_photos'] = df['num_pics'] > 0
    
    return df


def get_corpus(df, text_col='text'):
    """Return tokenized corpus suitable for Word2Vec training.
    
    This can be imported and called from notebooks for consistent preprocessing.
    """
    corpus = []
    for text in df[text_col].fillna(''):
        tokens = str(text).lower().split()
        if tokens:
            corpus.append(tokens)
    return corpus


def get_place_to_categories(metadata):
    """Build mapping from place name to list of categories.
    
    Can be imported and called from notebooks.
    Returns: dict(place_name -> list of categories)
    """
    place_to_categories = {}
    for gmap_id, info in (metadata or {}).items():
        name = info.get('name')
        cats = info.get('category', [])
        
        if not name:
            continue
        
        # Normalize categories to list
        if isinstance(cats, str):
            try:
                cats = json.loads(cats)
            except:
                cats = [cats]
        
        if not isinstance(cats, list):
            cats = [cats] if cats else []
        
        place_to_categories[name] = cats
    
    return place_to_categories


def load_preprocessed_data(data_dir='data'):
    """Load preprocessed data (use this in notebooks/analysis)"""
    parquet_file = os.path.join(data_dir, 'sandiego_reviews.parquet')
    meta_file = os.path.join(data_dir, 'sandiego_meta.json')
    
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(
            f"Preprocessed data not found at {parquet_file}. "
            "Please run: python preprocess.py"
        )
    
    # Load reviews
    df = pd.read_parquet(parquet_file)
    
    # Load metadata
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    return df, metadata


def run_processing():
    """Main processing function with comprehensive data extraction"""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    stats = {
        'processing_date': datetime.now().isoformat(),
        'source_files': {
            'meta': META_FILE,
            'reviews': REVIEW_FILE
        },
        'filters': {
            'city': None if PROCESS_FULL_CALIFORNIA else TARGET_CITY,
            'category': None if PROCESS_FULL_CALIFORNIA else TARGET_CATEGORY,
            'full_california': PROCESS_FULL_CALIFORNIA
        }
    }
    
    print("=" * 60)
    print("🚀 PHASE 1: Scanning Metadata for Valid Places...")
    print("=" * 60)
    
    if PROCESS_FULL_CALIFORNIA:
        print("⚠️  WARNING: Processing FULL California dataset!")
        print("   This will take significant time and disk space.")
    else:
        print(f"📍 Filtering for: {TARGET_CITY} {TARGET_CATEGORY}s")
    
    valid_gmap_ids = set()
    place_metadata = {}
    metadata_fields_found = set()
    
    count = 0
    for data in parse_gz(META_FILE):
        count += 1
        if count % 100000 == 0:
            print(f"   Scanned {count:,} places...")
        
        # Track all fields we encounter
        metadata_fields_found.update(data.keys())
        
        # Apply filters only if not processing full California
        if not PROCESS_FULL_CALIFORNIA:
            # Filter by city
            address = data.get('address', '')
            if not address or TARGET_CITY not in address:
                continue
            
            # Filter by category
            categories = data.get('category')
            if not categories or TARGET_CATEGORY not in str(categories):
                continue
        
        gmap_id = data['gmap_id']
        valid_gmap_ids.add(gmap_id)
        
        # Extract ALL metadata fields
        place_metadata[gmap_id] = extract_all_metadata_fields(data)
    
    stats['metadata'] = {
        'total_places_scanned': count,
        'valid_places_found': len(valid_gmap_ids),
        'fields_available': sorted(list(metadata_fields_found))
    }
    
    location_desc = "California" if PROCESS_FULL_CALIFORNIA else f"{TARGET_CITY}"
    print(f"✅ Found {len(valid_gmap_ids):,} places in {location_desc}")
    print(f"📊 Metadata fields: {', '.join(sorted(metadata_fields_found))}")
    
    print("\n" + "=" * 60)
    print("🚀 PHASE 2: Extracting Reviews with ALL Fields...")
    print("=" * 60)
    
    cleaned_reviews = []
    review_fields_found = set()
    review_count = 0
    skipped_empty_text = 0
    
    for review in parse_gz(REVIEW_FILE):
        review_count += 1
        if review_count % 100000 == 0:
            print(f"   Processed {review_count:,} raw reviews...")
        
        # Track all fields we encounter
        review_fields_found.update(review.keys())
        
        # Only keep reviews for our valid places
        if review['gmap_id'] not in valid_gmap_ids:
            continue
        
        # Skip empty text reviews
        if not review.get('text'):
            skipped_empty_text += 1
            continue
        
        # Extract ALL review fields
        review_data = extract_all_review_fields(review)
        
        # Add place name for convenience
        review_data['place_name'] = place_metadata[review['gmap_id']]['name']
        
        cleaned_reviews.append(review_data)
    
    stats['reviews'] = {
        'total_reviews_scanned': review_count,
        'valid_reviews_extracted': len(cleaned_reviews),
        'skipped_empty_text': skipped_empty_text,
        'fields_available': sorted(list(review_fields_found))
    }
    
    print(f"✅ Extracted {len(cleaned_reviews):,} reviews")
    print(f"📊 Review fields: {', '.join(sorted(review_fields_found))}")
    
    print("\n" + "=" * 60)
    print("💾 PHASE 3: Computing Derived Features...")
    print("=" * 60)
    
    df = pd.DataFrame(cleaned_reviews)
    df = compute_derived_features(df)
    
    # Compute aggregate statistics
    stats['final_dataset'] = {
        'total_reviews': len(df),
        'total_users': df['user_id'].nunique(),
        'total_places': df['place_name'].nunique(),
        'date_range': {
            'earliest': df['datetime'].min().isoformat() if df['datetime'].notna().any() else None,
            'latest': df['datetime'].max().isoformat() if df['datetime'].notna().any() else None
        },
        'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
        'columns': df.columns.tolist(),
        'sparsity': 1 - (len(df) / (df['user_id'].nunique() * df['place_name'].nunique()))
    }
    
    print(f"✅ Computed {len(df.columns)} total features")
    
    print("\n" + "=" * 60)
    print("💾 PHASE 4: Saving to Disk...")
    print("=" * 60)
    
    # Save dataframe
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Saved reviews: {OUTPUT_FILE}")
    
    # Save metadata (convert to serializable format)
    serializable_metadata = {}
    for gmap_id, data in place_metadata.items():
        serializable_metadata[gmap_id] = {
            k: str(v) if isinstance(v, (list, dict)) else v 
            for k, v in data.items()
        }
    
    with open(META_MAP_FILE, 'w') as f:
        json.dump(serializable_metadata, f, indent=2, default=str)
    print(f"✅ Saved metadata: {META_MAP_FILE}")
    
    # Save statistics
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"✅ Saved statistics: {STATS_FILE}")
    
    print("\n" + "=" * 60)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Dataset Summary:")
    print(f"   • Reviews: {len(df):,}")
    print(f"   • Users: {df['user_id'].nunique():,}")
    print(f"   • Places: {df['place_name'].nunique():,}")
    print(f"   • Features: {len(df.columns)}")
    print(f"   • Sparsity: {stats['final_dataset']['sparsity']:.2%}")
    print(f"\n📁 Output Files:")
    print(f"   • {OUTPUT_FILE}")
    print(f"   • {META_MAP_FILE}")
    print(f"   • {STATS_FILE}")
    
    return df, place_metadata, stats


if __name__ == "__main__":
    df, metadata, stats = run_processing()
