import pandas as pd
import numpy as np
import random
import json
import os

# Define where your data lives
DATA_PATH = "data/sandiego_reviews.parquet"
META_PATH = "data/sandiego_meta.json"

class RecSysEngine:
    def __init__(self):
        print("⚙️ Initializing RecSysEngine...")
        
        # 1. Load the Data (Cached by Streamlit in app.py)
        if os.path.exists(DATA_PATH):
            self.df = pd.read_parquet(DATA_PATH)
            print(f"✅ Loaded {len(self.df)} reviews from Parquet.")
        else:
            print("❌ Error: Parquet file not found. Run preprocess.py first.")
            self.df = pd.DataFrame() # Empty fallback

        # 2. Load Metadata
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f:
                self.meta_lookup = json.load(f)
            print(f"✅ Loaded metadata for {len(self.meta_lookup)} places.")
        else:
            self.meta_lookup = {}

        # 3. Warm-up / Training (To be replaced with real model loading later)
        # specific for your logic:
        self.unique_places = self.df['place_name'].unique().tolist() if not self.df.empty else ["Generic Place"]
        
    def get_place_details(self, gmap_id):
        """Helper to get readable names from IDs"""
        return self.meta_lookup.get(gmap_id, {}).get('name', 'Unknown Spot')

    # ---------------------------------------------------------
    # TASK A: Rating Prediction (The "Will I like it?" query)
    # ---------------------------------------------------------
    def predict_rating(self, user_text, place_name):
        """
        TODO: Replace with Matrix Factorization + Word2Vec Proxy
        Current: Random baseline
        """
        # Simulating a calculation delay
        return round(random.uniform(3.5, 5.0), 1)

    # ---------------------------------------------------------
    # TASK B: Visit Prediction (The "Where should I go?" query)
    # ---------------------------------------------------------
    def predict_visit(self, user_history_text):
        """
        TODO: Replace with BPR + Jaccard
        Current: Random baseline returning top 3 random places from dataset
        """
        if not self.unique_places:
            return ["No Data Loaded"]
        
        # Just pick 3 random places from your actual San Diego dataset
        recs = random.sample(self.unique_places, 3)
        return recs

    # ---------------------------------------------------------
    # TASK C: Category Prediction (The "I want Italian" query)
    # ---------------------------------------------------------
    def predict_category(self, user_query):
        """
        TODO: Replace with TF-IDF / Cosine Similarity
        Current: Keyword matching
        """
        # Simple rule-based baseline
        if "sushi" in user_query.lower():
            return "Japanese"
        elif "taco" in user_query.lower():
            return "Mexican"
        else:
            return "Restaurant"

    # ---------------------------------------------------------
    # The LLM "Router" (The Brain)
    # ---------------------------------------------------------
    def generate_response(self, user_input):
        """
        Decides which function to call based on user input.
        """
        user_input_lower = user_input.lower()

        # 1. Check for Rating Intent
        if "rate" in user_input_lower or "like" in user_input_lower:
            # Mock extracting a place name (in real life, use the LLM to extract this)
            target_place = "The Taco Stand" 
            score = self.predict_rating(user_input, target_place)
            return f"🤔 Based on your taste profile, I predict you'd give **{target_place}** a **{score}/5**."

        # 2. Check for Recommendation Intent
        elif "recommend" in user_input_lower or "go" in user_input_lower or "visit" in user_input_lower:
            recs = self.predict_visit(user_input)
            rec_list = "\n".join([f"- {r}" for r in recs])
            return f"📍 Here are 3 places in San Diego you might like:\n{rec_list}"

        # 3. Check for Category Intent
        elif "find" in user_input_lower or "looking for" in user_input_lower:
            cat = self.predict_category(user_input)
            return f"🔎 It sounds like you're interested in **{cat}**. I can find the best spots for that!"

        # 4. Fallback
        else:
            return "👋 I'm your San Diego Dining Assistant. You can ask me to:\n- **Predict** if you'll like a specific place.\n- **Recommend** a place to visit.\n- **Find** a category of food."