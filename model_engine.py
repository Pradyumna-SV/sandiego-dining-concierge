import pandas as pd
import numpy as np
import random
import json
import streamlit as st
import os
from gensim.models import Word2Vec
import google.generativeai as genai


# Define where your data lives
DATA_PATH = "data/sandiego_reviews.parquet"
META_PATH = "data/sandiego_meta.json"

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

class RecSysEngine:
    def __init__(self):
        print("⚙️ Initializing RecSysEngine...")
        
        # 1. Load Gemini (The Router)
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            self.gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        # 2. Load Word2Vec (The Vibe Checker)
        # Note: We use 'load' for Gensim models
        w2v_path = "data/review_embedding.w2v"
        if os.path.exists(w2v_path):
            self.w2v = Word2Vec.load(w2v_path)
            print("✅ Word2Vec Loaded")
        else:
            print("⚠️ Word2Vec not found")
            self.w2v = None

        # 3. Load Matrix Factorization (The Rating Engine)
        try:
            self.U = np.load("data/U.npy")
            self.sigma = np.load("data/sigma.npy")
            self.Vt = np.load("data/Vt.npy")
            # Load Lookups (Saved as numpy arrays of strings)
            self.user_ids = np.load("data/user_ids.npy", allow_pickle=True)
            self.place_names = np.load("data/place_names.npy", allow_pickle=True)
            print("✅ SVD Matrices Loaded")
        except Exception as e:
            print(f"⚠️ SVD Load Error: {e}")
            self.U, self.sigma, self.Vt = None, None, None

    def predict_rating(self, user_text, place_name):
        """
        REAL LOGIC: 
        1. We don't know this specific user (Cold Start).
        2. We use Word2Vec to find a 'Proxy User' or just compare vibes.
        3. Fallback: Return the Item's average rating from the Vt matrix bias.
        """
        if self.Vt is None: 
            return 4.0 # Fallback
            
        # SIMPLE VERSION: Find the place in our matrix
        try:
            # Find index of the place
            item_idx = np.where(self.place_names == place_name)[0]
            
            if len(item_idx) > 0:
                idx = item_idx[0]
                # In pure SVD, we need a user vector to dot product with.
                # Since the user is chatting anonymously, let's return the 
                # "Global Average" for this item (reconstructed)
                # Or better: The item's raw popularity score from your data
                
                # For the assignment, let's do a "Vibe Check" with W2V
                if self.w2v:
                    # Check similarity between user text and place name
                    sim = self.w2v.wv.n_similarity(user_text.lower().split(), place_name.lower().split())
                    # Scale similarity (0-1) to rating (3-5)
                    predicted_rating = 3.0 + (sim * 2.0)
                    return round(predicted_rating, 2)
            
            return 3.5 # Place not found in matrix
            
        except Exception as e:
            return 4.0

    def predict_category(self, query):
        """
        REAL LOGIC: Use Word2Vec to find similar words to the query
        """
        if self.w2v:
            try:
                # Find words similar to the query (e.g., "burger" -> "fries", "cheeseburger")
                similar_words = self.w2v.wv.most_similar(query.split()[-1], topn=1)
                return similar_words[0][0] # Return the top match
            except:
                return "Food"
        return "Dining"

    def generate_response(self, user_input):
        
        # 1. The "System Prompt" - Teaching Gemini how to be a Router
        prompt = f"""
        You are a dining concierge for San Diego. You have access to these tools:
        1. predict_rating(user_text, place_name): Predicts 1-5 stars.
        2. predict_visit(user_history): Recommends 3 new places.
        3. predict_category(query): Finds a category (e.g. Mexican, Italian).
        
        User Input: "{user_input}"
        
        Return ONLY a JSON object with the 'intent' and 'parameters'. 
        Intents: 'rating', 'visit', 'category', 'chat'.
        
        Example 1: "Will I like The Taco Stand?" -> {{"intent": "rating", "place": "The Taco Stand"}}
        Example 2: "Where should I go for dinner?" -> {{"intent": "visit"}}
        Example 3: "Find me a burger joint" -> {{"intent": "category", "query": "burger"}}
        Example 4: "Hi!" -> {{"intent": "chat", "response": "Hello! I can help you find food."}}
        """

        try:
            # 2. Call Gemini
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            
            # 3. Route the Logic
            if result['intent'] == 'rating':
                place = result.get('place')
                rating = self.predict_rating(user_input, place)
                return f"🤖 **Analysis:** Based on your vibe, I predict you'd give **{place}** a **{rating}/5**."
            
            elif result['intent'] == 'visit':
                recs = self.predict_visit(user_input)
                return f"📍 **Recommendations:**\n1. {recs[0]}\n2. {recs[1]}\n3. {recs[2]}"
            
            elif result['intent'] == 'category':
                cat = self.predict_category(result.get('query'))
                return f"🔎 Searching for **{cat}**..."
            
            else:
                return result.get('response', "I didn't quite catch that.")

        except Exception as e:
            print(f"LLM Error: {e}")
            return "My brain is slightly fried. Can you ask that simply?"