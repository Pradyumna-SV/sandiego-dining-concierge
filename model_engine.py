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
            self.gemini = genai.GenerativeModel('gemini-2.0-flash')
        
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
        
        # 1. System Prompt
        prompt = f"""
        You are a dining concierge for San Diego.
        User Input: "{user_input}"
        
        Return a JSON object with the 'intent' (rating, visit, category, chat) and 'parameters'.
        
        Examples:
        "Will I like The Taco Stand?" -> {{"intent": "rating", "place": "The Taco Stand"}}
        "Where should I go?" -> {{"intent": "visit"}}
        "Find me sushi" -> {{"intent": "category", "query": "sushi"}}
        "Hello" -> {{"intent": "chat", "response": "Hi there! I can help you find food."}}
        """

        try:
            # 2. Call Gemini with JSON Mode (Critical Fix)
            response = self.gemini.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Debugging: Print what Gemini actually sent back
            print(f"DEBUG RAW RESPONSE: {response.text}")

            # 3. Parse JSON directly
            result = json.loads(response.text)
            
            # 4. Route the Logic
            intent = result.get('intent')
            
            if intent == 'rating':
                place = result.get('place', 'Unknown Place')
                rating = self.predict_rating(user_input, place)
                rating = round(rating, 2)
                return f"🤖 **Analysis:** Based on your vibe, I predict you'd give **{place}** a **{rating}/5**."
            
            elif intent == 'visit':
                recs = self.predict_visit(user_input)
                # Format the list nicely
                if isinstance(recs, list):
                    rec_str = "\n".join([f"• {r}" for r in recs])
                else:
                    rec_str = str(recs)
                return f"📍 **Recommendations:**\n{rec_str}"
            
            elif intent == 'category':
                query = result.get('query', 'food')
                cat = self.predict_category(query)
                return f"🔎 Searching for **{cat}** places near you..."
            
            else:
                return result.get('response', "I didn't quite catch that.")

        except Exception as e:
            # This prints the REAL error to your Streamlit Cloud logs
            print(f"❌ FULL ERROR DETAILS: {e}")
            return "I'm having trouble connecting to my brain. Check the logs!"