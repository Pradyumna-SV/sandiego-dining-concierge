import pandas as pd
import numpy as np
import json
import os
import re
import streamlit as st
from gensim.models import Word2Vec
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
CONFIG = {
    "reviews_path": "data/sandiego_reviews.parquet",
    "meta_path": "data/sandiego_meta.json",
    "w2v_path": "data/review_embedding.w2v",
    "svd_vt": "data/Vt.npy",
    "place_names": "data/place_names.npy",
    "llm_model": "gemini-2.0-flash"
}

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

class RecSysEngine:
    def __init__(self):
        print("⚙️ Initializing RecSysEngine...")
        
        # Memory for Pronoun Resolution
        self.last_mentioned_place = None 
        
        if "GOOGLE_API_KEY" in st.secrets:
            self.gemini = genai.GenerativeModel(CONFIG['llm_model'])
        else:
            self.gemini = None

        if os.path.exists(CONFIG['w2v_path']):
            self.w2v = Word2Vec.load(CONFIG['w2v_path'])
            self.stop_words = {'place', 'spot', 'restaurant', 'food', 'double', 'stand', 'house', 'grill', 'joint', 'eatery', 'shop'}
        else:
            self.w2v = None

        try:
            self.Vt = np.load(CONFIG['svd_vt']) 
            self.place_names = np.load(CONFIG['place_names'], allow_pickle=True)
        except:
            self.Vt, self.place_names = None, None

        try:
            if os.path.exists(CONFIG['meta_path']):
                with open(CONFIG['meta_path'], 'r') as f:
                    raw_data = json.load(f)
                    self.meta_data = pd.DataFrame.from_dict(raw_data, orient='index')
            else:
                df = pd.read_parquet(CONFIG['reviews_path'])
                self.meta_data = df[['place_name', 'categories', 'rating']].drop_duplicates(subset='place_name')
                self.meta_data.rename(columns={'place_name': 'name', 'rating': 'avg_rating'}, inplace=True)
            
            self.meta_data['name_clean'] = self.meta_data['name'].astype(str).str.lower()
            
            if 'categories' in self.meta_data.columns:
                self.meta_data['cat_str'] = self.meta_data['categories'].astype(str).str.lower()
            else:
                self.meta_data['cat_str'] = ""
                
        except Exception as e:
            print(f"Error: {e}")
            self.meta_data = pd.DataFrame()

    def normalize_text(self, text):
        return re.sub(r'\W+', '', str(text)).lower()

    def get_place_details(self, place_name):
        if self.meta_data.empty: return None
        
        # 1. Strict Match
        mask = self.meta_data['name_clean'].str.contains(place_name.lower(), na=False)
        matches = self.meta_data[mask]
        
        # 2. Fuzzy Match
        if matches.empty:
            clean_query = self.normalize_text(place_name)
            temp_norm = self.meta_data['name'].apply(self.normalize_text)
            matches = self.meta_data[temp_norm.str.contains(clean_query, na=False)]

        if not matches.empty:
            if 'avg_rating' in matches.columns:
                return matches.sort_values('avg_rating', ascending=False).iloc[0].to_dict()
            return matches.iloc[0].to_dict()
        return None

    def find_similar_places_svd(self, place_name, top_k=3):
        if self.Vt is None or self.place_names is None: return []
        try:
            idx = np.where(self.place_names == place_name)[0]
            if len(idx) == 0: return []
            target_vector = self.Vt[:, idx[0]].reshape(1, -1)
            sim_scores = cosine_similarity(target_vector, self.Vt.T)[0]
            top_indices = sim_scores.argsort()[::-1][1:top_k+1]
            return [self.place_names[i] for i in top_indices]
        except: return []

    def predict_visit(self, query):
        query_core = query.lower()
        search_terms = {query_core}
        
        if self.w2v:
            try:
                core_noun = query_core.split()[-1]
                similar = self.w2v.wv.most_similar(core_noun, topn=5)
                for word, score in similar:
                    if score > 0.5 and word.lower() not in self.stop_words:
                        search_terms.add(word.lower())
            except: pass
        
        if "cheeseburger" in search_terms: search_terms.add("burger")
        if "taco" in search_terms: search_terms.add("mexican")

        if self.meta_data.empty: return []

        def score_place(row):
            score = 0
            name = str(row['name']).lower()
            cats = row['cat_str']
            for term in search_terms:
                if term in cats: score += 5
                if term in name: score += 3
            return score

        self.meta_data['search_score'] = self.meta_data.apply(score_place, axis=1)
        results = self.meta_data[self.meta_data['search_score'] > 0].copy()
        
        if results.empty and len(query_core.split()) > 1:
            return self.predict_visit(query_core.split()[-1])

        if 'avg_rating' in results.columns:
            results = results.sort_values(['search_score', 'avg_rating'], ascending=[False, False])
        
        return results['name'].head(5).tolist()

    def generate_response(self, user_input, history=[]):
        context_summary = "User History: " + " | ".join([msg['content'] for msg in history[-3:] if msg['role'] == 'user'])
        focus_info = f"Current Focus: {self.last_mentioned_place}" if self.last_mentioned_place else "Current Focus: None"

        prompt = f"""
        You are a San Diego dining concierge.
        {context_summary}
        {focus_info}
        User Input: "{user_input}"
        
        Analyze intent and extract specific subjects.
        
        Rules:
        1. **Rating**: "How is [Place]?", "Rate it", "Would I like [Place]?", "How about [Place]?".
        2. **Visit**: "Find me tacos", "Where should I go?".
        3. **Chat**: "Compare X and Y", "What is the difference?".
           - For comparisons, you MUST provide a helpful, opinionated response in the 'response' field.
        
        Return JSON ONLY:
        {{
            "intent": "rating" | "visit" | "chat",
            "place": "Name (if rating)",
            "query": "Category (if visit)",
            "subject": "Primary entity mentioned (for memory tracking)",
            "response": "Conversational answer (REQUIRED for chat)"
        }}
        """

        try:
            response = self.gemini.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            text = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            if isinstance(result, list): result = result[0]

            intent = result.get('intent')
            subject = result.get('subject')
            
            # 1. MEMORY UPDATE (Active Context Tracking)
            if subject and subject.lower() not in ['none', 'it', 'that', 'unknown']:
                details = self.get_place_details(subject)
                if details:
                    self.last_mentioned_place = details['name']
                    # SAFETY: If intent is 'chat' but we found a specific place via "How about...", 
                    # let's treat it as a rating request to show the data card.
                    if intent == 'chat' and "how about" in user_input.lower():
                        intent = 'rating'
                        result['place'] = details['name']

            # 2. ROUTING
            if intent == 'rating':
                place = result.get('place')
                if place and place.lower() in ['it', 'that', 'this', 'none'] and self.last_mentioned_place:
                    place = self.last_mentioned_place
                
                details = self.get_place_details(place)
                if details:
                    self.last_mentioned_place = details['name']
                    similar = self.find_similar_places_svd(details['name'])
                    sim_str = f" (Similar vibes: {', '.join(similar)})" if similar else ""
                    cats = details.get('categories', 'Food')
                    if isinstance(cats, list): cats = ", ".join(cats)
                    
                    return f"🤖 **Analysis:** **{details['name']}** ({cats}). Rated **{details.get('avg_rating', 'N/A')}**. {sim_str}"
                return f"I couldn't find data for **{place}**."
            
            elif intent == 'visit':
                query = result.get('query', 'food')
                recs = self.predict_visit(query)
                if recs: self.last_mentioned_place = recs[0]
                
                if not recs: return f"I searched for **{query}** but found nothing."
                return f"📍 **Top Recommendations for {query}:**\n" + "\n".join([f"• {r}" for r in recs])
            
            else:
                # SAFETY: Ensure we don't return None
                chat_response = result.get('response')
                if not chat_response:
                    if self.last_mentioned_place:
                        return f"I'm thinking about **{self.last_mentioned_place}**. What do you want to know?"
                    return "I'm listening! Ask me for a recommendation or about a specific place."
                return chat_response

        except Exception as e:
            print(f"Error: {e}")
            return "My brain briefly disconnected. Try asking again?"