import pandas as pd
import numpy as np
import json
import os
import re
import streamlit as st
from gensim.models import Word2Vec
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    OPENAI_IMPORTED = True
except ImportError:
    OPENAI_IMPORTED = False
    print("⚠️ OpenAI not installed. Run: pip install openai")

# --- CONFIGURATION ---
CONFIG = {
    "reviews_path": "data/sandiego_reviews.parquet",
    "meta_path": "data/sandiego_meta.json",
    "w2v_path": "data/review_embedding.w2v",
    "svd_vt": "data/Vt.npy",
    "place_names": "data/place_names.npy",
    # LLM Configuration - defaults to Gemini
    "llm_provider": "gemini",  # Options: "gemini" or "openai"
    "gemini_model": "gemini-2.0-flash-exp",
    "openai_model": "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
}

# San Diego Neighborhoods (for location matching)
SD_NEIGHBORHOODS = {
    # Coastal
    'la jolla': ['la jolla', 'ucsd', 'torrey pines'],
    'pacific beach': ['pacific beach', 'pb', 'pacific bch'],
    'mission beach': ['mission beach', 'mission bay'],
    'ocean beach': ['ocean beach', 'ob'],
    'point loma': ['point loma', 'liberty station'],
    'coronado': ['coronado', 'coronado island'],
    
    # Central
    'downtown': ['downtown', 'gaslamp', 'gaslamp quarter', 'east village', 'little italy', 'marina'],
    'hillcrest': ['hillcrest', 'north park', 'university heights'],
    'north park': ['north park'],
    'south park': ['south park', 'golden hill'],
    'normal heights': ['normal heights'],
    
    # North
    'del mar': ['del mar', 'carmel valley', 'carmel mountain'],
    'solana beach': ['solana beach'],
    'encinitas': ['encinitas', 'leucadia', 'cardiff'],
    'carlsbad': ['carlsbad'],
    
    # East
    'kearny mesa': ['kearny mesa', 'convoy'],
    'mission valley': ['mission valley', 'fashion valley'],
    'la mesa': ['la mesa'],
    'el cajon': ['el cajon'],
    
    # South
    'chula vista': ['chula vista'],
    'national city': ['national city'],
    'imperial beach': ['imperial beach', 'ib']
}

# Safe secrets check for both APIs
GEMINI_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        GEMINI_AVAILABLE = True
except:
    pass

try:
    if OPENAI_IMPORTED and "OPENAI_API_KEY" in st.secrets:
        OPENAI_AVAILABLE = True
except:
    pass

class RecSysEngine:
    def __init__(self):
        print("⚙️ Initializing RecSysEngine...")
        
        # Memory for Pronoun Resolution
        self.last_mentioned_place = None 
        
        # Determine which LLM provider to use (default to Gemini)
        self.llm_provider = CONFIG.get('llm_provider', 'gemini')
        
        # Initialize Gemini
        if GEMINI_AVAILABLE and self.llm_provider == 'gemini':
            self.gemini = genai.GenerativeModel(CONFIG['gemini_model'])
            self.openai_client = None
            print(f"✅ Using Gemini API ({CONFIG['gemini_model']})")
        # Initialize OpenAI
        elif OPENAI_AVAILABLE and self.llm_provider == 'openai':
            self.gemini = None
            self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            print(f"✅ Using OpenAI API ({CONFIG['openai_model']})")
        # Fall back to Gemini if available
        elif GEMINI_AVAILABLE:
            self.gemini = genai.GenerativeModel(CONFIG['gemini_model'])
            self.openai_client = None
            print(f"✅ Using Gemini API ({CONFIG['gemini_model']}) - fallback")
        else:
            self.gemini = None
            self.openai_client = None
            print("⚠️ No LLM API configured.")

        if os.path.exists(CONFIG['w2v_path']):
            self.w2v = Word2Vec.load(CONFIG['w2v_path'])
            self.stop_words = {'place', 'spot', 'restaurant', 'food', 'double', 'stand', 'house', 'grill', 'joint', 'eatery', 'shop'}
        else:
            self.w2v = None

        # Load reviews for sentiment analysis
        try:
            if os.path.exists(CONFIG['reviews_path']):
                self.reviews_df = pd.read_parquet(CONFIG['reviews_path'])
                print(f"✅ Loaded {len(self.reviews_df):,} reviews")
            else:
                self.reviews_df = None
                print("⚠️ Reviews file not found")
        except Exception as e:
            print(f"⚠️ Could not load reviews: {e}")
            self.reviews_df = None

        try:
            self.Vt = np.load(CONFIG['svd_vt']) 
            self.place_names = np.load(CONFIG['place_names'], allow_pickle=True)
        except:
            self.Vt, self.place_names = None, None

        try:
            # Load metadata
            if os.path.exists(CONFIG['meta_path']):
                with open(CONFIG['meta_path'], 'r') as f:
                    raw_data = json.load(f)
                    self.meta_data = pd.DataFrame.from_dict(raw_data, orient='index')
            else:
                # Fallback to reviews file
                df = pd.read_parquet(CONFIG['reviews_path'])
                self.meta_data = df[['place_name', 'rating']].groupby('place_name').agg({'rating': 'mean'}).reset_index()
                self.meta_data.columns = ['name', 'avg_rating']
            
            self.meta_data['name_clean'] = self.meta_data['name'].astype(str).str.lower()
            
            # Extract neighborhood from address for location-based search
            if 'address' in self.meta_data.columns:
                self.meta_data['address_lower'] = self.meta_data['address'].astype(str).str.lower()
                self.meta_data['neighborhood'] = self.meta_data['address_lower'].apply(self._extract_neighborhood)
            else:
                self.meta_data['neighborhood'] = None
            
            if 'categories' in self.meta_data.columns:
                self.meta_data['cat_str'] = self.meta_data['categories'].astype(str).str.lower()
            elif 'category' in self.meta_data.columns:
                self.meta_data['cat_str'] = self.meta_data['category'].astype(str).str.lower()
            else:
                self.meta_data['cat_str'] = ""
                
        except Exception as e:
            print(f"Error: {e}")
            self.meta_data = pd.DataFrame()

    def normalize_text(self, text):
        return re.sub(r'\W+', '', str(text)).lower()
    
    def _extract_neighborhood(self, address):
        """Extract neighborhood from address string"""
        if not address or pd.isna(address):
            return None
        
        address_lower = str(address).lower()
        
        # Check each neighborhood and its aliases
        for neighborhood, aliases in SD_NEIGHBORHOODS.items():
            for alias in aliases:
                if alias in address_lower:
                    return neighborhood
        
        return None
    
    def _parse_location_from_query(self, query):
        """Extract location from user query"""
        query_lower = query.lower()
        
        # Check for neighborhood mentions
        for neighborhood, aliases in SD_NEIGHBORHOODS.items():
            for alias in aliases:
                if alias in query_lower:
                    return neighborhood
        
        # Check for common location phrases
        location_keywords = ['in', 'near', 'around', 'at', 'by']
        for keyword in location_keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword)
                if len(parts) > 1:
                    potential_location = parts[1].strip().split()[0:3]
                    for neighborhood, aliases in SD_NEIGHBORHOODS.items():
                        if any(alias in ' '.join(potential_location) for alias in aliases):
                            return neighborhood
        
        return None

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
            result = None
            if 'avg_rating' in matches.columns:
                result = matches.sort_values('avg_rating', ascending=False).iloc[0].to_dict()
            else:
                result = matches.iloc[0].to_dict()
            
            # Add confidence score based on review count
            if result and 'review_count' in result:
                review_count = result.get('review_count', 0)
                if review_count > 50:
                    result['confidence'] = 'High'
                elif review_count > 10:
                    result['confidence'] = 'Medium'
                else:
                    result['confidence'] = 'Low'
            else:
                result['confidence'] = 'Medium'
            
            return result
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

    def get_restaurant_reviews(self, place_name, limit=10):
        """Get recent reviews for a restaurant"""
        if self.reviews_df is None:
            return []
        
        try:
            # Find matching reviews
            mask = self.reviews_df['place_name'].str.lower().str.contains(place_name.lower(), na=False)
            reviews = self.reviews_df[mask].copy()
            
            if reviews.empty:
                return []
            
            # Sort by rating (mix of high and low) and recency if timestamp available
            if 'timestamp' in reviews.columns:
                reviews = reviews.sort_values('timestamp', ascending=False)
            
            # Get a mix of ratings
            high_rated = reviews[reviews['rating'] >= 4].head(limit // 2)
            low_rated = reviews[reviews['rating'] <= 3].head(limit // 2)
            mixed_reviews = pd.concat([high_rated, low_rated]).head(limit)
            
            return mixed_reviews[['rating', 'text']].to_dict('records') if 'text' in mixed_reviews.columns else []
        except Exception as e:
            print(f"Error getting reviews: {e}")
            return []

    def summarize_reviews(self, place_name):
        """Summarize what people are saying about a restaurant using rating info + random sample"""
        if self.reviews_df is None:
            return None
        
        try:
            # Find all matching reviews
            mask = self.reviews_df['place_name'].str.lower().str.contains(place_name.lower(), na=False)
            all_reviews = self.reviews_df[mask].copy()
            
            if all_reviews.empty:
                return None
            
            # Get overall rating statistics
            total_count = len(all_reviews)
            avg_rating = all_reviews['rating'].mean()
            positive_count = len(all_reviews[all_reviews['rating'] >= 4])
            negative_count = len(all_reviews[all_reviews['rating'] <= 2])
            
            # Take a random sample of 10 reviews for text analysis
            sample_size = min(10, total_count)
            sampled_reviews = all_reviews.sample(n=sample_size, random_state=42) if total_count > 10 else all_reviews
            
            # Get sample review texts
            positive_samples = []
            negative_samples = []
            
            if 'text' in sampled_reviews.columns:
                positive_samples = [
                    r['text'] for _, r in sampled_reviews.iterrows() 
                    if r.get('text') and pd.notna(r['text']) and r['rating'] >= 4
                ][:3]
                negative_samples = [
                    r['text'] for _, r in sampled_reviews.iterrows() 
                    if r.get('text') and pd.notna(r['text']) and r['rating'] <= 2
                ][:2]
            
            return {
                'avg_rating': round(avg_rating, 1),
                'total_reviews': total_count,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_samples': positive_samples,
                'negative_samples': negative_samples
            }
        except Exception as e:
            print(f"Error summarizing reviews: {e}")
            return None

    def predict_visit(self, query, location=None):
        query_core = query.lower()
        search_terms = {query_core}
        
        # Try to extract location from query if not provided
        if not location:
            location = self._parse_location_from_query(query)
        
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

        # Score all places first
        self.meta_data['search_score'] = self.meta_data.apply(score_place, axis=1)
        results = self.meta_data[self.meta_data['search_score'] > 0].copy()
        
        # If location specified, try location-specific results first
        if location:
            location_results = results[results['neighborhood'] == location].copy()
            if not location_results.empty:
                # Boost scores for location matches
                location_results['search_score'] = location_results['search_score'] + 10
                
                # Sort location results
                if 'avg_rating' in location_results.columns:
                    location_results = location_results.sort_values(['search_score', 'avg_rating'], ascending=[False, False])
                
                top_location_results = location_results.head(5)
                
                # Check if we have good matches (score > 5 means category/name match)
                good_matches = top_location_results[top_location_results['search_score'] > 10]
                if not good_matches.empty:
                    # We have good category/name matches in the location
                    return top_location_results['name'].tolist()
                else:
                    # No good matches in location - expand search nearby
                    print(f"⚠️ No {query_core} places found in {location}. Showing results from nearby areas.")
                    results = results.copy()  # Use all results, not just location
            else:
                print(f"⚠️ No results found in {location}. Showing results from all areas.")
        
        # Fallback or no location specified - use all results
        if results.empty and len(query_core.split()) > 1:
            return self.predict_visit(query_core.split()[-1], location)

        if 'avg_rating' in results.columns:
            results = results.sort_values(['search_score', 'avg_rating'], ascending=[False, False])
        
        return results['name'].head(5).tolist()

    def _fallback_response(self, user_input):
        """Simple rule-based response when LLM is not available"""
        user_lower = user_input.lower()
        
        # Search queries
        if any(word in user_lower for word in ['find', 'search', 'looking for', 'want', 'recommend', 'suggest']):
            # Extract query term
            keywords = ['taco', 'mexican', 'italian', 'sushi', 'burger', 'pizza', 'seafood', 'bbq', 'chinese', 'thai']
            query = next((kw for kw in keywords if kw in user_lower), 'food')
            recs = self.predict_visit(query)
            if recs:
                return f"📍 **Top recommendations for {query}:**\n" + "\n".join([f"• {r}" for r in recs])
            return "I couldn't find specific matches. Try: 'Find me tacos' or 'Show Italian restaurants'"
        
        # Rating queries
        if any(word in user_lower for word in ['how is', 'rate', 'good', 'about']):
            # Try to extract restaurant name
            words = user_input.split()
            for i in range(len(words) - 1):
                potential_name = ' '.join(words[i:i+3])
                details = self.get_place_details(potential_name)
                if details:
                    rating = details.get('avg_rating', 'N/A')
                    cats = details.get('categories', 'Food')
                    if isinstance(cats, list): cats = ", ".join(cats)
                    return f"🤖 **{details['name']}** ({cats}) is rated **{rating}/5.0**"
            return "Which restaurant are you asking about? Please be specific!"
        
        # Default
        return "I can help you find restaurants or rate specific places. Try: 'Find me tacos' or 'How is Phil's BBQ?'"

    def generate_response(self, user_input, history=[]):
        # Check if any LLM is available
        if not self.gemini and not self.openai_client:
            return "❌ **No LLM API configured.** Add GOOGLE_API_KEY or OPENAI_API_KEY to `.streamlit/secrets.toml`"
        
        # Build conversation context
        recent_messages = []
        for msg in history[-4:]:
            role = msg['role']
            content = msg['content'][:100]  # Truncate for brevity
            recent_messages.append(f"{role}: {content}")
        
        context_summary = "\n".join(recent_messages) if recent_messages else "No prior context"
        focus_info = f"Last discussed restaurant: {self.last_mentioned_place}" if self.last_mentioned_place else "No restaurant currently in focus"

        prompt = f"""You are a friendly, knowledgeable San Diego dining concierge. You love food and helping people discover great places to eat.

CONVERSATION CONTEXT:
{context_summary}

CURRENT FOCUS:
{focus_info}

USER MESSAGE: "{user_input}"

YOUR ROLE:
- Be conversational, warm, and enthusiastic about San Diego food
- Show formatted restaurant cards when explicitly asked for recommendations along with some text and details. Make the conversation natural and engaging.
- For casual questions, respond naturally without triggering a search
- Remember context from the conversation and reference previous places mentioned
- For comparisons, provide thoughtful analysis drawing from what was discussed

INTENT CLASSIFICATION:
1. **"rating"** - User wants details about a SPECIFIC restaurant
   Examples: "How is Phil's BBQ?", "Tell me about Puesto", "What do you think of Cafe Coyote?", "would I like [place]?", "how about [place]?"
   
2. **"visit"** - User wants restaurant RECOMMENDATIONS (multiple options)
   Examples: "Find me pizza", "I want tacos in La Jolla", "Where should I eat?", "Give me recs", "show me burger places"
   IMPORTANT: Extract both the cuisine/food type AND the neighborhood/location if mentioned
   
3. **"reviews"** - User wants to know what PEOPLE ARE SAYING about a restaurant
   Examples: "What are people saying about [place]?", "Is [place] good?", "What do reviews say about [place]?", "What's the word on [place]?"
   
4. **"chat"** - Everything else (casual conversation, questions, comparisons, follow-ups)
   Examples: "What's the difference?", "Which is better?", "Tell me more", "Hi", "Thanks", "What types of food do you recommend?", "give me the difference", "can it compare to", "do they pair well"
   For these, provide a NATURAL, HELPFUL response that references conversation context and previously mentioned places

COMPARISON HANDLING:
When user asks to compare (e.g., "give me the difference", "can it compare", "which is better"):
- Look at conversation history to identify what's being compared
- Reference the specific restaurants/food types discussed
- Provide thoughtful comparison of flavors, styles, occasions
- Example: If discussing "The Taco Stand" and "greasy cheeseburger", compare authentic tacos vs comfort food burgers

SIMILARITY REQUESTS:
When user asks for "similar" or "like it" in the context of restaurants:
- Use the last mentioned restaurant as reference
- Treat as a "visit" intent to show recommendations
- Extract the cuisine type from the reference restaurant

LOCATION EXTRACTION:
If user mentions a neighborhood, extract it: La Jolla, Pacific Beach, Ocean Beach, Downtown, Gaslamp, Hillcrest, North Park, Del Mar, Carmel Valley, Carmel Mountain, Coronado, Point Loma, Mission Valley, Mission Beach, Little Italy, Chula Vista, National City, UTC, University City, Clairemont, etc.

RESPONSE FORMAT (JSON ONLY):
{{
    "intent": "rating" | "visit" | "reviews" | "chat",
    "place": "Exact restaurant name if rating or reviews intent",
    "query": "Food type/cuisine if visit intent (e.g., 'pizza', 'mexican', 'seafood', 'burger', 'taco')",
    "location": "Neighborhood name if mentioned (e.g., 'La Jolla', 'Pacific Beach', 'Del Mar', 'Carmel Valley')",
    "subject": "Main entity mentioned for context tracking",
    "response": "Your conversational response for 'chat' intent - be friendly, reference conversation context, and provide helpful comparisons!"
}}

EXAMPLES:
User: "i'd like a greasy cheeseburger on my way to carmel valley"
→ {{"intent": "visit", "query": "cheeseburger", "location": "Carmel Valley"}}

User: "would i like the taco stand?"
→ {{"intent": "rating", "place": "Tacos"}}

User: "give me the difference, do you think they pair well together or do i do separate trips"
→ {{"intent": "chat", "response": "The Taco Stand and Carl's Jr. are totally different experiences! The Taco Stand serves authentic Tijuana-style tacos with fresh ingredients and bold flavors, while Carl's Jr. is your classic greasy cheeseburger comfort food. They don't really pair together - I'd say separate trips! Hit Carl's Jr. when you're craving that indulgent burger, and save The Taco Stand for when you want fresh, flavorful tacos. Both are great, just for different moods!"}}

User: "can you find me a taco place similar to it?"
→ {{"intent": "visit", "query": "taco", "subject": "The Taco Stand"}}

User: "how about robertos?"
→ {{"intent": "rating", "place": "Roberto's Taco Shop"}}

User: "can it compare to a greasy cheeseburger"
→ {{"intent": "chat", "response": "Roberto's Taco Shop and a greasy cheeseburger satisfy totally different cravings! Roberto's gives you that fresh, zesty Mexican fast food vibe - flavorful carne asada, fresh salsas, and warm tortillas. A greasy cheeseburger is pure comfort food - rich, savory, and indulgent. If you want something lighter with bold spices, go Roberto's. If you're craving that fatty, umami burger satisfaction, stick with the cheeseburger!"}}

User: "Hi, I'm hungry"
→ {{"intent": "chat", "response": "Hey there! I'd love to help you find something delicious. What kind of food are you craving today?"}}

User: "What types of food do you recommend?"
→ {{"intent": "chat", "response": "San Diego has amazing variety! We're known for fish tacos, Mexican food (we're right by the border!), fresh seafood, and great craft beer spots. What sounds good to you right now?"}}
"""

        try:
            # Use Gemini API
            if self.gemini:
                response = self.gemini.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                text = response.text.replace("```json", "").replace("```", "").strip()
            # Use OpenAI API
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=CONFIG['openai_model'],
                    messages=[
                        {"role": "system", "content": "You are a friendly San Diego restaurant recommendation assistant. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                text = response.choices[0].message.content.strip()
            
            result = json.loads(text)
            if isinstance(result, list): result = result[0]

            intent = result.get('intent')
            subject = result.get('subject')
            
            # 1. MEMORY UPDATE (Active Context Tracking)
            if subject and subject.lower() not in ['none', 'it', 'that', 'unknown', '']:
                details = self.get_place_details(subject)
                if details:
                    self.last_mentioned_place = details['name']

            # 2. ROUTING
            if intent == 'rating':
                place = result.get('place')
                if place and place.lower() in ['it', 'that', 'this', 'none'] and self.last_mentioned_place:
                    place = self.last_mentioned_place
                
                details = self.get_place_details(place)
                if details:
                    self.last_mentioned_place = details['name']
                    similar = self.find_similar_places_svd(details['name'])
                    
                    # Build comprehensive response
                    cats = details.get('categories', 'Food')
                    if isinstance(cats, list): cats = ", ".join(cats)
                    
                    rating = details.get('avg_rating', 'N/A')
                    confidence = details.get('confidence', 'Medium')
                    review_count = details.get('review_count', 0)
                    
                    # Format rating with stars
                    if rating != 'N/A':
                        stars = '⭐' * int(round(float(rating)))
                        rating_str = f"**{rating}/5.0** {stars}"
                    else:
                        rating_str = "**N/A**"
                    
                    response = f"🤖 **{details['name']}**\n\n"
                    response += f"📍 {cats}\n"
                    response += f"⭐ Rated {rating_str}"
                    
                    if review_count > 0:
                        response += f" (based on {review_count} reviews)"
                    
                    if similar:
                        response += f"\n\n✨ **Similar vibes:** {', '.join(similar[:3])}"
                    
                    return response
                return f"Hmm, I couldn't find **{place}** in my database. Could you check the spelling or try another San Diego restaurant?"
            
            elif intent == 'visit':
                query = result.get('query', 'food')
                location = result.get('location')
                
                # If query is generic and we have a last mentioned place, extract its cuisine type
                if query in ['food', 'restaurant', 'place'] and self.last_mentioned_place:
                    last_details = self.get_place_details(self.last_mentioned_place)
                    if last_details:
                        cats = last_details.get('categories', [])
                        if isinstance(cats, str):
                            cats = [cats]
                        # Extract cuisine from categories
                        for cat in cats:
                            cat_lower = cat.lower()
                            if 'taco' in cat_lower or 'mexican' in cat_lower:
                                query = 'taco'
                                break
                            elif 'burger' in cat_lower:
                                query = 'burger'
                                break
                            elif 'pizza' in cat_lower:
                                query = 'pizza'
                                break
                            elif 'italian' in cat_lower:
                                query = 'italian'
                                break
                            elif 'sushi' in cat_lower or 'japanese' in cat_lower:
                                query = 'sushi'
                                break
                
                # Normalize location
                if location:
                    location_lower = location.lower()
                    for neighborhood in SD_NEIGHBORHOODS.keys():
                        if neighborhood in location_lower or location_lower in SD_NEIGHBORHOODS[neighborhood]:
                            location = neighborhood
                            break
                
                recs = self.predict_visit(query, location)
                if recs: self.last_mentioned_place = recs[0]
                
                if not recs:
                    location_str = f" in {location.title()}" if location else ""
                    return f"I searched for **{query}**{location_str} but couldn't find any matches. Try broader terms like 'Mexican', 'Italian', 'seafood', or 'burgers'!"
                
                # Build enhanced recommendation response
                location_header = f" in {location.title()}" if location else ""
                response = f"📍 **Top Recommendations for {query}{location_header}:**\n\n"
                
                # Get details for top recommendations with ratings and samples
                for i, rec in enumerate(recs[:5], 1):
                    rec_details = self.get_place_details(rec)
                    if rec_details:
                        rating = rec_details.get('avg_rating', 'N/A')
                        stars = '⭐' * int(round(float(rating))) if rating != 'N/A' else '⭐⭐⭐'
                        
                        # Get address/neighborhood
                        address = rec_details.get('address', '')
                        neighborhood = rec_details.get('neighborhood', '')
                        location_info = f"{neighborhood.title()}" if neighborhood else (address.split(',')[-2].strip() if ',' in address else '')
                        
                        # Get sample reviews
                        reviews = self.get_restaurant_reviews(rec, limit=3)
                        review_preview = ""
                        if reviews and len(reviews) > 0:
                            sample_review = reviews[0].get('text', '')
                            if sample_review and len(sample_review) > 80:
                                review_preview = f"\n   💬 *\"{sample_review[:80]}...\"*"
                            elif sample_review:
                                review_preview = f"\n   💬 *\"{sample_review}\"*"
                        
                        response += f"**{i}. {rec}**\n"
                        response += f"   {stars} {rating}/5.0"
                        if location_info:
                            response += f" • 📍 {location_info}"
                        response += review_preview
                        response += "\n\n"
                    else:
                        response += f"**{i}. {rec}**\n\n"
                
                # Add engaging follow-up question
                response += "\n💬 Which one catches your eye? I can tell you more about any of these, or compare them if you'd like!"
                
                return response
            
            elif intent == 'reviews':
                place = result.get('place')
                if place and place.lower() in ['it', 'that', 'this', 'none'] and self.last_mentioned_place:
                    place = self.last_mentioned_place
                
                if not place:
                    return "Which restaurant would you like to know about? Just give me the name!"
                
                details = self.get_place_details(place)
                if not details:
                    return f"I couldn't find **{place}** in my database. Could you check the spelling?"
                
                self.last_mentioned_place = details['name']
                review_summary = self.summarize_reviews(details['name'])
                
                if not review_summary:
                    return f"I don't have enough review data for **{details['name']}** yet. But it has an average rating of **{details.get('avg_rating', 'N/A')}/5.0**!"
                
                # Build review summary response
                response = f"📊 **What people are saying about {details['name']}:**\n\n"
                
                rating = review_summary['avg_rating']
                stars = '⭐' * int(round(rating))
                response += f"⭐ **{rating}/5.0** {stars}\n"
                response += f"📝 Based on {review_summary['total_reviews']} recent reviews\n\n"
                
                if review_summary['positive_count'] > 0:
                    response += f"👍 **{review_summary['positive_count']} positive reviews say:**\n"
                    for i, sample in enumerate(review_summary['positive_samples'][:2], 1):
                        if sample and len(sample) > 100:
                            response += f"   • *\"{sample[:100]}...\"*\n"
                        elif sample:
                            response += f"   • *\"{sample}\"*\n"
                    response += "\n"
                
                if review_summary['negative_count'] > 0:
                    response += f"👎 **{review_summary['negative_count']} reviews mentioned concerns:**\n"
                    for i, sample in enumerate(review_summary['negative_samples'][:2], 1):
                        if sample and len(sample) > 100:
                            response += f"   • *\"{sample[:100]}...\"*\n"
                        elif sample:
                            response += f"   • *\"{sample}\"*\n"
                
                response += "\n💡 Want to know anything else about this place or need other recommendations?"
                
                return response
            
            else:  # chat intent
                chat_response = result.get('response', '').strip()
                if chat_response:
                    return chat_response
                    
                # Fallback if LLM didn't provide a response
                if self.last_mentioned_place:
                    return f"I'm here to help! We were just talking about **{self.last_mentioned_place}**. What would you like to know?"
                return "I'm your San Diego dining guide! I can help you find great places to eat or tell you about specific restaurants. What are you curious about?"

        except Exception as e:
            error_str = str(e)
            provider_name = "Gemini" if self.gemini else "OpenAI"
            print(f"Error: {e}")
            
            # Show clear error messages instead of falling back
            if "429" in error_str or "quota" in error_str.lower() or "rate_limit" in error_str.lower():
                return f"⚠️ **Rate limit reached.** The {provider_name} API needs a minute to cool down. Try again in 60 seconds!"
            elif "404" in error_str:
                model_name = CONFIG['gemini_model'] if self.gemini else CONFIG['openai_model']
                return f"❌ **Model not found.** The model `{model_name}` isn't available with your API key."
            elif "403" in error_str or "permission" in error_str.lower() or "authentication" in error_str.lower():
                return f"❌ **Permission denied.** Your {provider_name} API key might need proper permissions."
            else:
                return f"❌ **Oops!** Something went wrong with {provider_name}: {str(e)[:150]}"
