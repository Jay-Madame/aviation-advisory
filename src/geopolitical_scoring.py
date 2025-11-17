# Handles the geopolitical scoring 
import pandas as pd
import re
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from .api_setup import config
from .api_setup import data_storage

# the assumption that no news = good news
DEFAULT_SCORE = 1.00

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    print("Downloading NLTK resource: vader_lexicon...")
    nltk.download('vader_lexicon')
# -----------------------------------------------------------


# High tension key words to search for in the news
HIGH_TENSION_KEYWORDS = [
    # General Tension
    r'\bsanctions\b', 
    r'\bblockade\b', 
    r'\bcyber\s+attack\b', 
    r'\bmilitary\s+drill\b',
    r'\bdispute\b', 
    r'\bsabotage\b', 
    r'\bthreat\b',
    
    # Missile & Nuclear-Specific Tension
    r'\bmissile\s+crisis\b',
    r'\bnuclear\s+crisis\b',
    r'\bballistic\s+missile\b',
    r'\binter\s*continental\s+missile\b',
    r'\bICBM\b', 
    r'\bSLBM\b',
    r'\bnuclear\s+weapon\b',
    r'\bnuclear\s+warhead\b',
    r'\bnuclear\s+missile\b',
    r'\btest\s+launch\b',
    r'\blaunch\s+warning\b',
    r'\bdecapitation\s+strike\b'
]
TENSION_AMPLIFIER = 0.2


sentiment_analyzer = SentimentIntensityAnalyzer() 


def calculate_tension_score_of(article: dict, analyzer: SentimentIntensityAnalyzer) -> dict:
    text_to_analyze = article.get('title', '') + " " + article.get('description', '')
    # Using VADER, analyze the sentiment on a range of -1.0 to +1.0
    vader_score = analyzer.polarity_scores(text_to_analyze)
    base_score = vader_score['compound']

    tension_boost = 0.0
    for pattern in HIGH_TENSION_KEYWORDS:
        if re.search(pattern, text_to_analyze, re.IGNORECASE):
            # Only amplify tension if the base sentiment is already negative
            if base_score < 0:
                tension_boost += TENSION_AMPLIFIER

    # Final score: Negative base score indicates tension. We subtract the boost 
    # (making the negative score larger in magnitude) to show greater tension.
    final_score = base_score - tension_boost

    return {
        'title': article.get('title', 'N/A'),
        'source': article.get('source', {}).get('name', 'N/A'),
        'article_sentiment_score': base_score,
        'tension_keywords_found': tension_boost > 0,
        'geopolitical_tension_score': final_score
    }


def analyze_geopolitical_context_from(raw_articles: list) -> float:
    # Verification
    if not raw_articles:
        print("No articles provided for analysis. Defaulting to score 1.00.")
        return DEFAULT_SCORE
    
    analysis_results = []
    # Pass the globally initialized analyzer to the calculation function
    for article in raw_articles:
        result = calculate_tension_score_of(article, sentiment_analyzer)
        analysis_results.append(result)
    
    df = pd.DataFrame(analysis_results)
    average_score = df['geopolitical_tension_score'].mean()

    data_storage.save_analysis_report(analysis_results, {})

    return average_score