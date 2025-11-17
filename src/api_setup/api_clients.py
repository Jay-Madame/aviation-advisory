import requests
from datetime import datetime, timedelta
import json
from . import config

def _fetch_articles_from_guardian(query: str, start_date: datetime, end_date: datetime) -> list:
    api_key = config.GUARDIAN_API_KEY
    if not api_key:
        return []
    
    url = "https://content.guardianapis.com/search"
    
    parameters = {
        'api-key':  api_key,
        'page-size': config.PAGE_SIZE,
        'query': query,
        'from-date': start_date.strftime('%Y-%m-%d'), 
        'to-date': end_date.strftime('%Y-%m-%d'),
        'api-key': api_key,
        'show-fields': 'all',
        'order-by': 'newest'
    }

    try:
        response = requests.get(url, params=parameters, timeout=10)
        if response.status_code == 401:
            print("GuardianAPI Error: 401 Unauthorized. Check your NEWS_API_KEY.")
            return []
        if response.status_code == 429:
            print("GuardianAPI Error: 429 Rate Limit Exceeded. Wait a few minutes.")
            return []
        
        response.raise_for_status()
        data = response.json()
        
        articles = data.get('response', {}).get('results', [])
        
        articles_pulled_from_guardian = [{
            'title': art.get('webTitle', 'N/A'),
            'description': art.get('fields', {}).get('trailText', 'N/A'), 
            'content': art.get('fields', {}).get('bodyText', 'N/A'),
            'source': {'name': 'The Guardian'},
            'publishedAt': art.get('webPublicationDate')
        } for art in articles]
        
        return articles_pulled_from_guardian
        
    except requests.exceptions.RequestException as e:
        print(f"Guardian API failed: {e}")
        print("Try request again, or check authentication")
        return []
    
def _fetch_articles_from_news(query: str, start_date: datetime, end_date: datetime) -> list:
    api_key = config.NEWS_API_KEY
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"

    parameters = {
        'query': query,
        'from': start_date.isoformat(), 
        'to': end_date.isoformat(),
        'page-size': config.PAGE_SIZE,
        'language': 'en',
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=parameters, timeout=10)
        
        if response.status_code == 401:
            print("NewsAPI Error: 401 Unauthorized. Check your NEWS_API_KEY.")
            return []
        if response.status_code == 429:
            print("NewsAPI Error: 429 Rate Limit Exceeded. Wait a few minutes.")
            return []
        
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"NewsAPI failed: {e}")
        return []

def grab_articles_for_geopolitical_climate() -> list:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.TIME_WINDOW_DAYS)
    query = config.WORDS_TO_QUERY_FOR
    
    articles_to_analyze = _fetch_articles_from_guardian(query, start_date, end_date)
    
    if not articles_to_analyze:
        articles_to_analyze = _fetch_articles_from_news(query, start_date, end_date)

    if not articles_to_analyze:
        print("No news found")

    return articles_to_analyze