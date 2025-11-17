# config.py

import os
from dotenv import load_dotenv

load_dotenv()

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY") 
NEWS_API_KEY = os.getenv("NEWS_API_KEY") 

# --- System Configuration ---
# News Fetching Parameters
API_KEY_WORDS_TO_SEARCH_FOR = [
    'sanctions', 'blockade', 'cyber attack', 'military drill', 'dispute', 'sabotage', 'threat', 
    'missile crisis', 'nuclear crisis', 'ballistic missile', 
    'inter continental missile', 'ICBM', 'SLBM', 
    'nuclear weapon', 'nuclear warhead', 'nuclear missile', 
    'test launch', 'launch warning', 'decapitation strike',
    'nuclear', 'missile'
]

WORDS_TO_QUERY_FOR = " OR ".join(API_KEY_WORDS_TO_SEARCH_FOR)
PAGE_SIZE = 50
TIME_WINDOW_DAYS = 3
FGSM_EPSILON = 0.5
SIM_SEQUENCE_LENGTH = 10
SIM_TRAINING_EPOCHS = 5