import json
import os
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'reports')

def save_analysis_report(analysis_results: list, metadata: dict):
    try:
        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
    except OSError as e:
        print(f"Error creating reports directory {REPORTS_DIR}: {e}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_data = {
        "timestamp": timestamp,
        "articles_analyzed_count": len(analysis_results),
        "analysis_results": analysis_results,
        "metadata": metadata
    }

    # 3. Define the file path
    filename = f"geopolitical_report_{timestamp}.json"
    file_path = os.path.join(REPORTS_DIR, filename)

    # 4. Write the JSON data to disk
    try:
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        print(f"Analysis report saved to {file_path}")
    except IOError as e:
        print(f"Error writing report file: {e}")

# --- Utility Function (Optional, but good practice) ---

def load_latest_report():
    """Loads the most recent report from the reports directory."""
    try:
        all_files = os.listdir(REPORTS_DIR)
        report_files = [f for f in all_files if f.endswith('.json')]
        
        if not report_files:
            return None
            
        # Sort files to find the latest one based on timestamp in the filename
        report_files.sort(reverse=True)
        latest_file = os.path.join(REPORTS_DIR, report_files[0])
        
        with open(latest_file, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"Error loading latest report: {e}")
        return None