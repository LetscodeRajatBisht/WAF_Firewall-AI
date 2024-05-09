from flask import Flask, request, jsonify, abort, render_template, Response
import re
import pickle
import logging
from logging.handlers import RotatingFileHandler
import feedparser  

# Load the ML model and vectorizer
model = pickle.load(open('model_rfc.pkl', 'rb'))
vectorizer = pickle.load(open('Vect_rfc.pkl', 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Rajat_waf'  # Set a secret key for CSRF protection


# Setup logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
app.logger.addHandler(handler)

# Define patterns for SQL Injection and XSS to filter out
SQL_INJECTION_PATTERNS = [
    r"(?:')|(?:--)|(/\*(?:.|[\n\r])*?\*/)|",
    r"(;\s*shutdown\s*;)|(;\\s*exec\\s*;)",
    r"union.*select",
    r"insert\s+into.*values",
    r"select.*from",
    r"delete\s+from",
    r"update.*set"
]
XSS_PATTERNS = [
    r"<script.*?>.*?</script.*?>",
    r"javascript\s*:\s*",
    r"\bon\w+\s*=\s*\"?[^\"]*\"?",
    r"\balert\s*\(.?\)",
    r"\bwindow\.location\s*=\s*[^;]+",
    r"\bdocument\.\w+"
]

def fetch_cybersecurity_news():
    """Fetches the latest cybersecurity news from The Hacker News."""
    feed_url = 'https://feeds.feedburner.com/TheHackersNews'
    news_feed = feedparser.parse(feed_url)
    news_items = [{'title': entry.title, 'description': entry.description, 'link': entry.link} for entry in news_feed.entries[:5]]
    return news_items

@app.route('/')
def home():
    news_items = fetch_cybersecurity_news()
    return render_template('home.html', news_items=news_items)

@app.route('/predict', methods=['POST'])
def predict():
    user_query = request.form['Query']
    vectorized_query = vectorizer.transform([user_query])
    prediction = model.predict(vectorized_query)
    result = 'Malicious' if prediction[0] == 1 else 'Valid'
    return render_template('after.html', data=result, Query=user_query)

@app.before_request
def before_every_request():
    # Combine both security checks into one function for clarity
    validate_request()
    check_for_attacks()

def validate_request():
    user_agent = request.headers.get('User-Agent', '')
    if re.search(r'(sqlmap|nmap|curl)', user_agent, re.I):
        app.logger.info(f"Blocked suspicious User-Agent: {user_agent} from {request.remote_addr}")
        abort(403, description="Access denied: Suspicious User-Agent detected.")
    if request.method not in ['GET', 'POST']:
        app.logger.info(f"Blocked unexpected method: {request.method} from {request.remote_addr}")
        abort(405, description="Method Not Allowed")

def check_for_attacks():
    if request.path not in ['/', '/predict', '/static']:
        request_data = request.get_data(as_text=True)
        if any(re.search(pattern, request.path + request_data, re.IGNORECASE) for pattern in SQL_INJECTION_PATTERNS):
            abort(403, description="Request blocked by WAF - SQL Injection detected")
        if any(re.search(pattern, request.path + request_data, re.IGNORECASE) for pattern in XSS_PATTERNS):
            abort(403, description="Request blocked by WAF - XSS detected")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
