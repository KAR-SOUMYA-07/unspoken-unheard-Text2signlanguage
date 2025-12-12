from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging
import spacy
import re
import os
import sqlite3
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT NOT NULL,
            isl_text TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect('history.db')
    conn.row_factory = sqlite3.Row
    return conn

# Download required NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'taggers/{resource}')
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error(f"Error downloading {resource}: {e}")

download_nltk_resources()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error("Error loading spaCy model. Run 'python -m spacy download en_core_web_sm' to download the model.")
    raise e

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/save_history', methods=['POST'])
def save_history():
    try:
        data = request.get_json()
        if not data or 'original_text' not in data or 'isl_text' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        original_text = data['original_text']
        isl_text = data['isl_text']
        timestamp = datetime.utcnow().isoformat()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO history (original_text, isl_text, timestamp) VALUES (?, ?, ?)',
                       (original_text, isl_text, timestamp))
        conn.commit()
        inserted_id = cursor.lastrowid
        conn.close()
        
        return jsonify({
            "message": "History saved successfully",
            "id": str(inserted_id)
        }), 201
    except Exception as e:
        logger.error(f"Error saving history: {e}")
        return jsonify({"error": "Server error saving history"}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        conn = get_db_connection()
        history_entries = conn.execute('SELECT * FROM history ORDER BY timestamp DESC LIMIT 100').fetchall()
        conn.close()
        
        entries = []
        for entry in history_entries:
            entries.append({
                "_id": str(entry['id']), # Frontend expects _id as string
                "original_text": entry['original_text'],
                "isl_text": entry['isl_text'],
                "timestamp": entry['timestamp']
            })
            
        return jsonify(entries), 200
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({"error": "Server error fetching history"}), 500

@app.route('/delete_history/<string:entry_id>', methods=['DELETE'])
def delete_history(entry_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM history WHERE id = ?', (entry_id,))
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        
        if deleted_count == 0:
            return jsonify({"error": "Entry not found"}), 404
        return jsonify({"message": "Entry deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting history entry: {e}")
        return jsonify({"error": "Invalid entry ID"}), 400

@app.route('/clear_history', methods=['DELETE'])
def clear_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM history')
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        
        return jsonify({
            "message": "History cleared successfully",
            "deleted_count": deleted_count
        }), 200
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({"error": "Server error clearing history"}), 500

# Dictionary for common English contractions
CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will", "it'll": "it will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "won't": "will not", "wouldn't": "would not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "mightn't": "might not", "mustn't": "must not"
}

def expand_contractions(text):
    """Expands contractions in the given text while preserving capitalization."""
    contractions_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS.keys()) + r')\b', re.IGNORECASE)
    
    def replace(match):
        word = match.group(0).lower()  # Match word case-insensitively
        expanded = CONTRACTIONS.get(word, word)  # Get expanded version
        return expanded.capitalize() if match.group(0)[0].isupper() else expanded  # Preserve capitalization

    return contractions_pattern.sub(replace, text)

def preprocess_text(text):
    """Enhanced preprocessing for ISL conversion with contraction handling and time format preservation."""
    text = text.lower().strip()
    
    # Expand contractions
    text = expand_contractions(text)

    # Convert time format (e.g., "10:30" → "10 30", "7:00" → "7")
    text = re.sub(r'\b(\d{1,2}):00\b', r'\1', text)  # Remove ':00'
    text = re.sub(r'\b(\d{1,2}):(\d{1,2})\b', r'\1 \2', text)  # Replace ':' with space

    # Remove all other non-alphanumeric characters except spaces
    text = re.sub(r'[^\w\s]', '', text)

    return text

def extract_isl_structure_spacy(text):
    """
    Converts each word in the input sentence to its base form and removes unnecessary words.
    Also identifies tense markers for ISL representation and preserves directional words.
    """
    doc = nlp(text)
    important_words = []
    tense_marker = ""

    # List of direction-related words that should not be lemmatized
    keep_words = {"left", "right", "back", "straight", "forward", "up", "down", "near", "next", "beside", "in", "on", "under", "from", "to"}

    for token in doc:
        # Preserve direction words as they are
        if token.text.lower() in keep_words:
            important_words.append(token.text.lower())
            continue

        # Remove auxiliary verbs
        if token.pos_ in ["AUX"] and token.lemma_ in ["be", "do", "have", "will"]:
            if token.lemma_ == "will":
                tense_marker = "FUTURE"
            continue 
        
        if token.pos_ in ["DET", "ADP"]:  # Remove determiners and prepositions
            continue
        
        if token.tag_ in ["VBD", "VBN"]:  # Past tense verbs
            tense_marker = "PAST"
            important_words.append(token.lemma_)
        elif token.tag_ in ["VBG", "VBZ", "VBP"]:  # Present continuous/simple present
            important_words.append(token.lemma_)
        else:
            important_words.append(token.lemma_)

    # Add the tense marker at the end if applicable
    if tense_marker:
        important_words.append(tense_marker)

    return " ".join(important_words) if important_words else text

@app.route('/save_text', methods=['POST'])
def save_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"isl_structure": "", "message": "No data provided", "error": True}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"isl_structure": "", "message": "No text provided", "error": True}), 400
        
        logger.info(f"Received text: {text}")
        processed_text = preprocess_text(text)
        isl_structure = extract_isl_structure_spacy(processed_text)
        logger.debug(f"spaCy extraction: {isl_structure}")
        
        return jsonify({
            "isl_structure": isl_structure,
            "original_text": text,
            "message": "Text processed successfully",
            "error": False
        })
    except Exception as e:
        logger.error(f"Error in save_text: {e}")
        return jsonify({"isl_structure": "", "message": "Server error occurred", "error": True}), 500

if __name__ == '__main__':
    app.run(debug=True)

