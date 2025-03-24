from flask import Flask, jsonify, request
from flask_cors import CORS
from news_api import get_curated_news_feed
import sqlite3
import hashlib
import jwt
import datetime
import os

app = Flask(__name__)
CORS(app)

# Secret key for JWT
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Generate JWT token
def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

@app.route('/auth/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400

    hashed_password = hash_password(password)

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                 (name, email, hashed_password))
        conn.commit()
        user_id = c.lastrowid
        
        # Generate token
        token = generate_token(user_id)
        
        # Get user data
        c.execute('SELECT id, name, email, created_at FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()

        return jsonify({
            'token': token,
            'user': {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'created_at': user[3]
            }
        }), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already registered'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'error': 'Missing required fields'}), 400

    hashed_password = hash_password(password)

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, name, email, created_at FROM users WHERE email = ? AND password = ?',
                 (email, hashed_password))
        user = c.fetchone()
        conn.close()

        if user:
            token = generate_token(user[0])
            return jsonify({
                'token': token,
                'user': {
                    'id': user[0],
                    'name': user[1],
                    'email': user[2],
                    'created_at': user[3]
                }
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/news-feed', methods=['GET'])
def get_news_feed():
    news_feed = get_curated_news_feed()
    return jsonify({
        "status": "success",
        "count": len(news_feed),
        "articles": news_feed
    })

# Initialize database on startup
init_db()

if __name__ == '__main__':
    app.run(debug=True, port=5000)