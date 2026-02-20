# Gemini Scanner Enterprise

A powerful market scanner faithful to the **AMA PRO TEMA** logic.

## 🚀 Quick Start

### 1. Prererequisites
Make sure you have Python 3.9+ installed and Git.

### 2. Setup
Clone the repository (if not already local):
```bash
git clone https://github.com/sahooaiagent/Gemini_Scanner.git
cd Gemini_Scanner
```

### 3. Run Backend (FastAPI)
The backend also serves the frontend automatically.

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 4. Access the Dashboard
Once the server is running, open your browser and go to:
**[http://localhost:8000](http://localhost:8000)**

---

## 🛠 Features
- **Strict index -2 logic:** Only reports confirmed signals from the previous closed bar.
- **Adaptive TEMA:** Automatically adjusts Fast/Slow periods based on market regime and timeframe.
- **Enterprise UI:** Real-time market heatmap, ticker tape, and configurable strategy parameters.
