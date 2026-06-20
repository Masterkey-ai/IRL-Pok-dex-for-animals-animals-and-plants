# NatureDex AI 🌿
### An AI-Powered Wildlife Identification and Learning Platform
*Congressional App Challenge Entry*

---

## Quick Start

### 1. Install dependencies
```bash
pip install opencv-python numpy tensorflow pillow python-dotenv openai PyQt6
```

### 2. Set up your API key
Create a `.env` file in this folder:
fda
cd /Users/tejo/Desktop/NatureDex
cat > README.md << 'EOF'
# NatureDex AI 🌿
### An AI-Powered Wildlife Identification and Learning Platform
*Congressional App Challenge Entry*

---

## Quick Start

### 1. Install dependencies
```bash
pip install opencv-python numpy tensorflow pillow python-dotenv openai PyQt6
```

### 2. Set up your API key
Create a `.env` file in this folder:
GROQ_API_KEY=your_groq_api_key_here

### 3. Run the app
```bash
python naturedex.py
```

---

## How to Use

1. **Point** your webcam at any plant, animal, insect, or object
2. **Click SCAN** (or press Enter)
3. **Watch** the AI identify and analyze what it sees
4. **Read** the full NatureDex entry — habitat, diet, conservation status, NC context, and more
5. **Ask** follow-up questions in the "ASK AI" tab
6. **Browse** your discovery collection in the left sidebar

---

## Features

| Feature | Status |
|---|---|
| Live webcam feed | ✅ |
| Object/wildlife identification (MobileNetV2) | ✅ |
| AI-generated structured entries (Groq LLM) | ✅ |
| Confidence scores + alternatives | ✅ |
| Habitat, diet, behavior, conservation status | ✅ |
| North Carolina local context | ✅ |
| Discovery collection (saved to disk) | ✅ |
| AI wildlife educator chat | ✅ |
| Pokédex-style scan animation | ✅ |
| Full GUI (PyQt6) | ✅ |

---

## Tech Stack

- **Python** — core language
- **OpenCV** — camera capture
- **TensorFlow / Keras / MobileNetV2** — image classification
- **Groq API (LLaMA 3.3 70B)** — educational entry generation + chat
- **PyQt6** — full desktop GUI

---

## Project Structure
naturedex/

├── naturedex.py        # Main application

├── requirements.txt    # Python dependencies

├── .env                # API keys (you create this)

└── README.md           # This file

Your scan history is saved to ~/.naturedex_collection.json

---

## For the Congressional App Challenge

NatureDex AI transforms AI from a simple recognition tool into an educational companion.
It helps students, teachers, and families explore and understand the natural world —
making wildlife learning as exciting as discovering a new Pokémon.

*Author: Tejo Mukkamala*
