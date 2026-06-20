<<<<<<< HEAD
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
```
GROQ_API_KEY=your_groq_api_key_here
```

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

```
naturedex/
├── naturedex.py        # Main application
├── requirements.txt    # Python dependencies
├── .env                # API keys (you create this)
└── README.md           # This file
```

Your scan history is saved to ~/.naturedex_collection.json

---

## For the Congressional App Challenge

NatureDex AI transforms AI from a simple recognition tool into an educational companion.
It helps students, teachers, and families explore and understand the natural world —
making wildlife learning as exciting as discovering a new Pokémon.

*Author: Tejo Mukkamala*
=======
**AI Pokédex Scanner**
A real-life Pokédex-style scanner that uses a webcam and artificial intelligence to identify objects or animals from an image. The project is inspired by the scanning device used in the world of Pokémon and aims to recreate that experience using computer vision and machine learning.

The goal of this project is not just simple image recognition, but to build a system that:

Scans the real world

Identifies a species or object

Generates informative descriptions automatically

Displays the results in a Pokédex-style interface

Ultimately, the project will evolve into a fully interactive application that feels like using a real Pokédex.

**Project Vision**
The long-term goal is to transform this prototype into a fully interactive Pokédex-style system.

Instead of relying on a manually created database, the project will use a second AI model to automatically generate information about the detected species.

**Future workflow:**
Camera → Image Recognition AI → Prediction → Information Generation AI → Pokédex Entry

This approach removes the need for manually storing large datasets and allows the system to dynamically explain what it detects.

**Planned Features**
1. AI Information Generator

A second AI system will take the predicted object and generate:

species name

habitat

description

interesting facts

This will simulate how a Pokédex explains each creature.

2. Pokédex Interface

A custom graphical interface will be created that resembles a Pokédex device with:

scanning animation

object display

species information panel

discovery log

3. Mobile / Desktop Application

The project will eventually become a full application where users can:

scan objects in the real world

collect discovered species

view Pokédex entries

explore a discovery log

4. Improved Recognition

Future improvements may include:

wildlife recognition models

plant identification

improved accuracy

faster inference


**Inspiration**

This project is inspired by the scanning device used in the Pokémon universe, which identifies creatures and displays detailed information about them.

The goal is to recreate that experience in the real world using modern AI tools.

**Future Vision**

The final version of this project should feel like a real digital Pokédex, combining:

computer vision

artificial intelligence

interactive design

The long-term goal is to create something that feels immersive, fun, and educational while demonstrating the capabilities of modern AI and computer vision systems.
>>>>>>> 122c33360f2a2bf339b71d3a6c1263062d0fca50
