__AI Pokédex__

An AI-powered Pokédex that uses real-time image recognition and large language models to identify objects and generate dynamic, Pokémon-style descriptions.

**Overview**

This project combines computer vision and AI language models to create a Pokédex-like experience for real-world objects.

Instead of relying on a static database, this app dynamically:

Captures an image using your webcam

Identifies the object using a pretrained deep learning model

Generates a Pokédex-style entry using AI

 **Features**
 
 Real-time camera capture
 Image recognition (MobileNetV2)
 AI-generated descriptions (Groq LLM)
 Pokédex-style formatted output
 Fully dynamic — works for any detectable object
 
**How It Works**

Camera → Image → MobileNetV2 → Label → Groq LLM → Pokédex Entry
Step-by-step:
Press 's' to scan an object
The image is saved and processed
The model predicts the object label
A language model generates:
Type
Description
Output is displayed in Pokédex format

**Tech Stack**
Python
OpenCV – camera input
TensorFlow / Keras – image classification
MobileNetV2 – pretrained model (ImageNet)
Groq API – fast LLM inference
NumPy – data processing

**Setup**
1. Clone the repo
git clone [https://github.com/Masterkey-ai/IRL-Pok-dex-for-animals-animals-and-plants]
cd ai-pokedex
2. Install dependencies
pip install opencv-python numpy tensorflow pillow python-dotenv openai
3. Create a .env file
GROQ_API_KEY=your_api_key_here
4. Run the app
python pokedex.py
📌 Example Output
📖 POKEDEX ENTRY
Name: Ping Pong Ball
Confidence: 95.12%

Type: Sports Object
Description: A lightweight spherical object used in table tennis, known for its speed and bounce.

**Design Philosophy**

Unlike traditional apps that rely on predefined datasets, this project uses AI as a dynamic knowledge engine.

**Why this matters:**
No need to manually build a database
Scales to virtually any object
More flexible and intelligent

 **Future Improvements**

This is just the beginning. Planned upgrades include:

**GUI & UX**
Full Pokédex-style graphical interface
Embedded live camera feed inside the app
**“Scan Again”** button (no restart needed)
Improved layout and animations
**Visual Experience**
Split-screen Pokédex design
Custom fonts and colors inspired by Pokémon
Smooth transitions and effects
**Expansion**
Convert into a desktop app
Potential mobile version
Persistent “Pokédex collection” (scan history)
AI Enhancements
Better classification models
More detailed and structured entries
Add abilities, traits, or fun facts
**Inspiration**

Inspired by the classic Pokédex from Pokémon I especially love rotomdex from Pokemon sun and moon 
— reimagined using modern AI.

**Author**

Tejo Mukkamala
High school student exploring AI, robotics, and software development.
