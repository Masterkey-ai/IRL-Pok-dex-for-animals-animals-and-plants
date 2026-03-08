**IRL-Pok-dex-for-animals-animals-and-plants**
Real-Life Pokédex Scanner

A computer vision project inspired by the Pokédex from Pokémon that uses AI to identify real-world animals and plants using your webcam.

**Project Overview**

This project is the first stage of building a real-life Pokédex.

The program opens your computer's camera, lets you capture an image, and then uses a pretrained neural network to analyze the image and predict what object or species it might be.

The goal of the full project is to eventually create a real-world scanning device that can identify wildlife similar to how a Pokédex identifies Pokémon.

**Current Features (Stage 1)**

Opens your computer's webcam

Displays a live camera feed

Press S to scan/capture an image

Saves the captured image

Uses an AI image classification model to analyze the image

Returns the top predictions for what the object might be

**Example output:**

--- AI Predictions --- jay: 87.2% magpie: 6.4% crow: 3.1% How It Works

The system works in four main steps:

Camera Capture

The program uses OpenCV to access the webcam.

A live video feed is displayed in a window.

Image Capture

When the user presses S, the current frame from the camera is saved as scan.jpg.

Image Preprocessing

The image is resized and formatted so it can be processed by the neural network.

AI Classification

The image is passed into a pretrained MobileNetV2 image classification model.

The model predicts what object it sees and returns the most likely matches.

**Technologies Used**

Python

OpenCV

TensorFlow

NumPy

PIL (Python Imaging Library)

**Installation**

Install required libraries:

pip install tensorflow opencv-python pillow numpy Running the Program Run the main script:

python pokedex.py

Then:

A camera window will open.

Press S to scan.

The image will be captured.

The AI model will analyze the image and print predictions in the terminal.

Project Structure pokedex_project/ │ ├── pokedex.py # Main scanner program ├── scan.jpg # Captured image from webcam └── README.md # Project documentation Future Goals

Pokédex-style graphical interface

Animated scan effect

Create on App for on the go scanning

Species database with detailed entries

Ability to save "discovered" species

Real-time object detection instead of single image scans

**Inspiration**

Inspired by the Pokédex device from the Pokémon series.

This project explores how computer vision and machine learning can be used to build real-world tools that mimic fictional technology.
