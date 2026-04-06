import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

model=MobileNetV2(weights="imagenet")
cap=cv2.VideoCapture(0)
print("Press 's' to scan")
while True:
    ret,frame=cap.read()
    cv2.imshow("Pokedex Scanner",frame)
    key=cv2.waitKey(10) &0xFF
    if key==ord('s'):
        cv2.imwrite("scan.jpg",frame)
        break
cap.release()
cv2.destroyAllWindows()
img=image.load_img("scan.jpg", target_size=(224,224))
img_array=image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=preprocess_input(img_array)
predictions=model.predict(img_array)
decoded=decode_predictions(predictions,top=1)[0]
print("\n--- AI Predictions ---")
for _, label, confidence in decoded:
    print(f"{label}: {confidence*100:.2f}%")
def format_label(label):
    return label.replace("_", " ").title() 
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def get_description(label):
    prompt = f"""
    Give a Pokédex-style entry for a {label}.

    Format:
    Type: (1-2 word category)
    Description: (1-2 sentences)
    """
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    return response.output_text

label = decoded[0][1]
description = get_description(label)
name = format_label(label)

print("\n==========================")
print("📖 POKEDEX ENTRY")
print("==========================")
print(f"Name: {name}")
print(f"Confidence: {decoded[0][2]*100:.2f}%")
print("--------------------------")
print(description)
print("==========================\n")
