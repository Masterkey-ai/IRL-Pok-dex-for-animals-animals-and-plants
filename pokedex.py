import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

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
decoded=decode_predictions(predictions,top=3)[0]
print("\n--- AI Predictions ---")
for _, label, confidence in decoded:
    print(f"{label}: {confidence*100:.2f}%")