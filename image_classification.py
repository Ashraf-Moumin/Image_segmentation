from transformers import AutoImageProcessor, TFResNetForImageClassification
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')

list_for_truck = {
    555: "fire engine, fire truck", 
    569: "garbage truck, dustcart",
    717: "pickup, pickup truck",
    864: "tow truck, tow car, wrecker",
    866: "tractor",
    867: "truck",
   }

def classify(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = TFResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    inputs = processor(image, return_tensors="tf")

    result = np.array(model(inputs).logits[0])
    
    plt.axis("off")
   
   

    if np.argmax(result) in list(list_for_truck.keys()):
        plt.title(f"Class: {list_for_truck[np.argmax(result)]}")

    else:
        plt.title("Class: Not truck-like")
    
    cv2.putText(image, "Ashraf's Computer Vision Studio", (100,550),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    plt.imshow(image)