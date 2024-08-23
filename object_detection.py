from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch 


model = YOLO('yolov8n.pt')


#List of classes (objects) available
class_list = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
}



def analyze(image_path):
    #cap = cv2.VideoCapture(image_path)
    img = cv2.imread(image_path)
    while True:
        #success,img = cap.read()
        result = model(img)

    
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])
                
            if cls in list(range(0,13)):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 =  int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
                    
                confidence  = round(float(box.conf[0]), ndigits=3)
                cv2.putText(img, f'{class_list[cls]} {confidence}',(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                cv2.putText(img, "Ashraf's Computer Vision Studio", (0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        cv2.imshow("Image",img)
     

def analyze_image(image_path):
    initial_img = plt.imread(image_path)
    img = cv2.imread(image_path)

    result = model(img)

    boxes = result[0].boxes
    
    for box in boxes:
        cls = int(box.cls[0])
                
        if cls in list(range(0,13)):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 =  int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
                    
            confidence  = round(float(box.conf[0])*100, ndigits=3)
            cv2.putText(img, f'{class_list[cls]} {confidence}%',(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
            cv2.putText(img, "Ashraf's Computer Vision Studio", (0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    fig, ax = plt.subplots(1,2)
    
    fig.set_size_inches(9,5.31)
    ax[1].set_title('YOLO-processed Image')
    ax[1].axis("off")     
    ax[1].imshow(img)
    
    ax[0].set_title('Initial Image')
    ax[0].axis('off') 
    ax[0].imshow(initial_img)
    
    

     
