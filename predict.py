import cv2
from ultralytics import YOLO

# Load a model
model = YOLO(f'C:/Users/Akanksha Pinto/Desktop/All desktop folders/IIST-Internship/Model1/runs/detect/train/weights/best.pt')  

# Path to the image
img_path = 'C:/Users/Akanksha Pinto/Desktop/imglelo16.png'

try:
    results = model(img_path)
    # print(results)

    img = cv2.imread(img_path)

    
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {img_path}")

    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = float(box.conf)  # Confidence score
            class_id = int(box.cls)  # Class ID

            # Put the label on the bounding box
            label = f'{model.names[class_id]}: {confidence:.2f}'

            if (class_id==0):
                cv2.rectangle(img, (x1, y1), (x2, y2), (252, 252, 55), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 252, 55), 2) #yellow
            elif (class_id==1):
                cv2.rectangle(img, (x1, y1), (x2, y2), (55, 225, 252), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 225, 252), 2) #light-blue
            elif (class_id==2):
                cv2.rectangle(img, (x1, y1), (x2, y2), (252, 94, 55), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 94, 55), 2) #orange
            elif (class_id==3):
                cv2.rectangle(img, (x1, y1), (x2, y2), (51, 44, 144), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 44, 144), 2) #navy-blue
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (140, 230, 146), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 230, 146), 2) #pista-green

            

    # Display the image with detections
    cv2.imshow('YOLOv8 Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")
