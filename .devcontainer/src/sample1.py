from imageai.Detection import ObjectDetection
import os
import pafy # YouTube video capture
import cv2 # OpenCV
import urllib.request # for ThingSpeak service

# set your video URL
# 渋谷スクランブル交差点
#videoURL = 'https://www.youtube.com/watch?v=itwiZmuY6Ls'
videoURL = 'https://www.youtube.com/watch?v=UuTy56M29qs'

# This parameter is used to determine the integrity of the detection results.
min_probability=50

execution_path = os.getcwd()
video_pafy = pafy.new(videoURL)
video_from_url = video_pafy.getbest().url
cap = cv2.VideoCapture(video_from_url)
ret, frame = cap.read()
cv2.imwrite(os.path.join(execution_path , "cap.jpg"),frame)
cap.release()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "cap.jpg"), output_image_path=os.path.join(execution_path , "capnew.jpg"), minimum_percentage_probability=min_probability)

person = 0
vehicle = 0

for eachObject in detections:
    # print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    person = person + (eachObject["name"]=="person")
    vehicle = vehicle + (eachObject["name"]=="car") + (eachObject["name"]=="bus")+ (eachObject["name"]=="truck")

print("person: ",person)
print("vehicle: ",vehicle)