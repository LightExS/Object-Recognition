import cv2

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = "models/model1/object_detection_classes_coco.txt"
with open(classFile, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")

configPath = "models/model1/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "models/model1/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

print(class_names[0])

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            try:
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(
                    img,
                    class_names[classId - 1] + " {:.2f}".format(confidence),
                    (box[0], box[1] - 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0),
                    2,
                )
            except:
                print(classId - 1)
                break

    cv2.imshow("Output", img)
    cv2.waitKey(1)
