import cv2

img = cv2.imread("images/street.png")

class_names = []
classfile = "models/model1/object_detection_classes_coco.txt"
with open(classfile, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")

inp = input("Enter object name: ")
while True:
    if inp in class_names:
        break
    else:
        inp = input("Object is not in database, type another object: ")

object_index = class_names.index(inp) + 1
config_file = "models/model1/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "models/model1/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(frozen_model, config_file)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.5)


for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    if classId != object_index:
        continue
    cv2.rectangle(img, box, color=(255, 0, 0), thickness=2)
    cv2.putText(
        img,
        class_names[classId - 1] + " {:.2f}".format(confidence),
        (box[0], box[1] - 15),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 0),
        2,
    )
cv2.imshow("Output", img)
cv2.waitKey(0)
