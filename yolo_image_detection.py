import cv2
import numpy as np


def findObjects(outputs, img, confThreshold, nmsThreshold, classNames, objectIndex):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        if objectIndex != None:
            if classIds[i] != objectIndex:
                continue
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            img,
            f"{classNames[classIds[i]].upper()} {int(confs[i]*100)}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )


# fmt:off
def recognize_object(imgpath, classesFilePath, modelConfigurationPath, modelWeightsPath, confThreshold=0.5, nmsThreshold=0.3, filter=False):
    img = cv2.imread(imgpath)
    wtH = 320

    classNames = []
    with open(classesFilePath, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    net = cv2.dnn.readNetFromDarknet(modelConfigurationPath, modelWeightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (wtH, wtH), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    object_index = None
    if filter:
        inp = input("Enter object name: ")
        while True:
            if inp in classNames:
                break
            else:
                inp = input("Object is not in database, type another object: ")
        object_index = classNames.index(inp)
    findObjects(outputs, img, confThreshold, nmsThreshold, classNames, object_index)

    cv2.imshow("Image", img)
    cv2.waitKey(0)


def main():
    img = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\images\street.png"
    classesFile = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\models\yolo-large\labels.txt"
    modelConfiguration = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\models\yolo-large\yolov3.cfg"
    modelWeights = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\models\yolo-large\yolov3.weights"

    recognize_object(img, classesFile, modelConfiguration, modelWeights, filter=False)


if __name__ == "__main__":
    main()
