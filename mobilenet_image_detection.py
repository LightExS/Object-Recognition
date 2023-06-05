import cv2


def recognize_object(
    imgpath,
    classesFilePath,
    modelConfigurationPath,
    modelWeightsPath,
    confThreshold=0.5,
    filter=False,
):
    img = cv2.imread(imgpath)

    class_names = []
    with open(classesFilePath, "rt") as f:
        class_names = f.read().rstrip("\n").split("\n")
    object_index = None
    if filter:
        inp = input("Enter object name: ")
        while True:
            if inp in class_names:
                break
            else:
                inp = input("Object is not in database, type another object: ")

        object_index = class_names.index(inp) + 1

    net = cv2.dnn_DetectionModel(modelWeightsPath, modelConfigurationPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold)

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if filter:
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


def main():
    img = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\images\street.png"
    classesFile = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\models\MobileNetv3\object_detection_classes_coco.txt"
    modelConfiguration = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\models\MobileNetv3\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozenFile = r"C:\Users\Dmytro\Desktop\VS Code\Object Recognition\models\MobileNetv3\frozen_inference_graph.pb"

    recognize_object(img, classesFile, modelConfiguration, frozenFile, filter=False)


if __name__ == "__main__":
    main()
