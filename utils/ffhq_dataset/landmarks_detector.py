import dlib
import cv2


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)
        #print('face bounding boxes', dets)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            #print('face landmarks', face_landmarks)
            yield face_landmarks

    def draw(img, landmarks):
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        return img


class DNNLandmarksDetector:
    def __init__(self, predictor_model_path, DNN='TF'):
        """
        :param
        DNN: "TF" or "CAFFE"
        predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        if DNN == "CAFFE":
            modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "deploy.prototxt"
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = "opencv_face_detector_uint8.pb"
            configFile = "opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def detect_faces(self, image, conf_threshold=0):
        H, W = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * W)
                y1 = int(detections[0, 0, i, 4] * H)
                x2 = int(detections[0, 0, i, 5] * W)
                y2 = int(detections[0, 0, i, 6] * H)
                bboxes.append(dlib.rectangle(x1, y1, x2, y2))
        return bboxes

    def get_landmarks(self, image):
        img = cv2.imread(image)
        dets = self.detect_faces(img, 0)
        print('face bounding boxes', dets)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            print('face landmarks', face_landmarks)
            yield face_landmarks
