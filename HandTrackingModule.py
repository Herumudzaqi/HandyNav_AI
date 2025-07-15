import cv2
import mediapipe as mp
import math
import numpy as np

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, smooth_factor=0.0):

        self.mode = mode  
        self.maxHands = maxHands  
        self.detectionCon = detectionCon  
        self.trackCon = trackCon  
        self.smooth_factor = smooth_factor  

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils  
        self.tipIds = [4, 8, 12, 16, 20]  
        self.lmList = []  
        self.prevLmList = None  
        self.leftHandLandmarks = None  

    def lerp(self, a, b, t):
        
        return a + (b - a) * t

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        self.results = self.hands.process(imgRGB)  
        self.leftHandLandmarks = None  

        if self.results.multi_hand_landmarks:
            img_w = img.shape[1]
            min_x = float('inf')
            for handLms in self.results.multi_hand_landmarks:
                wrist_x = handLms.landmark[self.mpHands.HandLandmark.WRIST].x * img_w
                if wrist_x < min_x:
                    min_x = wrist_x
                    self.leftHandLandmarks = handLms
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=False):
        xList, yList = [], []
        bbox = []

        if not self.leftHandLandmarks:
            return self.lmList, []

        h, w, _ = img.shape
        currentLmList = [
            [id, int(lm.x * w), int(lm.y * h)]
            for id, lm in enumerate(self.leftHandLandmarks.landmark)
        ]

        if self.prevLmList and len(self.prevLmList) == len(currentLmList) and self.smooth_factor > 0:
            self.lmList = [
                [id,
                 int(self.lerp(prev[1], curr[1], 1 - self.smooth_factor)),
                 int(self.lerp(prev[2], curr[2], 1 - self.smooth_factor))]
                for id, prev, curr in zip(range(len(currentLmList)), self.prevLmList, currentLmList)
            ]
        else:
            self.lmList = currentLmList

        self.prevLmList = currentLmList
#
        for _, cx, cy in self.lmList:
            xList.append(cx)
            yList.append(cy)

        if xList and yList:
            bbox = (min(xList), min(yList), max(xList), max(yList))

        return self.lmList, bbox

    def fingersUp(self):
        if len(self.lmList) < 21:
            return []  

        fingers = []

        fingers.append(1 if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1] else 0)

        for id in range(1, 5):
            fingers.append(1 if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2] else 0)

        return fingers  

    def findDistance(self, p1, p2, img, draw=False, r=15, t=3):
        if len(self.lmList) <= max(p1, p2):
            return 0, img, []  

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)  
        return length, img, [x1, y1, x2, y2, cx, cy]
