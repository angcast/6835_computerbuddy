from enum import Enum
import cv2
import mediapipe as mp
import pyautogui as gui

class Fingers(Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4

class LandMarkPoints(Enum):
    WRITE = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)

        return lmlist

    def isFingerUp(self, finger, landmarkList): 
        if len(landmarkList) != 0: 
            # for thumb, compare x-values
            if finger == Fingers.THUMB: 
                tip = landmarkList[LandMarkPoints.THUMB_TIP.value][1]
                ip = landmarkList[LandMarkPoints.THUMB_IP.value][1]
                return tip > ip
            # for fingers, compare y-values
            if finger == Fingers.INDEX: 
                pip = landmarkList[LandMarkPoints.INDEX_FINGER_PIP.value][2]
                tip = landmarkList[LandMarkPoints.INDEX_FINGER_TIP.value][2]
            elif finger == Fingers.MIDDLE: 
                pip = landmarkList[LandMarkPoints.MIDDLE_FINGER_PIP.value][2]
                tip = landmarkList[LandMarkPoints.MIDDLE_FINGER_TIP.value][2]
            elif finger == Fingers.PINKY:
                pip = landmarkList[LandMarkPoints.PINKY_PIP.value][2]
                tip = landmarkList[LandMarkPoints.PINKY_TIP.value][2]
            elif finger == Fingers.RING: 
                pip = landmarkList[LandMarkPoints.RING_FINGER_PIP.value][2]
                tip = landmarkList[LandMarkPoints.RING_FINGER_TIP.value][2]
            return tip < pip
        return False

    def fingersUp(self, landmarkList):
        '''
            Returns a list of the ids of the fingers that are up
        '''
        fingers = []
        for finger in Fingers: 
            isUp = self.isFingerUp(finger, landmarkList)
            if isUp: 
                fingers.append(finger)
        return fingers
    
    def fingersDown(self, landmarkList):
        fingers = [] 
        for finger in Fingers:
            isUp = self.isFingerUp(finger, landmarkList)
            if not isUp:
                fingers.append(finger)
        return fingers

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        # tracker.isPointing(lmList)
        fingersUp = tracker.fingersUp(lmList)
        fingersDown = tracker.fingersDown(lmList)
        print(fingersUp)
        # handle pointing
        if len(fingersUp) == 1 and fingersUp[0] == Fingers.INDEX: 
            gui.moveTo(lmList[LandMarkPoints.INDEX_FINGER_TIP.value][1], lmList[LandMarkPoints.INDEX_FINGER_TIP.value][2])
        cv2.imshow("Video",image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()