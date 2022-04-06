from enum import Enum
import cv2
import numpy as np
import mediapipe as mp
import pyautogui as gui

widthCam, heightCam = 640, 480
frameReduction = 200 
widthScreen, heightScreen = gui.size()

feedbackFontSize = 2 
feedbackFontFace = cv2.FONT_HERSHEY_DUPLEX
feedbackColor = (5, 15, 128)
feedbackThickness = 3


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
                print("h w ", h, w)
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

    def isScrollingUpGesture(self, fingersUp):
        if len(fingersUp) == 2: 
            return Fingers.INDEX in fingersUp and Fingers.MIDDLE in fingersUp
        elif len(fingersUp) == 3: # include thumb
            return Fingers.INDEX in fingersUp and Fingers.MIDDLE in fingersUp and Fingers.THUMB in fingersUp
        return False

    def isScrollingDownGesture(self, fingersDown):
        if len(fingersDown) == 2: 
            return Fingers.INDEX in fingersDown and Fingers.MIDDLE in fingersDown
        elif len(fingersDown) == 3: # include thumb
            return Fingers.INDEX in fingersDown and Fingers.MIDDLE in fingersDown and Fingers.THUMB in fingersDown
        return False
    
    def isPointingGesture(self, fingersUp):
        # ignore thumb
        if Fingers.THUMB in fingersUp:
            fingersUp.remove(Fingers.THUMB)
        return len(fingersUp) == 1 and fingersUp[0] == Fingers.INDEX

    def getPointingScreenCoordinates(self, x, y): 
        """
        Maps the video cam coordinates to that of the current screen
        """
        # Since OpenCV does not detect finger in some x and y values making it
        # harder to point downward and side to side, we reduce the frame to make 
        # these cases easier
        yFrameReduction = 200
        xFrameReduction = 100
        new_x = np.interp(x, (xFrameReduction, widthCam-xFrameReduction), (0, widthScreen))
        new_y = np.interp(y, (yFrameReduction, heightCam-yFrameReduction), (0, heightScreen))
        return new_x, new_y

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, widthCam)
    cap.set(4, heightCam)
    tracker = handTracker()

    while True:
        success, image = cap.read()
        if success:
            image = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)
            image = cv2.flip(image, 1)
            # tracker.isPointing(lmList)
            fingersUp = tracker.fingersUp(lmList)
            print("fingersUp", fingersUp)
            fingersDown = tracker.fingersDown(lmList)
            # handle pointing
            if tracker.isPointingGesture(fingersUp):
                cam_x = lmList[LandMarkPoints.INDEX_FINGER_TIP.value][1]
                cam_y = lmList[LandMarkPoints.INDEX_FINGER_TIP.value][2]
                x, y = tracker.getPointingScreenCoordinates(cam_x, cam_y)
                cv2.putText(image, "moving cursor", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                gui.moveTo(widthScreen - x, y)
            #handle scrolling 
            # elif len(fingersUp) == 2 and Fingers.INDEX in fingersUp and Fingers.MIDDLE in fingersUp: 
            #     gui.scroll(1)
            # elif len(fingersDown) == 2 and Fingers.INDEX in fingersDown and Fingers.MIDDLE in fingersDown:
            #     gui.scroll(-1)
            elif tracker.isScrollingUpGesture(fingersUp):
                print("Scrolling up...")
                gui.scroll(5)
            elif tracker.isScrollingDownGesture(fingersDown):
                print("Scrolling down...")
                gui.scroll(-5)
            cv2.imshow('MediaPipe Hands', image)
            # cv2.imshow("Video",image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()