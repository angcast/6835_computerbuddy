from enum import Enum
import math
from operator import index # need python 3.8
import cv2
import numpy as np
import mediapipe as mp
import pyautogui as gui

widthCam, heightCam = 640, 480
frameReduction = 200 
widthScreen, heightScreen = gui.size()
windowSize = 10

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

class HandTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.joint_list = [
            { 'IP': [2, 3, 4], 'MCP': [0, 2, 3], 'CMC': [0, 1, 2] },
            { 'DIP': [6, 7, 8], 'PIP': [5, 6, 7], 'MCP': [0, 5, 6] },
            { 'DIP': [10, 11, 12], 'PIP': [9, 10, 11], 'MCP': [0, 9, 10] }, 
            { 'DIP': [14, 15, 16], 'PIP': [13, 14, 15], 'MCP': [0, 13, 14] }, 
            { 'DIP': [18, 19, 20], 'PIP': [17, 18, 19], 'MCP': [0, 17, 18] }
        ]

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
                # print("h w ", h, w)
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
        index_res = self.compute_finger_joint_angle(Fingers.INDEX, "PIP")
        middle_res = self.compute_finger_joint_angle(Fingers.MIDDLE, "PIP")
        ring_res = self.compute_finger_joint_angle(Fingers.RING, "PIP")
        pinky_res = self.compute_finger_joint_angle(Fingers.PINKY, "PIP")
        if index_res['joint'] is not None:
            is_index_straight, is_middle_straight = index_res['angle'] >= 160, middle_res['angle'] >= 160
            is_ring_closed, is_pinky_closed = ring_res['angle'] <= 70, pinky_res['angle'] <= 70
            is_scrolling_gesture = all([is_index_straight, is_middle_straight, is_ring_closed, is_pinky_closed])
            if is_scrolling_gesture:
                if len(fingersUp) == 2:
                    return Fingers.INDEX in fingersUp and Fingers.MIDDLE in fingersUp
                elif is_scrolling_gesture and len(fingersUp) == 3: # include thumb
                    return Fingers.INDEX in fingersUp and Fingers.MIDDLE in fingersUp and Fingers.THUMB in fingersUp
        return False

    def isScrollingDownGesture(self, fingersDown):
        index_res = self.compute_finger_joint_angle(Fingers.INDEX, "PIP")
        middle_res = self.compute_finger_joint_angle(Fingers.MIDDLE, "PIP")
        ring_res = self.compute_finger_joint_angle(Fingers.RING, "PIP")
        pinky_res = self.compute_finger_joint_angle(Fingers.PINKY, "PIP")
        # TODO: modularize OPEN/CLOSED/STRAIGHT/BENT classification
        #  with THUMB finger as an exception
        if ring_res['joint'] is not None:
            is_index_straight, is_middle_straight = index_res['angle'] >= 160, middle_res['angle'] >= 160
            is_ring_closed, is_pinky_closed = ring_res['angle'] <= 70, pinky_res['angle'] <= 70
            is_scrolling_gesture = all([is_index_straight, is_middle_straight, is_ring_closed, is_pinky_closed])
            if is_scrolling_gesture:
                if len(fingersDown) == 2: 
                    return Fingers.INDEX in fingersDown and Fingers.MIDDLE in fingersDown
                elif len(fingersDown) == 3: # include thumb
                    return Fingers.INDEX in fingersDown and Fingers.MIDDLE in fingersDown and Fingers.THUMB in fingersDown
        return False
    
    def isPointingGesture(self, fingersUp):
        index_res = self.compute_finger_joint_angle(Fingers.INDEX, "PIP")
        if index_res['joint'] is not None:
            is_index_straight = index_res['angle'] >= 175
            if Fingers.THUMB in fingersUp:
                fingersUp.remove(Fingers.THUMB)
            return len(fingersUp) == 1 and fingersUp[0] == Fingers.INDEX and is_index_straight
        return False

    def isClickingGesture(self, landmarks):
        if len(landmarks) != 0:
            index_res = self.compute_finger_joint_angle(Fingers.INDEX, "PIP")
            index_tip = landmarks[LandMarkPoints.INDEX_FINGER_TIP.value][1:]
            thumb_tip = landmarks[LandMarkPoints.THUMB_TIP.value][1:]
            thumb_ip = landmarks[LandMarkPoints.THUMB_IP.value][1:]
            clickingThreshold = 30
            # index tip connecting with thumb tip
            if index_res['joint'] is not None:
                is_index_bent, is_index_open = index_res['angle'] < 175, index_res['angle'] >= 90
                is_index_clicking = is_index_bent and is_index_open
                if is_index_clicking and math.dist(index_tip, thumb_tip) <= clickingThreshold:
                    return True
                # index tip connecting with thumb ip
                if is_index_clicking and math.dist(index_tip, thumb_ip) <= clickingThreshold: 
                    return True
        return False
    
    def isGrabbing(self, fingersDown):
        #ignore thumb
        if Fingers.THUMB in fingersDown: 
            fingersDown.remove(Fingers.THUMB)
        return len(fingersDown) == 4

    def isDropping(self, fingersUp):
        #ignore thumb
        if Fingers.THUMB in fingersUp: 
            fingersUp.remove(Fingers.THUMB)
        return len(fingersUp) == 4

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
    
    def compute_finger_joint_angle(self, finger, joint):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            finger_joints = self.joint_list[finger.value]
            joint_seg = finger_joints[joint]
            # Coordinates of the three joints of a finger
            p1 = np.array([hand.landmark[joint_seg[0]].x, hand.landmark[joint_seg[0]].y])
            p2 = np.array([hand.landmark[joint_seg[1]].x, hand.landmark[joint_seg[1]].y])
            p3 = np.array([hand.landmark[joint_seg[2]].x, hand.landmark[joint_seg[2]].y])
            finger_angle = self.compute_joint_angle(p1, p2, p3)
            drawing_pos = np.array([1-hand.landmark[joint_seg[1]].x, hand.landmark[joint_seg[1]].y])
            return { 'joint': joint, 'angle': finger_angle, 'pos': drawing_pos }
        return { 'joint': None, 'angle': None, 'pos': None }
    
    def compute_joint_angle(self, p1, p2, p3):
        # radian_angle = np.arctan2(np.linalg.norm(np.cross(p1-p2, p3-p2)), np.dot(p1-p2, p3-p2))
        radian_angle = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
        angle = np.abs(radian_angle*180.0/np.pi)

        if angle > 180.0:
            angle = 360-angle
        return angle

    def draw_joint_angle(self, image, joint, angle, drawing_pos, font_size=None):
        font_size = 1.25
        if font_size == "S":
            font_size = 0.5
        cv2.putText(image, "{joint}: {angle:.2f}".format(joint=joint, angle=angle), tuple(np.multiply(drawing_pos, [widthCam, heightCam]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 1, cv2.LINE_AA)

    def draw_all_joint_angles(self, image):
        for finger in Fingers:
            for joint in self.joint_list[finger.value]:
                res = self.compute_finger_joint_angle(finger, joint)
                if res['joint'] is not None:
                    self.draw_joint_angle(image, res['joint'], res['angle'], res['pos'])
                else:
                    return

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, widthCam)
    cap.set(4, heightCam)
    tracker = HandTracker()
    prevCursorPosition = (gui.position().x, gui.position().y)
    wasGrabbing = False
    indexTipWindow = []
    
    while True:
        success, image = cap.read()
        if success:
            image = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)
            image = cv2.flip(image, 1)
            draw_joints = []
            draw_joints.append(tracker.compute_finger_joint_angle(Fingers.INDEX, "PIP"))
            # draw_joints.append(tracker.compute_finger_joint_angle(Fingers.INDEX, "DIP"))
            # draw_joints.append(tracker.compute_finger_joint_angle(Fingers.INDEX, "MCP"))
            
            for res in draw_joints:
                if res['joint'] is not None:
                    tracker.draw_joint_angle(image, res['joint'], res['angle'], res['pos'], font_size="S")
            # tracker.draw_all_joint_angles(image)
            fingersUp = tracker.fingersUp(lmList)
            fingersDown = tracker.fingersDown(lmList)
            # print("Fingers Up:", fingersUp)
            # print("Fingers Down:", fingersDown)

            # hand was taken off the screen
            if len(lmList) == 0: 
                # restart window
                indexTipWindow = []
            else: 
                indexTipPosition = lmList[LandMarkPoints.INDEX_FINGER_TIP.value]
                if len(indexTipWindow) <= windowSize: 
                    indexTipWindow.append(indexTipPosition)
                else: 
                    indexTipWindow = indexTipWindow[1:] + [indexTipPosition]

            if tracker.isClickingGesture(lmList):
                cv2.putText(image, "clicking", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                gui.click()
            elif tracker.isPointingGesture(fingersUp):
                camX = sum(position[1] for position in indexTipWindow) / len(indexTipWindow)
                camY = sum(position[2] for position in indexTipWindow) / len(indexTipWindow)
                x, y = tracker.getPointingScreenCoordinates(camX, camY)
                try:
                    gui.moveTo(widthScreen - x, y)
                    cv2.putText(image, "moving cursor", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                    prevCursorPosition = (widthScreen - x , y)
                # handles the case where the user tries to go out of bounds of the screen
                except (gui.FailSafeException):
                    # TODO: fix bc not working when u go to left corner
                    gui.moveTo(prevCursorPosition[0], prevCursorPosition[1])
            # elif len(lmList) != 0 and tracker.isGrabbing(fingersDown) and not wasGrabbing:
            #     wasGrabbing = True 
            #     gui.mouseDown()
            #     cv2.putText(image, "drag & drop", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
            # elif len(lmList) != 0 and tracker.isGrabbing(fingersDown) and wasGrabbing:
            #     wasGrabbing = True 
            #     cam_x = lmList[LandMarkPoints.MIDDLE_FINGER_TIP.value][1]
            #     cam_y = lmList[LandMarkPoints.MIDDLE_FINGER_TIP.value][2]
            #     x, y = tracker.getPointingScreenCoordinates(cam_x, cam_y)
            #     gui.dragTo(widthScreen - x, y, button='left')
            #     cv2.putText(image, "drag & drop", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
            # elif wasGrabbing and tracker.isDropping(fingersUp):
            #     gui.mouseUp(button='left') 
            #     cv2.putText(image, "release drop", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
            #     wasGrabbing = False
            elif tracker.isScrollingUpGesture(fingersUp):
                gui.scroll(5)
                cv2.putText(image, "Scroll Up", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
            elif tracker.isScrollingDownGesture(fingersDown):
                gui.scroll(-5)
                cv2.putText(image, "Scroll Down", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
            cv2.imshow('MediaPipe Hands', image)
            # cv2.imshow("Video",image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()