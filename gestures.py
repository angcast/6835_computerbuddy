from enum import Enum
import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui as gui
import subprocess
import time
import sys
import pyttsx3



widthCam, heightCam = 640, 480
frameReduction = 200 
widthScreen, heightScreen = gui.size()
windowSize = 8
standard_padding = 40
y_bottom_padding_offset = 200

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


engine = pyttsx3.init()

def system_reply(audio):
    engine.say(audio)
    engine.runAndWait()

class HandTracker():
    def __init__(self, mode=False, max_hands=1, min_detection_confidence=0.5, model_complexity=1, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = min_detection_confidence
        self.model_complexity = model_complexity
        self.tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, 
            self.max_hands,
            self.model_complexity,
            self.detection_confidence, 
            self.tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.joint_list = [
            { 'IP': [2, 3, 4], 'MCP': [0, 2, 3], 'CMC': [0, 1, 2] },
            { 'DIP': [6, 7, 8], 'PIP': [5, 6, 7], 'MCP': [0, 5, 6] },
            { 'DIP': [10, 11, 12], 'PIP': [9, 10, 11], 'MCP': [0, 9, 10] }, 
            { 'DIP': [14, 15, 16], 'PIP': [13, 14, 15], 'MCP': [0, 13, 14] }, 
            { 'DIP': [18, 19, 20], 'PIP': [17, 18, 19], 'MCP': [0, 17, 18] }
        ]

    def hands_finder(self, image, draw=True):
        # image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
        return image
    
    def hand_position_finder(self, hand_no=0):
        hand_landmark_list = []
        if self.results.multi_hand_world_landmarks:
            hand = self.results.multi_hand_world_landmarks[hand_no]
            for id, landmark in enumerate(hand.landmark):
                hand_landmark_list.append([id, landmark.x, landmark.y, landmark.z])
        return hand_landmark_list

    def camera_position_finder(self, image, hand_no=0, draw=True):
        camera_landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, landmark in enumerate(hand.landmark):
                ch, cw, cz = image.shape
                cx, cy = int(landmark.x*cw), int(landmark.y*ch)
                camera_landmark_list.append([id, cx, cy])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
        return camera_landmark_list

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
        if index_res['joint'] is not None:
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
            is_index_straight = index_res['angle'] >= 170
            if Fingers.THUMB in fingersUp:
                fingersUp.remove(Fingers.THUMB)
            return len(fingersUp) == 1 and fingersUp[0] == Fingers.INDEX and is_index_straight
        return False
    
    def is_clicking_gesture(self, landmarks):
        if len(landmarks) != 0:
            index_res = self.compute_finger_joint_angle(Fingers.INDEX, "PIP")
            index_tip = landmarks[LandMarkPoints.INDEX_FINGER_TIP.value][1:]
            thumb_tip = landmarks[LandMarkPoints.THUMB_TIP.value][1:]
            thumb_ip = landmarks[LandMarkPoints.THUMB_IP.value][1:]
            clicking_threshold = 0.05
            # index tip connecting with thumb tip
            if index_res['joint'] is not None:
                is_index_bent, is_index_open = index_res['angle'] < 175, index_res['angle'] >= 90
                is_index_clicking = is_index_bent and is_index_open
                if is_index_clicking and math.dist(index_tip, thumb_tip) <= clicking_threshold:
                    return True
                # index tip connecting with thumb ip
                if is_index_clicking and math.dist(index_tip, thumb_ip) <= clicking_threshold: 
                    return True
        return False

    def is_closed_fist(self):
        index_res = self.compute_finger_joint_angle(Fingers.INDEX, "PIP")
        middle_res = self.compute_finger_joint_angle(Fingers.MIDDLE, "PIP")
        ring_res = self.compute_finger_joint_angle(Fingers.RING, "PIP")
        pinky_res = self.compute_finger_joint_angle(Fingers.PINKY, "PIP")

        if index_res['joint'] is not None:
            is_index_closed, is_middle_closed = index_res['angle'] <= 70, middle_res['angle'] <= 70
            is_ring_closed, is_pinky_closed = ring_res['angle'] <= 70, pinky_res['angle'] <= 70

            return all([is_index_closed, is_middle_closed, is_ring_closed, is_pinky_closed])
        return False

    def is_open_palm(self):
        index_res = self.compute_finger_joint_angle(Fingers.INDEX, "PIP")
        middle_res = self.compute_finger_joint_angle(Fingers.MIDDLE, "PIP")
        ring_res = self.compute_finger_joint_angle(Fingers.RING, "PIP")
        pinky_res = self.compute_finger_joint_angle(Fingers.PINKY, "PIP")

        if index_res['joint'] is not None:
            is_index_open, is_middle_open = index_res['angle'] >= 160, middle_res['angle'] >= 160
            is_ring_open, is_pinky_open = ring_res['angle'] >= 160, pinky_res['angle'] >= 160

            return all([is_index_open, is_middle_open, is_ring_open, is_pinky_open])
        return False

    def isGrabbing(self, landmarks, fingersDown):
        if len(landmarks) != 0: 
            #ignore thumb
            if Fingers.THUMB in fingersDown: 
                fingersDown.remove(Fingers.THUMB)
            is_grabbing_gesture = self.is_closed_fist()
            return len(fingersDown) == 4 and is_grabbing_gesture

    def isDropping(self, landmarks, fingersUp):
        if len(landmarks) != 0:
            #ignore thumb
            if Fingers.THUMB in fingersUp: 
                fingersUp.remove(Fingers.THUMB)
            is_dropping_gesture = self.is_open_palm()
            return len(fingersUp) == 4 and is_dropping_gesture

    def getPointingScreenCoordinates(self, x, y): 
        """
        Maps the video cam coordinates to that of the current screen
        """
        # Since OpenCV does not detect finger in some x and y values making it
        # harder to point downward and side to side, we reduce the frame to make 
        # these cases easier
        # # normalize x-coords
        # x_cam_max = widthCam-standard_padding
        # x_cam_min = standard_padding
        # x_cam_range = x_cam_max - x_cam_min
        # x_screen_max = widthScreen
        # x_screen_min = 0
        # x_screen_range = x_screen_max - x_screen_min
        # new_x = (((x - x_cam_min) * x_screen_range) / x_cam_range) + x_screen_min
        # # normalize y-coords
        # y_cam_max = heightCam-y_bottom_padding_offset
        # y_cam_min = standard_padding
        # y_cam_range = y_cam_max - y_cam_min
        # y_screen_max = heightScreen
        # y_screen_min = 0 
        # y_screen_range = y_screen_max - y_screen_min
        # new_y = ((y - y_cam_min) * y_screen_range) / y_cam_range + y_screen_min
        # yFrameReduction = 200
        # xFrameReduction = 100
        new_x = np.interp(x, (standard_padding, widthCam-standard_padding), (0, widthScreen))
        new_y = np.interp(y, (standard_padding, heightCam-y_bottom_padding_offset), (0, heightScreen))
        return new_x, new_y
    
    def compute_finger_joint_angle(self, finger, joint):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            finger_joints = self.joint_list[finger.value]
            joint_seg = finger_joints[joint]
            # Coordinates of the three joints of a finger
            p1 = np.array([hand.landmark[joint_seg[0]].x, hand.landmark[joint_seg[0]].y, hand.landmark[joint_seg[0]].z])
            p2 = np.array([hand.landmark[joint_seg[1]].x, hand.landmark[joint_seg[1]].y, hand.landmark[joint_seg[1]].z])
            p3 = np.array([hand.landmark[joint_seg[2]].x, hand.landmark[joint_seg[2]].y, hand.landmark[joint_seg[2]].z])
            finger_angle = self.compute_joint_angle(p1, p2, p3)
            drawing_pos = np.array([1-hand.landmark[joint_seg[1]].x, hand.landmark[joint_seg[1]].y])
            return { 'joint': joint, 'angle': finger_angle, 'pos': drawing_pos }
        return { 'joint': None, 'angle': None, 'pos': None }
    
    def compute_joint_angle(self, p1, p2, p3):
        radian_angle = np.arctan2(np.linalg.norm(np.cross(p1-p2, p3-p2)), np.dot(p1-p2, p3-p2))
        # radian_angle = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
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
    def fingers_in_right_region(self, landmarks):
        if len(landmarks) != 0:
            x_center = widthCam/2
            x_index_tip = landmarks[LandMarkPoints.INDEX_FINGER_TIP.value][1]
            x_thumb_tip = landmarks[LandMarkPoints.THUMB_TIP.value][1]
            x_middle_tip = landmarks[LandMarkPoints.MIDDLE_FINGER_TIP.value][1]
            x_ring_finger_tip = landmarks[LandMarkPoints.RING_FINGER_TIP.value][1]
            x_pinky_tip = landmarks[LandMarkPoints.PINKY_TIP.value][1]
            return x_index_tip < x_center and x_thumb_tip < x_center and x_middle_tip < x_center \
                 and x_ring_finger_tip < x_center and x_pinky_tip < x_center
    
    def fingers_in_left_region(self, landmarks):
        if len(landmarks) != 0:
            x_center = widthCam/2
            x_index_tip = landmarks[LandMarkPoints.INDEX_FINGER_TIP.value][1]
            x_thumb_tip = landmarks[LandMarkPoints.THUMB_TIP.value][1]
            x_middle_tip = landmarks[LandMarkPoints.MIDDLE_FINGER_TIP.value][1]
            x_ring_finger_tip = landmarks[LandMarkPoints.RING_FINGER_TIP.value][1]
            x_pinky_tip = landmarks[LandMarkPoints.PINKY_TIP.value][1]
            return x_index_tip > x_center and x_thumb_tip > x_center and x_middle_tip > x_center \
                 and x_ring_finger_tip > x_center and x_pinky_tip > x_center

    def control_swipe(self, swipe_left, swipe_right):
        '''
        detecting swipes
        '''
        if (swipe_left):
            print(" swiping left ")
            return True
        elif (swipe_right):
            print(" swiping right ")
            return True
        return False


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, widthCam)
    cap.set(4, heightCam)
    tracker = HandTracker()
    prevCursorPosition = (gui.position().x, gui.position().y)
    currentlyGrabbing = False
    initiate_left_swipe = False 
    initiate_right_swipe = False
    desktopGesturePrevPosition = None
    indexTipWindow = []
    img_rows,img_cols=64, 64 
    while True:
        success, image = cap.read()
        if success:
            image = tracker.hands_finder(image)
            camera_landmark_list = tracker.camera_position_finder(image)
            hand_landmark_list = tracker.hand_position_finder()
            image = cv2.flip(image, 1)
            cv2.rectangle(image, (standard_padding, standard_padding), (widthCam-standard_padding, heightCam-y_bottom_padding_offset), feedbackColor, feedbackThickness)
            draw_joints = []
            # draw_joints.append(tracker.compute_finger_joint_angle(Fingers.INDEX, "PIP"))
            
            for res in draw_joints:
                if res['joint'] is not None:
                    tracker.draw_joint_angle(image, res['joint'], res['angle'], res['pos'], font_size="S")
            # tracker.draw_all_joint_angles(image)
            fingersUp = tracker.fingersUp(camera_landmark_list)
            fingersDown = tracker.fingersDown(camera_landmark_list)
       
            # hand was taken off the screen
            if len(camera_landmark_list) == 0: 
                # restart window
                indexTipWindow = []
            else: 
                indexTipPosition = camera_landmark_list[LandMarkPoints.INDEX_FINGER_TIP.value]
                if len(indexTipWindow) <= windowSize: 
                    indexTipWindow.append(indexTipPosition)
                else: 
                    indexTipWindow = indexTipWindow[1:] + [indexTipPosition]

            if tracker.is_clicking_gesture(hand_landmark_list):
                initiate_left_swipe = False 
                initiate_right_swipe = False
                if (currentlyGrabbing):
                    currentlyGrabbing = False
                    gui.mouseUp(button='left')
                cv2.putText(image, "clicking", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                gui.click()
            elif tracker.isPointingGesture(fingersUp):
                initiate_left_swipe = False 
                initiate_right_swipe = False
                if (currentlyGrabbing):
                    currentlyGrabbing = False
                    gui.mouseUp(button='left')
                camX = sum(position[1] for position in indexTipWindow) / len(indexTipWindow)
                camY = sum(position[2] for position in indexTipWindow) / len(indexTipWindow)
                x, y = tracker.getPointingScreenCoordinates(camX, camY)
                if desktopGesturePrevPosition is None:
                    desktopGesturePrevPosition = widthScreen - x, y
                try:
                    gui.moveTo(widthScreen - x, y)
                    cv2.putText(image, "moving cursor", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                    prevCursorPosition = (widthScreen - x , y)
                # handles the case where the user tries to go out of bounds of the screen
                except (gui.FailSafeException):
                    # TODO: fix bc not working when u go to left corner
                    gui.moveTo(prevCursorPosition[0], prevCursorPosition[1])     
            elif (tracker.isDropping(camera_landmark_list, fingersUp) or len(indexTipWindow) == 0) and currentlyGrabbing:
                initiate_left_swipe = False 
                initiate_right_swipe = False
                currentlyGrabbing = False
                cv2.putText(image, "dropped", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                gui.mouseUp(button='left')
            elif (tracker.is_open_palm()): 
                if (tracker.fingers_in_left_region(camera_landmark_list)):
                    if (initiate_left_swipe):
                        initiate_left_swipe = False
                        cv2.putText(image, "swiping left", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                        gui.hotkey('win','ctrl','left') 
                    # can't do elif on this condition as user might want to swipe back and forth between desktops repeatedly 
                    initiate_right_swipe = True
                elif (tracker.fingers_in_right_region(camera_landmark_list)):
                    if (initiate_right_swipe):
                        initiate_right_swipe = False
                        cv2.putText(image, "swiping right", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                        gui.hotkey('win','ctrl','right')
                    # can't do elif on this condition as user might want to swipe back and forth between desktops repeatedly 
                    initiate_left_swipe = True
            elif currentlyGrabbing and len(indexTipWindow) > 0:
                initiate_left_swipe = False 
                initiate_right_swipe = False
                cv2.putText(image, "dragging", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
                currentlyGrabbing = True
                cam_x = sum(position[1] for position in indexTipWindow) / len(indexTipWindow)
                cam_y = sum(position[2] for position in indexTipWindow) / len(indexTipWindow)
                x, y = tracker.getPointingScreenCoordinates(cam_x, cam_y)
                gui.dragTo(widthScreen - x, y, button='left')
            elif tracker.isGrabbing(camera_landmark_list, fingersDown): 
                initiate_left_swipe = False 
                initiate_right_swipe = False
                currentlyGrabbing = True 
                gui.mouseDown(button='left')
                cv2.putText(image, "start dragging", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness) 
            elif tracker.isScrollingUpGesture(fingersUp):
                initiate_left_swipe = False 
                initiate_right_swipe = False
                if (currentlyGrabbing):
                    currentlyGrabbing = False
                    gui.mouseUp(button='left')
                gui.scroll(5)
                cv2.putText(image, "Scroll Up", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
            elif tracker.isScrollingDownGesture(fingersDown):
                initiate_left_swipe = False 
                initiate_right_swipe = False
                if (currentlyGrabbing):
                    currentlyGrabbing = False
                    gui.mouseUp(button='left')
                gui.scroll(-5)
                cv2.putText(image, "Scroll Down", (10, 70), feedbackFontFace, feedbackFontSize, feedbackColor, feedbackThickness)
            cv2.imshow("Video",image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()