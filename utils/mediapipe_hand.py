# -*- ecoding: utf-8 -*-
# @ModuleName: mediapipe_hand
# @Function: 
# @Author: Xie Zhongzhao
# @Time: 2021/6/21 14:17

import mediapipe as mp
import cv2
import numpy as np

class MediapipeHand:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.pre_joint3d = np.random.rand(21, 3) #

        hand_joints = dict()
        hand_joints['Right'] = np.random.rand(21, 3)
        hand_joints['Left'] = np.random.rand(21, 3)
        self.pre_two_joints = hand_joints

        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)

    def get_hands_info(self, frame):

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Flip on horizontal
        # image = cv2.flip(image, 1)
        # Set flag
        image.flags.writeable = False
        # Detections
        results = self.hands.process(image)
        # RGB 2 BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                           circle_radius=4),
                                               self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                           circle_radius=2),
                                               )
        cv2.imshow('Hand Tracking', image)
        return results

    def get_3djoints(self, frame, hand_num):
        if hand_num<1 or hand_num>2:
            hand_num = 1

        joints_3d = list()
        joints_left_right = dict()
        joints_left_right['Right'] = np.random.rand(21, 3)
        joints_left_right['Left'] = np.random.rand(21, 3)

        with self.mp_hands.Hands(max_num_hands=2,
                                 min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5) as hands:
            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Set flag
            image.flags.writeable = False
            # Detections
            results = hands.process(image)
            # RGB 2 BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # print("results.multi_hand_landmarks: ", results.multi_hand_landmarks)
            if hand_num == 1:
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    for joint in range(0, 21):
                        joints_3d.append([hand.landmark[joint].x, hand.landmark[joint].y, hand.landmark[joint].z])
                    self.pre_joint3d = joints_3d
                else:
                    joints_3d = self.pre_joint3d

            if hand_num == 2:
                if results.multi_hand_landmarks:
                    for hand_landmarks, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        if handness.classification[0].label == 'Left':
                            pre_joints_left = list()
                            for id, landmark in enumerate(hand_landmarks.landmark):
                                pre_joints_left.append([landmark.x, landmark.y, landmark.z])
                            pre_joints_left = np.array(pre_joints_left)
                            joints_left_right['Left'] = pre_joints_left

                        elif handness.classification[0].label == 'Right':
                            pre_joints_right = list()
                            for id, landmark in enumerate(hand_landmarks.landmark):
                                pre_joints_right.append([landmark.x, landmark.y, landmark.z])
                            pre_joints_right = np.array(pre_joints_right)
                            joints_left_right['Right'] = pre_joints_right

                    self.pre_two_joints = joints_left_right
                else:
                    joints_left_right = self.pre_two_joints

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS,
                                                   self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                               circle_radius=4),
                                                   self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                               circle_radius=2),
                                                   )
            cv2.imshow('Hand Tracking', image)

        return np.array(joints_3d), joints_left_right














