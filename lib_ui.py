import csv
import copy
import argparse
import itertools
import math

import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

detectCount = [0,0,0,0,0,0,0,0,0,0,0,0]
def detect(num):
    for i in range(0,12):
        if num!=i: detectCount[i]=0
    detectCount[num] = detectCount[num] + 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--number", type=int, default=0)
    args = parser.parse_args()

    return args

def DrawUI():
    sg.theme('Black')

    # ---===--- define the window layout --- #
    ui_column = [
            [sg.Graph(canvas_size=(700, 700), graph_bottom_left=(0,0), graph_top_right=(700, 700), background_color='gray', key='graph'),sg.Slider(range=(0, 100), size=(30, 10), enable_events=True, orientation='v', key='-SLIDER_V-')],      
            [sg.Slider(range=(0, 100), size=(60, 10), enable_events=True, orientation='h', key='-SLIDER_H-')],
            [sg.Radio(text="1 - Move", group_id=1, size=(30, 10), key='-RADIO1-')],
            [sg.Radio(text="2 - Volume", group_id=1, size=(30, 10), key='-RADIO2-')],
            [sg.Radio(text="3 - Slider Vertical", group_id=1, size=(30, 10), key='-RADIO3-')],
            [sg.Radio(text="4 - Slider Horizontal", group_id=1, size=(30, 10), key='-RADIO4-')],
            [sg.T('Test Buttons:'), sg.Button('Clock'), sg.Button('Move')],
            [sg.Push(), sg.Button('Exit', font='Helvetica 14')]
    ]

    video_column = [
            [sg.Text('Hand Demo', size=(15, 1), font='Helvetica 20')],
              [sg.Image(key='-IMAGE-')]
    ]

    layout = [
        [
            sg.Column(ui_column),
            sg.VSeperator(),
            sg.Column(video_column),
        ]
    ]
    
    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration', layout, no_titlebar=False, location=(0, 0), finalize=True)

    return window

def RotateVolume(graph, point, vol, tempVol, defAngle):
    defAngle = defAngle/15
    if defAngle > 0:
        if defAngle > tempVol-vol:
            graph.TKCanvas.itemconfig(point[tempVol], fill = "yellow")
            tempVol+=1
            if tempVol>=12: tempVol=0
            graph.TKCanvas.itemconfig(point[tempVol], fill = "red")
    else:
        if defAngle < tempVol-vol:
            graph.TKCanvas.itemconfig(point[tempVol], fill = "yellow")
            tempVol-=1
            if tempVol<0: tempVol=11
            graph.TKCanvas.itemconfig(point[tempVol], fill = "red")

    return tempVol

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    landmark_3dpoint = []

    # 키 포인트
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z * image_height

        landmark_point.append([landmark_x, landmark_y])
        landmark_3dpoint.append([landmark_x, landmark_y, landmark_z])

    return landmark_point,landmark_3dpoint


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 0 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

def draw_Gesture(image, num, gesture):
    info_text = str(num) + " - " + gesture
    cv.putText(image, info_text, (300, 300),
               cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv.LINE_AA)

    return image

def draw_temp(image, st):
    cv.putText(image, st, (100, 500), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
    return image

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 상대 좌표로 변환
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 1次元リストに変換
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
