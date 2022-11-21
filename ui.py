
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


def main():
    args = get_args()
    mode = args.mode
    number = args.number

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode='store_true',
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]



    cap = cv.VideoCapture(0)
    # 카메라 준비
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print("cap size : "+str(cap_width)+"  "+str(cap_height))
    # ---===--- Get some Stats --- #
    fps = cap.get(cv.CAP_PROP_FPS)

    sg.theme('Black')

    # ---===--- define the window layout --- #
    ui_column = [
            [sg.Graph(canvas_size=(800, 800), graph_bottom_left=(0,0), graph_top_right=(400, 400), background_color='gray', key='graph'),sg.Slider(range=(0, 100), size=(20, 20), enable_events=True, orientation='v', key='-SLIDER_V-')],      
            [sg.Slider(range=(0, 100), size=(50, 10), enable_events=True, orientation='h', key='-SLIDER_H-')],
            [sg.T('Test Buttons:'), sg.Button('Clock'), sg.Button('Move')],
            [sg.Radio(text="1 - Move", group_id=1, size=(30, 10), key='-RADIO1-')],
            [sg.Radio(text="2 - Volume", group_id=1, size=(30, 10), key='-RADIO2-')],
            [sg.Radio(text="3 - Slider Vertical", group_id=1, size=(30, 10), key='-RADIO3-')],
            [sg.Radio(text="4 - Slider Horizontal", group_id=1, size=(30, 10), key='-RADIO4-')],
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

    # locate the elements we'll be updating. Does the search only 1 time
    image_elem = window['-IMAGE-']
    slider_v = window['-SLIDER_V-']
    slider_h = window['-SLIDER_H-']
    slider_v.update(50)
    slider_h.update(50)
    radio1 = window['-RADIO1-']
    radio2 = window['-RADIO2-']
    radio3 = window['-RADIO3-']
    radio4 = window['-RADIO4-']
    radio1.update(True)
    curRadio = 1

    svV = 200
    svH = 200

    timeout = 1000//fps                 # time in ms to use for window reads
    
    graph = window['graph']
    circle = graph.DrawCircle((50,50), 50, fill_color='dark gray',line_color='black')
    line_v = graph.DrawLine((200,0),(200,400),color='white')
    line_h = graph.DrawLine((0,200),(400,200),color='red')

    cx=50
    cy=50
    point = []
    px=[50,73,88,95,88,73,50,27,12,5 ,12,27]
    py=[95,88,73,50,27,12,5 ,12,27,50,73,88]
    bef=0
    nxt=1

    rot=0
    hold=False

    moveList=[[50,50],[200,750],[600,300],[50,550],[300,600],[650,50]]
    m = 0

    for i in range(0,12):
        if i==0:
            p = graph.DrawPoint((px[i],py[i]), 8, color='red')
        else:
            p = graph.DrawPoint((px[i],py[i]), 8, color='yellow')

        point.append(p)


    # ---===--- LOOP through video file by frame --- #
    cur_frame = 0
    while cap.isOpened():
        
        event, values = window.read(timeout=timeout)
        if event in ('Exit', None):
            break
        ret, image = cap.read()
        if not ret:  # if out of data stop looping
            break
        # if someone moved the slider manually, the jump to that frame
        # if int(values['-SLIDER_V-']) != cur_frame-1:
        #     cur_frame = int(values['-SLIDER_V-'])
        #     vidFile.set(cv.CAP_PROP_POS_FRAMES, cur_frame)
        # slider_v.update(cur_frame)
        # cur_frame += 1


        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 검출 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):                                  
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # calc_bounding_rect
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 랜드마크
                landmark_list, landmark_3Dlist = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # sign 분류
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                debug_image = draw_info_text(
                            debug_image,
                            brect,
                            handedness,
                            keypoint_classifier_labels[hand_sign_id],
                        )
                
                v0 = [1,0]
                v = [landmark_list[5][0]-landmark_list[17][0],landmark_list[5][1]-landmark_list[17][1]]
                a1 = angle(v0,v)
                
                ang=math.degrees(a1)
                print("angle : "+str)


                # 1,2,3,4 표시
                if keypoint_classifier_labels[hand_sign_id]=="Gesture1" or keypoint_classifier_labels[hand_sign_id]=="Pointer":
                    print("Gesture1 : slider Vertical")
                    radio1.update(True)
                    curRadio = 1
                elif keypoint_classifier_labels[hand_sign_id]=="Gesture2" or keypoint_classifier_labels[hand_sign_id]=="V":
                    print("Gesture2 : slider Horizontal")
                    radio2.update(True)
                    curRadio = 2
                elif keypoint_classifier_labels[hand_sign_id]=="Gesture3" or keypoint_classifier_labels[hand_sign_id]=="OK":
                    print("Volume")
                    radio3.update(True)
                    curRadio = 3
                elif keypoint_classifier_labels[hand_sign_id]=="Gesture4" or keypoint_classifier_labels[hand_sign_id]=="Thumb":
                    print("Pointer")
                    radio4.update(True)
                    curRadio = 4


                # 제스처 컨트롤
                if curRadio == 1: # Move
                    if keypoint_classifier_labels[hand_sign_id]=="Pointer":
                        print("Move")
                        
                        pointerX = landmark_list[8][0]
                        pointerY = landmark_list[8][1]

                        mx=pointerX-cx
                        my=pointerY-cy
                        print("cx:"+str(cx)+" xy:"+str(cy))
                        graph.MoveFigure(circle, mx,my)
                        for i in range(0,12):
                            graph.MoveFigure(point[i], mx,my)
                        
                        cx=pointerX
                        cy=pointerY
                    
                elif curRadio == 2: # Volume
                    # v0 = [1,0]
                    # v = [landmark_list[5][0]-landmark_list[17][0],landmark_list[5][1]-landmark_list[17][1]]
                    # a1 = angle(v0,v)
                    
                    # ang=math.degrees(a1)-90
                    # print("angle : "+str)



                    graph.TKCanvas.itemconfig(point[bef], fill = "yellow")
                    graph.TKCanvas.itemconfig(point[nxt], fill = "red")
                    bef+=1
                    if bef>=12: bef=0
                    nxt+=1
                    if nxt>=12: nxt=0
                    
                    if keypoint_classifier_labels[hand_sign_id]=="Open":
                        print("Volume")
                        if hold:
                            trot = rot

                        hold = False
                    elif keypoint_classifier_labels[hand_sign_id]=="Close":
                        print("Volume")
                        hold = True



                if curRadio == 3: # slider Vertical
                    if keypoint_classifier_labels[hand_sign_id]=="Grab":
                        print("slider Vertical")
                        slider_v.update(50)
                        h = int(values['-SLIDER_V-']) * 4
                        graph.MoveFigure(line_h,0, h-svH)
                        svH=h

                elif curRadio == 4: # slider Horizontal
                    if keypoint_classifier_labels[hand_sign_id]=="Grab":
                        print("slider Horizontal")
                        slider_h.update(50)
                        v = int(values['-SLIDER_H-']) * 4
                        graph.MoveFigure(line_v,v-svV, 0)
                        svV=v



        debug_image = draw_info(debug_image, fps, mode, number)

        imgbytes = cv.imencode('.ppm', debug_image)[1].tobytes()  # can also use png.  ppm found to be more efficient
        image_elem.update(data=imgbytes)

        if event is 'Move':
            m+=1
            if m>=6: m=0
            print("move:"+str(m))

            mx=moveList[m][0]-cx
            my=moveList[m][1]-cy
            print("cx:"+str(cx)+" xy:"+str(cy))
            graph.MoveFigure(circle, mx,my)
            for i in range(0,12):
                graph.MoveFigure(point[i], mx,my)
            
            cx=moveList[m][0]
            cy=moveList[m][1]

        if event is 'Clock':
            graph.TKCanvas.itemconfig(point[bef], fill = "yellow")
            graph.TKCanvas.itemconfig(point[nxt], fill = "red")
            bef+=1
            if bef>=12: bef=0
            nxt+=1
            if nxt>=12: nxt=0
        
        if event is '-SLIDER_H-':
            v = int(values['-SLIDER_H-']) * 8
            graph.MoveFigure(line_v,v-svV, 0)
            svV=v

        if event is '-SLIDER_V-':
            h = int(values['-SLIDER_V-']) * 8
            graph.MoveFigure(line_h,0, h-svH)
            svH=h





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--number", type=int, default=0)
    args = parser.parse_args()

    return args

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

main()