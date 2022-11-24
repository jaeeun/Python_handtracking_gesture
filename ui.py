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
from lib_ui import *


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

    window = DrawUI()
    
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

    svV = 400
    svH = 400

    timeout = 1000//fps                 # time in ms to use for window reads
    
    graph = window['graph']
    circle = graph.DrawCircle((50,50), 50, fill_color='dark gray',line_color='black')
    line_v = graph.DrawLine((350,0),(350,700),color='white')
    line_h = graph.DrawLine((0,350),(700,350),color='red')

    cx=50
    cy=50
    point = []
    px=[50,73,88,95,88,73,50,27,12,5 ,12,27]
    py=[95,88,73,50,27,12,5 ,12,27,50,73,88]
    vol=0

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
        
        image = cv.flip(image, 1)
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
                
                
                # v0 = [0,1]
                # v = [landmark_list[5][0]-landmark_list[13][0],landmark_list[5][1]-landmark_list[13][1]]
                # a1 = angle(v0,v)
                
                # ang=math.degrees(a1)
                # print("angle : "+str(ang))
                # debug_image = draw_temp(debug_image,"angle : "+str(ang))


                detect(hand_sign_id)

                # 1,2,3,4 표시
                if keypoint_classifier_labels[hand_sign_id]=="Gesture1" or keypoint_classifier_labels[hand_sign_id]=="Pointer":
                    if detectCount[hand_sign_id]>3 and curRadio!=1:
                        radio1.update(True)
                        curRadio = 1
                elif keypoint_classifier_labels[hand_sign_id]=="Gesture2" or keypoint_classifier_labels[hand_sign_id]=="V":
                    if detectCount[hand_sign_id]>3 and curRadio!=2:
                        radio2.update(True)
                        curRadio = 2
                        
                elif keypoint_classifier_labels[hand_sign_id]=="Gesture3" or keypoint_classifier_labels[hand_sign_id]=="OK":
                    if detectCount[hand_sign_id]>3 and curRadio!=3:
                        radio3.update(True)
                        curRadio = 3
                elif keypoint_classifier_labels[hand_sign_id]=="Gesture4" or keypoint_classifier_labels[hand_sign_id]=="Thumb":
                    if detectCount[hand_sign_id]>3 and curRadio!=4:
                        radio4.update(True)
                        curRadio = 4

                debug_image = draw_Gesture(debug_image,curRadio, "number "+str(curRadio))
                

                # 제스처 컨트롤
                if curRadio == 1: # Move
                    print("landmark x:"+str(landmark_list[8][0])+" y:"+str(landmark_list[8][1]))
                    
                    pointerX = landmark_list[8][0]
                    pointerY = cap_height - landmark_list[8][1]

                    mx=pointerX-cx
                    my=pointerY-cy
                    # print("cx:"+str(cx)+" cy:"+str(cy))
                    graph.MoveFigure(circle, mx,my)
                    for i in range(0,12):
                        graph.MoveFigure(point[i], mx,my)
                    
                    cx=pointerX
                    cy=pointerY
                    
                elif curRadio == 2: # Volume
                    v0 = [0,1]
                    v = [landmark_list[5][0]-landmark_list[13][0],landmark_list[5][1]-landmark_list[13][1]]
                    a1 = angle(v0,v)
                    
                    ang=math.degrees(a1)

                    # 화면에 각도 보여주고 있어서 필요없으면 지워야함
                    debug_image = draw_temp(debug_image,"angle : "+str(ang))
                    
                    if ang>110:
                        graph.TKCanvas.itemconfig(point[vol], fill = "yellow")
                        vol+=1
                        if vol>=12: vol=0
                        graph.TKCanvas.itemconfig(point[vol], fill = "red")
                    elif ang<70:
                        graph.TKCanvas.itemconfig(point[vol], fill = "yellow")
                        vol-=1
                        if vol<0: vol=11
                        graph.TKCanvas.itemconfig(point[vol], fill = "red")

                    

                if curRadio == 3: # slider Vertical
                    if keypoint_classifier_labels[hand_sign_id]=="Grab":
                        print("slider Vertical")
                        percent = 100 - (landmark_list[8][1]-100) / (cap_height-200) * 100
                        # print("vert landmark_list[8][1] : "+str(landmark_list[8][1])+"  percent : "+str(percent))
                        slider_v.update(percent)
                        h = int(values['-SLIDER_V-']) * 7
                        graph.MoveFigure(line_h,0, h-svH)
                        svH=h

                elif curRadio == 4: # slider Horizontal
                    if keypoint_classifier_labels[hand_sign_id]=="Grab":
                        print("slider Horizontal")
                        percent = (landmark_list[8][0]-100) / (cap_width-200) * 100
                        # print("hori landmark_list[8][0] : "+str(landmark_list[8][0])+"  percent : "+str(percent))
                        slider_h.update(percent)
                        v = int(values['-SLIDER_H-']) * 7
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
            graph.TKCanvas.itemconfig(point[vol], fill = "yellow")
            vol+=1
            if vol>=12: vol=0
            graph.TKCanvas.itemconfig(point[vol], fill = "red")
            
        
        if event is '-SLIDER_H-':
            v = int(values['-SLIDER_H-']) * 7
            graph.MoveFigure(line_v,v-svV, 0)
            svV=v

        if event is '-SLIDER_V-':
            h = int(values['-SLIDER_V-']) * 7
            graph.MoveFigure(line_h,0, h-svH)
            svH=h




main()
