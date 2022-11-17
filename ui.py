import PySimpleGUI as sg
# import PySimpleGUIQt
import cv2 as cv


def main():
    video = "C:\\Users\\jaeeun84.yang\\Videos\\ARGlass\\2020-05-27 19-53-13.mp4"
    vidFile = cv.VideoCapture(0)
    # ---===--- Get some Stats --- #
    fps = vidFile.get(cv.CAP_PROP_FPS)

    sg.theme('Black')

    # ---===--- define the window layout --- #
    ui_column = [
            [sg.Graph(canvas_size=(400, 400), graph_bottom_left=(0,0), graph_top_right=(400, 400), background_color='gray', key='graph'),sg.Slider(range=(0, 100), size=(20, 20), enable_events=True, orientation='v', key='-SLIDER_V-')],      
            [sg.Slider(range=(0, 100), size=(50, 10), enable_events=True, orientation='h', key='-SLIDER_H-')],
            [sg.T('Test Buttons:'), sg.Button('Clock'), sg.Button('Move')],
            [sg.Radio(text="1 - Volume", group_id=1, size=(30, 10), key='-RADIO1-')],
            [sg.Radio(text="2 - Slider Vertical", group_id=1, size=(30, 10), key='-RADIO2-')],
            [sg.Radio(text="3 - Slider Horizontal", group_id=1, size=(30, 10), key='-RADIO3-')],
            [sg.Radio(text="4 - Move", group_id=1, size=(30, 10), key='-RADIO4-')],
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

    moveList=[[50,50],[200,350],[100,300],[50,150],[300,300],[250,50]]
    m = 0

    for i in range(0,12):
        if i==0:
            p = graph.DrawPoint((px[i],py[i]), 8, color='red')
        else:
            p = graph.DrawPoint((px[i],py[i]), 8, color='yellow')

        point.append(p)


    # ---===--- LOOP through video file by frame --- #
    cur_frame = 0
    while vidFile.isOpened():
        event, values = window.read(timeout=timeout)
        if event in ('Exit', None):
            break
        ret, frame = vidFile.read()
        if not ret:  # if out of data stop looping
            break
        # if someone moved the slider manually, the jump to that frame
        # if int(values['-SLIDER_V-']) != cur_frame-1:
        #     cur_frame = int(values['-SLIDER_V-'])
        #     vidFile.set(cv.CAP_PROP_POS_FRAMES, cur_frame)
        # slider_v.update(cur_frame)
        # cur_frame += 1

        imgbytes = cv.imencode('.ppm', frame)[1].tobytes()  # can also use png.  ppm found to be more efficient
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
            v = int(values['-SLIDER_H-']) * 4
            graph.MoveFigure(line_v,v-svV, 0)
            svV=v

        if event is '-SLIDER_V-':
            h = int(values['-SLIDER_V-']) * 4
            graph.MoveFigure(line_h,0, h-svH)
            svH=h

main()