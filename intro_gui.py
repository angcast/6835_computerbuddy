import PySimpleGUI as sg

#define layout
TAB_FONT = ("Helvetica", 20)
FONT = ("Helvetica", 18)
IMAGE_SIZE = (400,300)
TEXT_SIZE = (100,2)

image1 = [[sg.Image(r'img_resources/g1.png', size=IMAGE_SIZE)]]
image2 = [[sg.Image(r'img_resources/g2.png', size=IMAGE_SIZE)]]
image3 = [[sg.Image(r'img_resources/g3.png', size=IMAGE_SIZE)]]
image4 = [[sg.Image(r'img_resources/g4.png', size=IMAGE_SIZE)]]
image5 = [[sg.Image(r'img_resources/g5.png', size=IMAGE_SIZE)]]

image_layout = [
    [sg.Frame("", image1)],
    [sg.Frame("", image2)],
    [sg.Frame("", image3)],
    [sg.Frame("", image4)],
    [sg.Frame("", image5)]
]

layout1 = [
            [sg.Text('Use the following gestures to control the mouse!',size=TEXT_SIZE, font=FONT, text_color="Black", background_color="White")],
            [sg.Column(image_layout, scrollable=True, size=(1500,500))]
           ]

layout2=[   
            [sg.Text('Use the following voice commands to control the keyboard!',size=TEXT_SIZE, font=FONT, text_color="Black", background_color="White")],
            [sg.Text('"Type <phrase>" : Types the phrase', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Open <application>" : Opens an application', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Close <application>" : Closes the application', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Add tab/window" : Adds a new tab or window', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Enable instructions" : Turns on instructions feature', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Help" : Opens this instruction screen', size=TEXT_SIZE, font=FONT)],
        ]
            


#Define Layout with Tabs         
tabgrp = [  [[sg.Text('Welcome to Computer Buddy!', expand_y=True, size=TEXT_SIZE, font=FONT, text_color="Black")]],
            [sg.TabGroup([
                        [sg.Tab('Gesture Commands', layout1, title_color='Red',border_width =10, background_color='White'),
                         sg.Tab('Voice_Commands', layout2 ,title_color='Blue',background_color='White')]
                        ], tab_location='centertop',
                       title_color='Black', tab_background_color='Grey',selected_title_color='Green', font=TAB_FONT,
                       selected_background_color='Gray', border_width=5), sg.Button('Close')]]  
        
window = sg.Window("ComputerBuddy Manual", tabgrp, size=(600, 500))

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "Close" or event == sg.WIN_CLOSED:
        break

window.close()