import PySimpleGUI as sg

#define layout
TAB_FONT = ("Helvetica", 20)
FONT = ("Helvetica", 18)
IMAGE_SIZE = (400,300)
TEXT_SIZE = (100,2)

sg.theme('LightBrown 3')

image1 = [[sg.Image(r'img_resources/g1.png', size=IMAGE_SIZE)]]
image2 = [[sg.Image(r'img_resources/g2.png', size=IMAGE_SIZE)]]
image3 = [[sg.Image(r'img_resources/g3.png', size=IMAGE_SIZE)]]
image4 = [[sg.Image(r'img_resources/clicking.png', size=IMAGE_SIZE)]]
image5 = [[sg.Image(r'img_resources/g5.png', size=IMAGE_SIZE)]]
image6 = [[sg.Image(r'img_resources/swiping_left.png', size=IMAGE_SIZE)]]
image7 = [[sg.Image(r'img_resources/swiping_right.png', size=IMAGE_SIZE)]]


image_layout = [
    [sg.Frame("", image1)],
    [sg.Frame("", image2)],
    [sg.Frame("", image3)],
    [sg.Frame("", image4)],
    [sg.Frame("", image5)],
    [sg.Frame("", image6)],
    [sg.Frame("", image7)],
]

layout1 = [
            [sg.Text('Use the following gestures to control the mouse!',size=TEXT_SIZE, font=FONT, text_color="Black", background_color="White")],
            [sg.Column(image_layout, scrollable=True, size=(1500,500))]
           ]

help_text=[   
            
            [sg.Text('Use the following voice commands to control the keyboard!',size=TEXT_SIZE, font=FONT, text_color="Black", background_color="White")],
            [sg.Text('"<first command> and <second command>" : Do the <first command>, and then the <second command>', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please Enable gestures" : Turn on ComputerBuddy gesture recognition system', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Enable/Disable learning mode" : Turns on/off pop up window showing where keyboard shorcuts are on a keyboard', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please type <phrase>" : Types the phrase', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Open <application>" : Opens an application', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please save" : Saves the current file', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Close <application>" : Closes the application', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Add tab/window" : Adds a new tab or window', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Go to <website>" : Navigate to <website>', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Zoom In/Out <number> times" : Zooms the screen in/out <number> times', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Cut/Copy/Select all/Paste" : Cuts/Copies/Selects all/Pastes the highlighted text', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please delete: Deletes the selected / last inputted text', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please Play/Pause: Plays/Pauses a video', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Increase/Decrease speed" : Increase/Decrease the playback speed of a video', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please Skip/Fast forward <time> seconds: Fast forwards <time> seconds in a video', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please Go Back/Rewind <time> seconds: Rewind <time> seconds in a video', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please Increase/Decrease volume by <number>" : Increase/Decrease the volume by <number>', size=TEXT_SIZE, font=FONT)],
            [sg.Text('"Please Maximize/Minimize screen" : Maximize/Minimize the current screen', size=TEXT_SIZE, font=FONT)],
        ]


layout2 = [
            [sg.Column(help_text, scrollable=True, size=(1500,500))]
           ]
    


#Define Layout with Tabs         
tabgrp = [  [[sg.Text('Welcome to ComputerBuddy!', expand_y=True, size=TEXT_SIZE, font=FONT, text_color="Black")]],
            [sg.TabGroup([
                        [sg.Tab('Gesture Commands', layout1, title_color='Red',border_width =10, background_color='White'),
                         sg.Tab('Voice Commands', layout2 ,title_color='Blue',background_color='White')]
                        ], tab_location='centertop',
                       title_color='Black', tab_background_color='Grey',selected_title_color='Green', font=TAB_FONT,
                       selected_background_color='Gray', border_width=5), sg.Button('Close')]]  
        
window = sg.Window("ComputerBuddy Manual", tabgrp, size=(900, 500))

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "Close" or event == sg.WIN_CLOSED:
        break

window.close()