import PySimpleGUI as sg
import sys

# TAB_FONT = ("Helvetica", 20)
# FONT = ("Helvetica", 18)
# IMAGE_SIZE = (400,300)
# TEXT_SIZE = (100,2)

def show_keyboard_shortcut(command_used):
    
    im_path = "./img_resources/{}.png".format(command_used)
    # im_path = "./img_resources/open.png"
    image = [[sg.Image(im_path,  size=(500,500), key='-IMAGE-')]]
    layout1 = [
                [sg.Frame("", image)]
            ]


    window = sg.Window('Keyboard Shortcut', layout1, margins=(0, 0), finalize=True)

    # Convert im to ImageTk.PhotoImage after window finalized


    # update image in sg.Image
    while True:

        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Terminate':
            break
    window.close()


# for i, arg in enumerate(sys.argv):
    # print(f"Argument {i:>6}: {arg}")
show_keyboard_shortcut('all')