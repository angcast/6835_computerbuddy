from os import system
from pynput.keyboard import Key, Controller, Events
from requests import delete
import speech_recognition as sr
import pyautogui
import pyttsx3
import subprocess
import sys
import pyaudio

keyboard = Controller()
engine = pyttsx3.init()

def system_reply(audio):
    engine.say(audio)
    engine.runAndWait()

def recognize_audio(r, mic):
    transcript = None
    with mic as source:
        # r.adjust_for_ambient_noise(source, duration=5)
        r.energy_threshold = 500 
        r.dynamic_energy_threshold = False
        audio = r.listen(source)
        try:  
            transcript = r.recognize_google(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            # response["success"] = False
            # response["error"] = "API unavailable"
            print("API unavailable")
        except sr.UnknownValueError:
            # speech was unintelligible
            # response["error"] = "Unable to recognize speech"
            system_reply("unable to recognize speech. Please try again")
    return transcript


def contains_url(words):
    for w in words:
        if any(domain in w for domain in [".com", ".edu", ".org"]):
            return w
    return ""

def video_control(words, is_skip=False):
    # requires headphones!
    # extract number from transcript
    ten_sec_increment = 0
    five_sec_increment = 0
    for w in words:
        print(w)
        if w.isdigit():
            print("digit recorded:", w)
            amount = int(w)
            ten_sec_increment = amount // 10
            amount -= 10 * ten_sec_increment
            if amount > 5:
                five_sec_increment = amount // 5
            break
    if is_skip:
        print("skipping! 10s / 5s:", ten_sec_increment, five_sec_increment)
        for _ in range(ten_sec_increment):
            print("pressing l")
            keyboard.tap('l')
        for _ in range(five_sec_increment):
            keyboard.tap(Key.right)
    else:
        print("coming in here")
        print("rewinding! 10s / 5s:", ten_sec_increment, five_sec_increment)
        for _ in range(ten_sec_increment):
            print("tapping j")
            keyboard.tap('j')
        for _ in range(five_sec_increment):
            keyboard.tap(Key.left) 



def scrape_transcript_for_commands(transcript, instructions_enabled, delete_length):
    transcript = transcript.lower()
    words = transcript.split(" ")
    command_used = None
    # application controls
    if "type" in transcript:
        system_reply("typing")
        phrase = ' '.join(words[1:])
        keyboard.type(phrase) # assuming phrase is "type <phrase>"
        length = len(phrase)
        return length
    elif "close" in transcript: # MAC specific, quitting application
        command_used = "close"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('q')
    elif "delete" in transcript:
        system_reply("deleting")
        print("what is delete length @ delete op:", delete_length)
        for _ in range(delete_length):
            keyboard.tap(Key.backspace)

    # browser controls
    elif "tab" in transcript:
        command_used = "tab"
        system_reply("adding a new tab")
        with keyboard.pressed(Key.cmd):
            keyboard.tap('t')
    elif "window" in transcript:
        command_used = "window"
        system_reply("adding a new window")
        with keyboard.pressed(Key.cmd):
            keyboard.tap('n')
            
    elif "help" in transcript:
        print("spawning subprocess.......")
        system_reply("Opening help menu")
        subprocess.Popen([sys.executable, './intro_gui.py', '--username', 'root'])
    
    elif "gestures" in transcript:
        subprocess.Popen([sys.executable, './gestures.py', '--username', 'root'])

    elif "open" in transcript:
        command_used = "open"
        system_reply("Opening {}".format(' '.join(words[1:])))
        with keyboard.pressed(Key.cmd): # open spotlight search, only for MACs
            keyboard.tap(Key.space)
        # for some reason pynput does not work in spotlight search?
        pyautogui.typewrite(' '.join(words[1:])) # assuming phrase is "open <app>"
        keyboard.tap(Key.enter)
    # video controls
    if any(word in transcript for word in ["skip", "forward", "fast forward"]):
        command_used = "skip"
        video_control(words, is_skip=True)
    
    if any(word in transcript for word in ["rewind", "back", "go back"]):
        command_used = "rewind"
        video_control(words, is_skip=False)
    
    if any(word in transcript for word in ["play", "pause", "stop"]):
        command_used = "play"
        keyboard.tap('k')

    if "speed" in transcript:
        if "increase" in transcript:
            system_reply("increasing video playback speed")
            with keyboard.pressed(Key.shift):
                pyautogui.press(">")
        else:
            system_reply("decreasing video playback speed")
            with keyboard.pressed(Key.shift):
                pyautogui.press("<")    

    else:
        url = contains_url(words)
        if url != "":
            with keyboard.pressed(Key.cmd):
                keyboard.tap('l')
            keyboard.type(url)
            keyboard.tap(Key.enter)

    if instructions_enabled:
        print("relaying instruction")
        relay_keyboard_instruction(command_used)

def relay_keyboard_instruction(command_used):
    buddy_transcript = { # list of instructions that machine will reply with
        "open": "Press the command and space keys, and then type the application name and press the Enter key",
        "tab": "Press the command and t keys to add a new tab",
        "window": "Press the command and w keys to add a new window",
        "skip": "Press the l or right arrow key while watching a video in order to skip",
        "rewind": "Press the j or left arrow key while watching a vidoe in order to rewind",
        "play": "Press the k or spacebar key in order to play or pause a video"
    }
    if command_used is not None:
        system_reply(buddy_transcript[command_used])
 


if __name__ == "__main__":
    r = sr.Recognizer()
    mic = sr.Microphone() 
    instructions_enabled = False
    delete_length = 0
    system_reply("Starting voice assistant")
    try:
        while True:
            # system_reply("Please say a command") # maybe too annoying
            transcript = recognize_audio(r, mic)
            if transcript is not None:
                if "instructions" in transcript:
                    if "on" in transcript:
                        instructions_enabled = True
                        system_reply("turning on instructions")
                    else:
                        instructions_enabled = False
                        system_reply("turning off instructions")
                print("recognized speech:", transcript)
                result = scrape_transcript_for_commands(transcript, instructions_enabled, delete_length)
                if "type" in transcript: # store phrase length
                    delete_length = result
    except KeyboardInterrupt:
        print("Quitting Application") 
    # pass