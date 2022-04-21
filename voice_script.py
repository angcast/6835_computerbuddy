from os import system
from pynput.keyboard import Key, Controller, Events
import speech_recognition as sr
import pyautogui
import pyttsx3
import subprocess
import sys
import time

keyboard = Controller()
engine = pyttsx3.init()

def system_reply(audio):
    engine.say(audio)
    engine.runAndWait()

def recognize_audio(r, mic):
    print("recognizing audio")
    transcript = None
    with mic as source:
        r.adjust_for_ambient_noise(source)
        r.energy_threshold = 500 
        r.dynamic_energy_threshold = False
        audio = r.listen(source)
        try:
            transcript = r.recognize_google(audio)
            print('what is transcript:', transcript)
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


def scrape_transcript_for_commands(transcript, instructions_enabled):
    print("coming into functions")
    transcript = transcript.lower()
    words = transcript.split(" ")
    command_used = None
    # application controls
    if "type" in transcript:
        system_reply("typing")
        keyboard.type(' '.join(words[1:])) # assuming phrase is "type <phrase>" 
    elif "open" in transcript:
        command_used = "open"
        system_reply("Opening {}".format(' '.join(words[1:])))
        with keyboard.pressed(Key.cmd): # open spotlight search, only for MACs
            keyboard.tap(Key.space)
        # for some reason pynput does not work in spotlight search?
        pyautogui.typewrite(' '.join(words[1:])) # assuming phrase is "open <app>"
        keyboard.tap(Key.enter)
    elif "close" in transcript: # MAC specific, quitting application
        command_used = "close"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('q')

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
    system_reply("Starting voice assistant")
    try:
        while True:
            transcript = recognize_audio(r, mic)
            if transcript is not None:
                if "instructions" in transcript:
                    if "enable" in transcript:
                        instructions_enabled = True
                        system_reply("turning on instructions")
                    else:
                        instructions_enabled = False
                        system_reply("turning off instructions")

                print("recognized speech:", transcript)
                scrape_transcript_for_commands(transcript, instructions_enabled)
    except KeyboardInterrupt:
        print("Quitting Application") 
    # pass