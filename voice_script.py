from pynput.keyboard import Key, Controller
import speech_recognition as sr
import pyautogui
import pyttsx3
import subprocess
import sys
import pyaudio
import time

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
    if "and" in transcript: # isolate sub commands and run them sequentially
        join_idx = words.index("and")
        command1 = ' '.join(words[:join_idx])
        command2 = ' '.join(words[join_idx+1:])
        scrape_transcript_for_commands(command1, instructions_enabled, delete_length)
        time.sleep(2)
        scrape_transcript_for_commands(command2, instructions_enabled, delete_length)
    # application controls
    elif "type" in transcript:
        system_reply("typing")
        phrase = ' '.join(words[1:])
        keyboard.type(phrase) # assuming phrase is "type <phrase>"
        length = len(phrase)
        return length
    elif "copy" in transcript:
        command_used = "copy"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('c')
        system_reply("copied")
    elif "cut" in transcript:
        command_used = "cut"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('x') 
        system_reply("cut")  
    elif "all" in transcript:
        command_used = "all"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('a') 
        system_reply("selecting all")  

    elif "paste" in transcript:
        command_used = "paste"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('v') 
        system_reply("pasted")  
    elif "close" in transcript: # MAC specific, quitting application
        command_used = "close"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('q')
    elif "delete" in transcript:
        system_reply("deleting")
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
                
    elif "open" in transcript:
        command_used = "open"
        system_reply("Opening {}".format(' '.join(words[1:])))
        with keyboard.pressed(Key.cmd): # open spotlight search, only for MACs
            keyboard.tap(Key.space)
        # for some reason pynput does not work in spotlight search?
        pyautogui.typewrite(' '.join(words[1:])) # assuming phrase is "open <app>"
        keyboard.tap(Key.enter)

    # video controls
    elif any(word in transcript for word in ["skip", "forward", "fast forward"]):
        command_used = "skip"
        video_control(words, is_skip=True)
    
    elif any(word in transcript for word in ["rewind", "back", "go back"]):
        command_used = "rewind"
        video_control(words, is_skip=False)
    
    elif any(word in transcript for word in ["play", "pause", "stop"]):
        command_used = "play"
        keyboard.tap('k')

    elif "speed" in transcript:
        if "increase" in transcript:
            command_used = "increase_speed"
            system_reply("increasing video playback speed")
            with keyboard.pressed(Key.shift):
                pyautogui.press(">")
        else:
            command_used = "decrease_speed"
            system_reply("decreasing video playback speed")
            with keyboard.pressed(Key.shift):
                pyautogui.press("<")  
    elif "zoom" in transcript:
        command_used = "zoom"
        zoom_in = True if "in" in transcript else False
        factor = None
        factors = [("one", 1), ("two", 2), ("three", 3), ("four", 4)]
        num_found = False
        for word in words:
            if num_found:
                break
            if word.isdigit():
                factor = int(word)
                num_found = True
            else:
                for num_word, num in factors:
                    if word == num_word:
                        factor = num
                        num_found = True
                        break
            
        system_reply("Zooming {} {} times".format("in" if zoom_in else "out", factor))
        for _ in range(factor):
                if zoom_in:
                    pyautogui.hotkey('command', '=')
                else:
                    pyautogui.hotkey('command', '-')
    elif "volume" in transcript:
        command_used = "zoom"
        factor = None
        is_increase = True if "increase" in transcript else False
        is_mute = True if "mute" in transcript else False
        for word in words:
            if word.isdigit():
                factor = int(word)
                print("factor detected!", factor)
                break
        if not is_mute:
            system_reply("{} volume by {}".format("Increasing" if is_increase else "Decreasing", factor))
        else:
            system_reply("Muting volume")
        for _ in range(factor):
                if is_increase:
                    keyboard.tap(Key.media_volume_up)
                elif not is_increase:
                    keyboard.tap(Key.media_volume_up)
                elif is_mute:
                    keyboard.tap(Key.media_volume_mute) 
    
    elif "save" in transcript:
        system_reply("saving")
        command_used = "save"
        with keyboard.pressed(Key.cmd):
            keyboard.tap('s')
        
                    
    elif "gestures" in transcript:
        system_reply("turning on gestures")
        subprocess.Popen([sys.executable, './gestures.py', '--username', 'root']) 
    
    elif "screen" in transcript:
        system_reply("Maximizing window")
        pyautogui.hotkey('win','ctrl','command', 'f')

    else:
        url = contains_url(words)
        if url != "":
            with keyboard.pressed(Key.cmd):
                keyboard.tap('l')
            keyboard.type(url)
            keyboard.tap(Key.enter)

    if instructions_enabled and command_used is not None:
    # if True and command_used is not None:
        relay_keyboard_instruction(command_used)

def relay_keyboard_instruction(command_used):
    buddy_transcript = { # list of instructions that machine will reply with
        "open": "Press the command and space keys, and then type the application name and press the Enter key",
        "tab": "Press the command and t keys to add a new tab",
        "window": "Press the command and n keys to add a new window",
        "skip": "Press the l or right arrow key while watching a video in order to skip",
        "rewind": "Press the j or left arrow key while watching a vidoe in order to rewind",
        "play": "Press the k or spacebar key in order to play or pause a video",
        "copy": "Press the command and c keys to copy",
        "cut": "Press the command and x keys to cut",
        "paste": "Press the command and v keys to paste",
        "all": "Press the command and a keys to select all",
        "increase_speed": "Press the shift and greater than keys to increase playback speed",
        "decrease_speed": "Press the shift and less than keys to decrease playback speed",
        "zoom": "To zoom in or out, press the command key and equal or minus key respectively",
        "save": "To save, press the command and s keys"
    }


    if command_used is not None:
        p = subprocess.Popen([sys.executable, './keyboard_gui.py', command_used]) 
        #your code
        system_reply(buddy_transcript[command_used])
        time.sleep(1) # may need this for reability
        p.kill()
 


if __name__ == "__main__":

    r = sr.Recognizer()
    mic = sr.Microphone() 
    instructions_enabled = False
    delete_length = 0
    subprocess.Popen([sys.executable, './intro_gui.py', '--username', 'root'])
    system_reply("Starting voice assistant")
    try:
        while True:
            # system_reply("Please say a command") # maybe too annoying
            transcript = recognize_audio(r, mic)
            if transcript is not None:
                if "learning" in transcript:
                    if "enable" in transcript:
                        instructions_enabled = True
                        system_reply("turning on learning mode")
                    else:
                        instructions_enabled = False
                        system_reply("turning off learning mode")
                print("recognized speech:", transcript)
                result = scrape_transcript_for_commands(transcript, instructions_enabled, delete_length)
                if "type" in transcript: # store phrase length
                    delete_length = result
    except KeyboardInterrupt:
        print("Quitting Application") 
    # pass