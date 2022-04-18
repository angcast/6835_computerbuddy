from pynput.keyboard import Key, Controller
import speech_recognition as sr
import pyautogui
import pyttsx3

keyboard = Controller()
engine = pyttsx3.init()

def system_reply(audio):
    print(audio)
    engine.say(audio)
    engine.runAndWait()

def recognize_audio(r, mic):
    transcript = None
    with mic as source:
        r.adjust_for_ambient_noise(source)
        r.energy_threshold = 500 
        r.dynamic_energy_threshold = False
        audio = r.listen(source)
        try:
            transcript = r.recognize_google(audio)
            print("what is transcript dood:", transcript)
        except sr.RequestError:
            # API was unreachable or unresponsive
            # response["success"] = False
            # response["error"] = "API unavailable"
            print("API unavailable")
        except sr.UnknownValueError:
            # speech was unintelligible
            # response["error"] = "Unable to recognize speech"
            print("unable to recognize speech")
    return transcript


def contains_url(words):
    for w in words:
        if any(domain in w for domain in [".com", ".edu", ".org"]):
            return w
    return ""

def video_control(words, is_skip=False):
    # requires headphones!
    print("coming into video controls!")
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


def scrape_transcript_for_commands(transcript):
    transcript = transcript.lower()
    words = transcript.split(" ")
    # application controls
    if "type" in transcript:
        keyboard.type(' '.join(words[1:])) # assuming phrase is "type <phrase>" 
    if "open" in transcript:
        with keyboard.pressed(Key.cmd): # open spotlight search, only for MACs
            keyboard.tap(Key.space)
        # for some reason pynput does not work in spotlight search?
        pyautogui.typewrite(' '.join(words[1:])) # assuming phrase is "open <app>"
        keyboard.tap(Key.enter)
    if "close" in transcript: # MAC specific, quitting application
        with keyboard.pressed(Key.cmd):
            keyboard.tap('q')

    # browser controls
    if "tab" in transcript:
        print("tab in transcript")
        with keyboard.pressed(Key.cmd):
            keyboard.tap('t')
        print('pressed keyboard')
    if "window" in transcript:
        with keyboard.pressed(Key.cmd):
            keyboard.tap('n')

    # video controls
    if any(word in transcript for word in ["skip", "forward", "fast forward"]):
        video_control(words, is_skip=True)
    
    if any(word in transcript for word in ["rewind", "back", "go back"]):
        video_control(words, is_skip=False)
    
    if any(word in transcript for word in ["play", "pause", "stop"]):
        keyboard.tap('k')

    else:
        url = contains_url(words)
        if url != "":
            with keyboard.pressed(Key.cmd):
                keyboard.tap('l')
            keyboard.type(url)
            keyboard.tap(Key.enter)


if __name__ == "__main__":
    r = sr.Recognizer()
    mic = sr.Microphone() 
    try:
        while True:
            transcript = recognize_audio(r, mic)
            if transcript is not None:
                print("recognized speech:", transcript)
                scrape_transcript_for_commands(transcript)
            else:
                print("could not recognize speech. Try again!")
    except KeyboardInterrupt:
        print("Quitting Application") 
    # system_reply("In order to do this, press the commnand and t keys on your keyboard")
    pass