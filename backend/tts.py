import pyttsx3

engine = pyttsx3.init()
engine.setProperty('voice', 'french')

def speak(text: str):
    engine.say(text)
    engine.runAndWait()
