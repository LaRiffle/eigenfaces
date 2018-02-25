from pynput import keyboard
import picamera
from time import sleep

name_of_subject = 'max'

camera = picamera.PiCamera()
camera.color_effects = (128,128)
path = '/home/pi/Documents/images/'

camera.start_preview()

def on_press(key):
    for i in range(100):
        camera.capture(path + name_of_subject + '_' + str(i) + '.jpg')
        sleep(0.5)
    return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()