import picamera
from time import sleep

camera = picamera.PiCamera()
camera.color_effects = (128,128)
path = '/home/pi/Documents/images/'
# camera.capture('/home/pi/Desktop/img3.jpg')
camera.start_preview()
sleep(10)
for i in range(10):
    camera.capture(path + 'max_2_' + str(i) + '.jpg')
    sleep(0.4)
