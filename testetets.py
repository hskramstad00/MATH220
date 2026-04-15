from djitellopy import Tello
from time import sleep

tello = Tello()
tello.connect()
tello.takeoff()
sleep(5)
tello.land()

