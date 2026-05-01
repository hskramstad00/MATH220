from djitellopy import Tello
from time import sleep


tello = Tello()

print(f'Battery: {tello.get_battery()}')


tello.takeoff()

sleep(3)

tello.land()