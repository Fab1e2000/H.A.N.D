import pygame
import serial
import time

control = serial.Serial('COM7',9600)
time.sleep(2)

pygame.init()
pygame.joystick.init()
controller = pygame.joystick.Joystick(0)
controller.init()

print("controller initiated")

last_angle = -1

while True:
    pygame.event.pump()

    rt = controller.get_axis(5)
    rt_normalized = (rt+1)/2
    angle = int(rt_normalized*160)

    if angle != last_angle:#为了防止
        control.write(f"{angle}\n".encode())
        print(f"send angle: {angle}")
        last_angle = angle

    time.sleep(0.2)





