import pygame
import serial
import time

ser = serial.Serial('COM7', 9600)
time.sleep(2)  # 等待 Arduino 初始化

# 初始化手柄
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

print("按 A 点亮LED，按 B 熄灭LED")

while True:
    pygame.event.pump()
    a = joystick.get_button(0)  # A 按钮
    b = joystick.get_button(1)  # B 按钮

    if a:
        ser.write(b'1')   # 发送 '1'，开灯
        print("LED ON")
        time.sleep(0.2)

    elif b:
        ser.write(b'0')   # 发送 '0'，关灯
        print("LED OFF")
        time.sleep(0.2)
