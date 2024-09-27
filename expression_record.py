import cv2
import mediapipe as mp
import numpy as np
import os

# 初始化MediaPipe的面部关键点检测器
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# 获取用户输入的表情
expression = input("Enter the expression you want to capture: ")

# 创建保存关键点的目录
os.makedirs(f'dataset/{expression}', exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(1)

# 图像计数器
img_counter = len(os.listdir(f'dataset/{expression}'))

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 转换为RGB图像
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 检测面部关键点
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 将关键点转换为numpy数组
            landmarks = np.array([(lmk.x, lmk.y, lmk.z) for lmk in face_landmarks.landmark])
            # 绘制关键点
            for (x, y, z) in landmarks:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)

    # 显示图像
    cv2.imshow('Press Space to take a photo, press Esc to quit', frame)

    # 检查是否按下了键盘上的某个键
    k = cv2.waitKey(1)

    if k%256 == 27:
        # 如果按下了Esc键，退出循环
        print("Escape hit, closing...")
        break
    elif k%256 == ord('h'):
        np.save(f'dataset/{expression}/face_landmarks_{img_counter}.npy', landmarks)
        print(f'Landmarks saved as dataset/{expression}/face_landmarks_{img_counter}.npy')
        img_counter += 1

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()