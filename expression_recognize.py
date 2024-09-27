import cv2
import mediapipe as mp
import numpy as np
from joblib import load



# 初始化MediaPipe的面部关键点检测器和面部检测器
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# 加载模型
clf = load('bp_model.joblib')

# 打开摄像头
cap = cv2.VideoCapture(1)

while True:
    # 读取一帧图像
    ret, img = cap.read()

    # 转换为RGB图像
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测面部关键点
    results = face_mesh.process(rgb)

    # 检测面部
    detection_results = face_detection.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 将关键点转换为numpy数组
            landmarks = np.array([(lmk.x, lmk.y, lmk.z) for lmk in face_landmarks.landmark])
            # 预测表情
            expression = clf.predict([landmarks.flatten()])

            #print(f'Predicted expression: {expression[0]}')

    if detection_results.detections:
        for detection in detection_results.detections:
            # 获取面部的边界框
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # 绘制矩形和文本
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, expression[0], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # 显示图像
    cv2.imshow('Press Esc to quit', img)

    # 检查是否按下了键盘上的某个键
    k = cv2.waitKey(1)

    if k%256 == 27:
        # 如果按下了Esc键，退出循环
        print("Escape hit, closing...")
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()