import cv2
for i in range(5):  # 尝试0-4索引
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头索引 {i} 可用")
        cap.release()


# import cv2
#
# # 视频流 URL (替换为你的 ESP32-CAM 的 IP 地址)
# url = "http://192.168.1.6:81/stream"
#
# # 打开视频流
# cap = cv2.VideoCapture(url)
#
# # 检查是否成功打开视频流
# if not cap.isOpened():
#     print("Error: Couldn't open video stream.")
#     exit()
#
# while True:
#     # 读取视频帧
#     ret, frame = cap.read()
#
#     # 如果读取失败，退出
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break
#
#     # 显示视频帧
#     cv2.imshow("ESP32-CAM Stream", frame)
#
#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()



