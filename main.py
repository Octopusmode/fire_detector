import numpy as np
import tensorflow as tf
import cv2
import time
print(tf.__version__)

Model_Path = r"data\fire_detector-fp16.tflite"
Video_path = r"rammstein.mp4"

interpreter = tf.lite.Interpreter(model_path=Model_Path)
interpreter.allocate_tensors()

print(interpreter.get_input_details())

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['fire', 'other', 'smoke']

cap = cv2.VideoCapture(Video_path)
ok, frame_image = cap.read()
original_image_height, original_image_width, _ = frame_image.shape
thickness = original_image_height // 500  
fontsize = original_image_height / 1500
# print(thickness)
# print(fontsize)

while True:
    ok, frame_image = cap.read()
    if not ok:
        break

    model_interpreter_start_time = time.time()
    resize_img = cv2.resize(frame_image, (416, 416), interpolation=cv2.INTER_CUBIC)
    reshape_image = resize_img.reshape(416, 416, 3)
    image_np_expanded = np.expand_dims(reshape_image, axis=0)
    image_np_expanded = image_np_expanded.astype('float32')  # float32

    interpreter.set_tensor(input_details[0]['index'], image_np_expanded) 
    interpreter.invoke()

    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # output_data_1 = interpreter.get_tensor(output_details[1]['index']) 
    # output_data_2 = interpreter.get_tensor(output_details[2]['index'])
    # output_data_3 = interpreter.get_tensor(output_details[3]['index'])  
    each_interpreter_time = time.time() - model_interpreter_start_time
    
    print(output_details)

    # for i in range(len(output_data_1[0])):
    #     confidence_threshold = output_data_2[0][i]
    #     if confidence_threshold > 0.3:
    #         label = "{}: {:.2f}% ".format(class_names[int(output_data_1[0][i])], output_data_2[0][i] * 100) 
    #         label2 = "inference time : {:.3f}s" .format(each_interpreter_time)
    #         left_up_corner = (int(output_data[0][i][1]*original_image_width), int(output_data[0][i][0]*original_image_height))
    #         left_up_corner_higher = (int(output_data[0][i][1]*original_image_width), int(output_data[0][i][0]*original_image_height)-20)
    #         right_down_corner = (int(output_data[0][i][3]*original_image_width), int(output_data[0][i][2]*original_image_height))
    #         cv2.rectangle(frame_image, left_up_corner_higher, right_down_corner, (0, 255, 0), thickness)
    #         cv2.putText(frame_image, label, left_up_corner_higher, cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), thickness=thickness)
    #         cv2.putText(frame_image, label2, (30, 30), cv2.FONT_HERSHEY_DUPLEX, fontsize, (255, 255, 255), thickness=thickness)
    # cv2.namedWindow('detect_result', cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow('detect_result', 800, 600)
    # cv2.imshow("detect_result", frame_image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == 32:
        cv2.waitKey(0)
        continue
cap.release()
cv2.destroyAllWindows()