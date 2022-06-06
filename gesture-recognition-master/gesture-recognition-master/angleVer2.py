import cv2
import mediapipe as mp
import numpy as np
import time, os
import itertools

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

actions = ['구경D']
seq_length = 30
secs_for_action = 100
start_frame = 5

created_time = int(time.time())

vidio = cv2.VideoCapture('NIA_SL_WORD1503_REAL01_D_구경.mp4')

vidio.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

result = []

all=[]

LEFT_EYE_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)))
MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]
POSE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
a=0
for idx, action in enumerate(actions):
    idx=0
    ret, img = vidio.read()

    cv2.waitKey(1000)
    num = 0
    start_time = time.time()

    while time.time() - start_time < secs_for_action:
        ret, img = vidio.read()
        if not ret: break
        
        with mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                refine_face_landmarks=True,
                min_detection_confidence=0.1,
                min_tracking_confidence=0.1) as holistic:

            image_height, image_width, _ = img.shape
            results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            annotated_image = img.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(img.shape, dtype=np.uint8)
            bg_image[:] = (255, 255, 255)
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose, left and right hands, and face landmarks on the image.

            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                    get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                    get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                    get_default_pose_landmarks_style())

            print(num)

            keypoint_pos1 = []
            data_left = []
            data_right = []
            hands=[]

            # 왼손
            if results.left_hand_landmarks:
                landmarks = results.left_hand_landmarks.landmark

                joint = np.zeros((21, 4))
                for j, lm in enumerate(landmarks):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        :3]  # Child joint
                v = v2 - v1

                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)


                d = np.concatenate([joint.flatten(), angle])
                print(len(d))

                hands.extend(d)

                # 학습 input인 npy 파일 형태로 만들어서 저장
                #data_left = np.array(data_left)
                #np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data_left)
                print("left_angle")

            else:
                joint = np.zeros((21, 4))
                label=[0]*15
                d=np.concatenate([joint.flatten(),label])
                hands.extend(d)

            # 오른손
            if results.right_hand_landmarks:
                landmarks = results.right_hand_landmarks.landmark

                joint2 = np.zeros((21, 4))
                for j, lm in enumerate(landmarks):
                    joint2[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v3 = joint2[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                        :3]  # Parent joint
                v4 = joint2[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        :3]  # Child joint
                v5 = v4 - v3

                v5 = v5 / np.linalg.norm(v5, axis=1)[:, np.newaxis]

                angle2 = np.arccos(np.einsum('nt,nt->n',
                                                v5[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v5[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle2 = np.degrees(angle2)

                d2 = np.concatenate([joint2.flatten(), angle2])
                print(len(d2))

                hands.extend(d2)

            else:
                joint2 = np.zeros((21, 4))
                label=[0]*15
                d2=np.concatenate([joint2.flatten(),label])
                hands.extend(d2)


            #hands = np.array(hands)
            #np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), hands)

            print("right_angle")


            # POSE
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                joint3 = np.zeros((6, 4))
                for j, lm in enumerate(landmarks):
                    if j in [11,12,13,14,15,16]:
                        if j==11: j=0
                        if j==12: j=1
                        if j==13: j=2
                        if j==14: j=3
                        if j==15: j=4
                        if j==16: j=5
                        
                        joint3[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v6 = joint3[[1, 3, 0, 2], :3]
                v7 = joint3[[3, 5, 2, 4], :3]
                v8=v7-v6
                v8 = v8 / np.linalg.norm(v8, axis=1)[:, np.newaxis]


                angle3 = np.arccos(np.einsum('nt,nt->n',
                                            v8[[0, 2], :],
                                            v8[[1, 3],  :]))
                angle3 = np.degrees(angle3)
                # angle3_label = np.array([angle3], dtype=np.float32)
                # angle3_label = np.append(angle3_label, idx)
                #print("길이",len(angle3_label))
                d3 = np.concatenate([joint3.flatten(), angle3])
                print(len(d3))

                hands.extend(d3)

            else:
                joint3 = np.zeros((6, 4))
                label=[0]*2
                d3=np.concatenate([joint3.flatten(),label])
                hands.extend(d3)


            # 얼굴
            if results.face_landmarks:
                landmarks = results.face_landmarks.landmark
                
                pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)

                joint4=np.zeros((pointSize,4))
                
                for j, lm in enumerate(landmarks):
                    if j in MOUTH:
                        joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                        a+=1
                    if j in LEFT_EYE_INDEXES:
                        joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                        a+=1
                    if j in RIGHT_EYE_INDEXES:
                        joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                        a+=1
                #idx_label=[idx]*pointSize
                #d4=np.concatenate([joint4.flatten(),idx_label])
                d4=joint4.flatten()
                d4 = np.append(d4, idx)
                print(len(d4))
                hands.extend(d4)

            else:
                pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
                joint4=np.zeros((pointSize,4))
                d4=joint4.flatten()
                d4 = np.append(d4, idx)
                hands.extend(d4)

            #     for i in MOUTH:
            #         x = int(landmarks[i].x * img.shape[1])
            #         y = int(landmarks[i].y * img.shape[0])
            #         img = cv2.line(img, (int(x), int(y)), (int(x), int(y)), (0, 0, 255), 5)
            #         keypoint_pos1.append((landmarks[i].x, landmarks[i].y))
            #     for i in LEFT_EYE_INDEXES:
            #         x = int(landmarks[i].x * img.shape[1])
            #         y = int(landmarks[i].y * img.shape[0])
            #         img = cv2.line(img, (int(x), int(y)), (int(x), int(y)), (0, 0, 255), 5)
            #         keypoint_pos1.append((landmarks[i].x, landmarks[i].y))
            #     for i in RIGHT_EYE_INDEXES:
            #         x = int(landmarks[i].x * img.shape[1])
            #         y = int(landmarks[i].y * img.shape[0])
            #         img = cv2.line(img, (int(x), int(y)), (int(x), int(y)), (0, 0, 255), 5)
            #         keypoint_pos1.append((landmarks[i].x, landmarks[i].y))

            # if results.pose_landmarks:
            #     landmarks = results.pose_landmarks.landmark
            #     for i in POSE:
            #         x = int(landmarks[i].x * img.shape[1])
            #         y = int(landmarks[i].y * img.shape[0])
            #         img = cv2.line(img, (int(x), int(y)), (int(x), int(y)), (0, 0, 255), 5)


            #result.append(keypoint_pos1)
            #print("왼손 각도", angle)
            #print("오른손 각도", angle2)
            #print("pose 각도", d3)
            #print("얼굴", d4)
            # cv2.imwrite('tmp4/'+str(num)+'.png',img)
            all.append(hands)
            num += 1
        a=0

    #result = np.array(result)
    result = np.array(all)
    np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), result)
    # Create sequence data
    full_seq_data = []
    for seq in range(len(result) - seq_length):
        full_seq_data.append(result[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)
    np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)

