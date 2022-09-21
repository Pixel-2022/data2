import cv2
import mediapipe as mp
import numpy as np
import time, os
import itertools

import os

dir_path = "CameraRoll"

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_path2 = file_path.split("\\")
        file_path3=file_path2[1].split(".")
        print(file_path3[0])
        print(file_path)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic

        actions = [file_path3[0]]
        seq_length = 30
        secs_for_action = 100
        start_frame = 5

        created_time = int(time.time())

        video = cv2.VideoCapture(file_path)

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        result = []

        all=[]

        #LEFT_EYE_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_LEFT_EYE)))
        #RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)))
        #MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        #POSE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]

        RIGHT_EYE_INDEXES = [46,53,52,65,55,33,246,161,160,159,158,157,173,133,7,163,144,145,153,154,155]
        LEFT_EYE_INDEXES = [276,283,282,295,285,362,398,384,385,386,387,388,466,263,382,381,380,374,373,390,249]
        MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        a=0

        for idx, action in enumerate(actions):
            idx=0
            ret, img = video.read()

            cv2.waitKey(1000)
            num = 0
            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                ret, img = video.read()
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
                        print("right_angle")

                    else:
                        joint2 = np.zeros((21, 4))
                        label=[0]*15
                        d2=np.concatenate([joint2.flatten(),label])
                        hands.extend(d2)


                    #hands = np.array(hands)
                    #np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), hands)

                    


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
                        print("pose")

                    else:
                        joint3 = np.zeros((6, 4))
                        label=[0]*2
                        d3=np.concatenate([joint3.flatten(),label])
                        hands.extend(d3)
                    


                    # 얼굴
                    #if results.face_landmarks:
                    #    landmarks = results.face_landmarks.landmark
                        
                    #    pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)

                    #    joint4=np.zeros((pointSize,4))
                        
                    #    for j, lm in enumerate(landmarks):
                    #        if j in MOUTH:
                    #            joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                    #            a+=1
                    #        if j in LEFT_EYE_INDEXES:
                    #            joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                    #            a+=1
                    #        if j in RIGHT_EYE_INDEXES:
                    #            joint4[a] = [lm.x, lm.y, lm.z, lm.visibility]
                    #            a+=1
                        #idx_label=[idx]*pointSize
                        #d4=np.concatenate([joint4.flatten(),idx_label])
                    #    d4=joint4.flatten()
                    #    d4 = np.append(d4, idx)
                #     print(len(d4))
                #     hands.extend(d4)
                #     print("face")

                # else:
                    #    pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
                    #   joint4=np.zeros((pointSize,4))
                    #   d4=joint4.flatten()
                    #   d4 = np.append(d4, idx)
                    #   hands.extend(d4)
                    # 얼굴
                    if results.face_landmarks:
                        landmarks = results.face_landmarks.landmark
                
                        pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)

                        joint4 = np.zeros((pointSize, 4))
                        for j, lm in enumerate(landmarks):
                            if j in [46,53,52,65,55,276,283,282,295,285, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,33,246,161,160,159,158,157,173,133, 7,163,144,145,153,154,155, 362,398,384,385,386,387,388,466,263, 382,381,380,374,373,390,249]:
                                if j==46: x=0
                                if j==53: x=1
                                if j==52: x=2
                                if j==65: x=3
                                if j==55: x=4

                                if j==276: x=5
                                if j==283: x=6
                                if j==282: x=7
                                if j==295: x=8
                                if j==285: x=9

                                if j==78: x=10
                                if j==191: x=11
                                if j==80: x=12
                                if j==81: x=13
                                if j==82: x=14
                                if j==13: x=15
                                if j==312: x=16
                                if j==311: x=17
                                if j==310: x=18
                                if j==415: x=19
                                if j==308: x=20

                                if j==95: x=21
                                if j==88: x=22
                                if j==178: x=23
                                if j==87: x=24
                                if j==14: x=25
                                if j==317: x=26
                                if j==402: x=27
                                if j==318: x=28
                                if j==324: x=29
                                
                                if j==33: x=30
                                if j==246: x=31
                                if j==161: x=32
                                if j==160: x=33
                                if j==159: x=34
                                if j==158: x=35
                                if j==157: x=36
                                if j==173: x=37
                                if j==133: x=38
                                if j==7: x=39
                                if j==163: x=40
                                if j==144: x=41
                                if j==145: x=42
                                if j==153: x=43
                                if j==154: x=44
                                if j==155: x=45

                                if j==362: x=46
                                if j==398: x=47
                                if j==384: x=48
                                if j==385: x=49
                                if j==386: x=50
                                if j==387: x=51
                                if j==388: x=52
                                if j==466: x=53
                                if j==263: x=54
                                if j==382: x=55
                                if j==381: x=56
                                if j==380: x=57
                                if j==374: x=58
                                if j==373: x=59
                                if j==390: x=60
                                if j==249: x=61  

                        
                                joint4[x] = [lm.x, lm.y, lm.z, 0]

                
                
                        #v_face1 = joint4[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 46, 55, 56, 57, 58, 59, 60, 61], :3]
                        #v_face2 = joint4[[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 45, 38, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 54], :3]
                        v_face1 = joint4[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 46, 55, 56, 57, 58, 59, 60, 61], :3]
                        v_face2 = joint4[[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 38, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 54], :3]
                        v_face3 =v_face2-v_face1
                        v_face3 = v_face3 / np.linalg.norm(v_face3, axis=1)[:, np.newaxis]

                        angle4 = np.arccos(np.einsum('nt,nt->n',
                                            v_face3[[0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58], :],
                                            v_face3[[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59],  :]))
                        angle4 = np.degrees(angle4)

                        d4=np.concatenate([joint4.flatten(), angle4])

                        a=0
                        hands.extend(d4)
                        print("face")

                    else:
                        pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
                        joint4=np.zeros((pointSize,4))

                        label=[0]*52
                        d4=np.concatenate([joint4.flatten(),label])
                
                        #print(len(d4))
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
            np.save(os.path.join('dataset2', f'raw_{action}'), result)
            # Create sequence data
            full_seq_data = []
            for seq in range(len(result) - seq_length):
                full_seq_data.append(result[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('dataset2', f'seq_{action}'), full_seq_data)
