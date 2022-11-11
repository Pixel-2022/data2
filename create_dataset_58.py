import cv2
import mediapipe as mp
import numpy as np
import time, os
import itertools

import os
dir_path = "hey"
for (root, directories, files) in os.walk(dir_path):
    for dir in directories:
        d_path=os.path.join(root,dir)
        d_path2 = d_path.split("\\")
        d_path2=d_path2[1]
        idx=float(d_path2)
        for(root2, directories, files)in os.walk(d_path):
            for file in files:
                file_path = os.path.join(root2, file)
                file_path2 = file_path.split("\\")
                file_path3=file_path2[2].split(".")
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

                #이전 손목 값
                last_left=180
                last_right=180

                for idxx, action in enumerate(actions):
                    idxx=0
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

                            print(num)

                            keypoint_pos1 = []
                            #왼손 오른손 유무
                            data_left = 0
                            data_right = 0

                            hands=[]
                            #연결부분 손(왼손0,오른손1),포즈(11,12,13,14,15,16->2,3,4,5,6,7),왼쪽 눈끝(8)
                            joint_concat = np.zeros((6, 4))
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

                                #연결부분 손(왼손0,오른손1),포즈(11,12,13,14,15,16->2,3,4,5,6,7),왼쪽 눈끝(8)
                                joint_concat[0] = joint[9]


                                hands.extend(angle)

                                # 학습 input인 npy 파일 형태로 만들어서 저장
                                #data_left = np.array(data_left)
                                #np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data_left)
                                print("left_angle")


                            else:
                                data_left=1

                            
                                
                                joint = np.zeros((21, 4))
                                joint_concat[0] = joint[9]
                                #왼손 없을때 추가되는 각도
                                label=[0]*15
                                hands.extend(label)

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
                                #연결부분 손(왼손0,오른손1),포즈(11,12,13,14,15,16->2,3,4,5,6,7),왼쪽 눈끝(8)
                                joint_concat[1] = joint2[9]
                                hands.extend(angle2)
                                print("right_angle")

                            else:
                                if data_left==1:
                                    continue
                                else:
                                    data_right=1
                                joint2 = np.zeros((21, 4))
                                joint_concat[1] = joint2[9]
                                #오른손 없을 때 추가되는 각도
                                label=[0]*15
                                hands.extend(label)


                            #hands = np.array(hands)
                            #np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), hands)

                            


                            # POSE
                            if results.pose_landmarks:
                                landmarks = results.pose_landmarks.landmark

                                joint3 = np.zeros((8, 4))
                                for j, lm in enumerate(landmarks):
                                    if j in [11,12,13,14,15,16]:
                                        if j==11: j=0
                                        if j==12: j=1
                                        if j==13: j=2
                                        if j==14: j=3
                                        if j==15: j=4
                                        if j==16: j=5
                                        
                                        joint3[j] = [lm.x, lm.y, lm.z, lm.visibility]

                                joint3[6]=joint3[0]
                                joint3[6][1]=joint3[6][1]+0.05
                                joint3[7]=joint3[1]
                                joint3[7][1]=joint3[7][1]+0.05

                                v6 = joint3[[1, 3, 0, 2, 7, 6], :3]
                                v7 = joint3[[3, 5, 2, 4, 1, 0], :3]
                                v8=v7-v6
                                v8 = v8 / np.linalg.norm(v8, axis=1)[:, np.newaxis]


                                angle3 = np.arccos(np.einsum('nt,nt->n',
                                                            v8[[0, 2, 4, 5], :],
                                                            v8[[1, 3, 0, 2],  :]))
                                #라디안->도
                                angle3 = np.degrees(angle3)
                                # angle3_label = np.array([angle3], dtype=np.float32)
                                # angle3_label = np.append(angle3_label, idx)
                                #print("길이",len(angle3_label))

                                 #연결부분 손(왼손0,오른손1),포즈(11,12,13,14,15,16->2,3,4,5,6,7),왼쪽 눈끝(8)
                                
                            
                                # joint_concat[2]=joint3[0]
                                # joint_concat[3]=joint3[1]
                                joint_concat[2]=joint3[2]
                                joint_concat[3]=joint3[3]
                                joint_concat[4]=joint3[4]
                                joint_concat[5]=joint3[5]

                                hands.extend(angle3)
                                print("pose")

                            else:
                                joint3 = np.zeros((8, 4))
                                joint_concat[2]=joint3[2]
                                joint_concat[3]=joint3[3]
                                joint_concat[4]=joint3[4]
                                joint_concat[5]=joint3[5]

                                label=[0]*4

                                hands.extend(label)
                            


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
                                    # if j in [46,53,52,65,55,276,283,282,295,285, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,33,246,161,160,159,158,157,173,133, 7,163,144,145,153,154,155, 362,398,384,385,386,387,388,466,263, 382,381,380,374,373,390,249]:
                                    if j in [46,53,52,65,55,276,283,282,295,285, 78, 81, 13, 311, 308, 178, 14, 402, 33,159,133,145, 362,386,263, 374]:
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
                                        # if j==191: x=11
                                        # if j==80: x=12
                                        if j==81: x=11
                                        # if j==82: x=14
                                        if j==13: x=12
                                        # if j==312: x=16
                                        if j==311: x=13
                                        # if j==310: x=18
                                        # if j==415: x=19
                                        if j==308: x=14

                                        # if j==95: x=21
                                        # if j==88: x=22
                                        if j==178: x=15
                                        # if j==87: x=24
                                        if j==14: x=16
                                        # if j==317: x=26
                                        if j==402: x=17
                                        # if j==318: x=28
                                        # if j==324: x=29
                                        
                                        if j==33: x=18
                                        # if j==246: x=31
                                        # if j==161: x=32
                                        # if j==160: x=33
                                        if j==159: x=19
                                        # if j==158: x=35
                                        # if j==157: x=36
                                        # if j==173: x=37
                                        if j==133: x=20
                                        # if j==7: x=39
                                        # if j==163: x=40
                                        # if j==144: x=41
                                        if j==145: x=21
                                        # if j==153: x=43
                                        # if j==154: x=44
                                        # if j==155: x=45

                                        if j==362: x=22
                                        # if j==398: x=47
                                        # if j==384: x=48
                                        # if j==385: x=49
                                        if j==386: x=23
                                        # if j==387: x=51
                                        # if j==388: x=52
                                        # if j==466: x=53
                                        if j==263: x=24
                                        # if j==382: x=55
                                        # if j==381: x=56
                                        # if j==380: x=57
                                        if j==374: x=25
                                        # if j==373: x=59
                                        # if j==390: x=60
                                        # if j==249: x=61  

                                
                                        joint4[x] = [lm.x, lm.y, lm.z, 0]

                        
                    
                                # v_face1 = joint4[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 46, 55, 56, 57, 58, 59, 60, 61], :3]
                                # v_face2 = joint4[[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 38, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 54], :3]
                                v_face1 = joint4[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17, 16, 15, 18, 19, 20, 21, 22, 23, 24, 25], :3]
                                v_face2 = joint4[[1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 17, 16, 15, 10, 19, 20, 21, 18, 23, 24, 25, 22], :3]
                                v_face3 =v_face2-v_face1
                                v_face3 = v_face3 / np.linalg.norm(v_face3, axis=1)[:, np.newaxis]

                                angle4 = np.arccos(np.einsum('nt,nt->n',
                                                    v_face3[[0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12,13,14,15, 16,17,18,19, 20,21,22,23], :],
                                                    v_face3[[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13,14,15,8, 17,18,19,16, 21,22,23,20],  :]))
                                angle4 = np.degrees(angle4)

                                #연결부분 손(왼손0,오른손1),포즈(11,12,13,14,15,16->2,3,4,5,6,7),왼쪽 눈끝(8)
                                # joint_concat[8] = joint4[263]

                                a=0
                                hands.extend(angle4)
                                print("face")

                            else:
                                pointSize=len(LEFT_EYE_INDEXES)+len(RIGHT_EYE_INDEXES)+len(MOUTH)
                                joint4=np.zeros((pointSize,4))
                                # joint_concat[8] = joint4[263]
                                

                                label=[0]*22
                                hands.extend(label)

                        

                        #연결부분 손(왼손0,오른손1),포즈(13,14,15,16->2,3,4,5)
                        
                        #각도 순서 오른팔부터손(14(3),16(5),1), 왼팔부터 손(13(2),15(4),0), 오른쪽어깨왼쪽어깨왼쪽눈끝(12,11,8)
                        # if joint_concat[0]==np.zeros((1,4)):

                        v9_concat = joint_concat[[3,5,2,4], :3]
                        v10_concat = joint_concat[[5,1,4,0], :3]
                        v11_concat = v10_concat-v9_concat
                        v11_concat=v11_concat/np.linalg.norm(v11_concat, axis=1)[:, np.newaxis]
                        angle5=np.arccos(np.einsum('nt,nt->n',
                                                            v11_concat[[0, 2], :],
                                                            v11_concat[[1, 3],  :]))
                        angle5 = np.degrees(angle5)

                        if data_left==1:
                            angle5[0]=last_left
                            last_right=angle5[1]
                        elif data_right==1:
                            angle5[1]=last_right
                            last_left=angle5[0]
                        else:
                            last_right=angle5[1]
                            last_left=angle5[0]

                        
                        hands.extend(angle5)


                        hands.extend([idx])
                        all.append(hands)
                        num += 1
                        a=0

                    #result = np.array(result)
                    result = np.array(all)
                    print(idx)
                    # np.save(os.path.join('dataset2', f'raw_{action}'), result)
                    # Create sequence data
                    full_seq_data = []
                    for seq in range(len(result) - seq_length):
                        full_seq_data.append(result[seq:seq + seq_length])

                    full_seq_data = np.array(full_seq_data)
                    print(action, full_seq_data.shape)
                    np.save(os.path.join('hey2', f'seq_{action}'), full_seq_data)
