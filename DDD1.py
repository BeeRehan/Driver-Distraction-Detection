import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
import pygame


pygame.init()
war=pygame.mixer.Sound("src/audio.wav")
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3
EAR_AVG = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('src/shape_predictor_68_face_landmarks.dat')
COUNTER = 0
TOTAL = 0


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while True:
        ret, frame = cap.read()
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if ret:
        # convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects= detector(gray, 0)
        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()
             # get the facial landmarks
        landmarks = np.matrix([[p.x, p.y] 
            for p in predictor(frame, rect).parts()])
            # get the left eye landmarks
                left_eye = landmarks[LEFT_EYE_POINTS]
            # get the right eye landmarks
                right_eye = landmarks[RIGHT_EYE_POINTS]
            # draw contours on the eyes
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)  #(image, [contour], all_contours, color, thickness)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            # compute the EAR for the left eye
                ear_left = eye_aspect_ratio(left_eye)
                # compute the EAR for the right eye
                ear_right = eye_aspect_ratio(right_eye)
            # compute the average EAR
                ear_avg = (ear_left + ear_right) / 2.0
            #frame = cv2.flip(frame,0)

        # write the flipped frame
            #out.write(frame)
            if ear_avg < EYE_AR_THRESH:
                a="Distraction"
                cv2.putText(frame, "Status {}".format(a), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
                print("focused")                
                COUNTER += 1
                war.play()
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1                    
                COUNTER = 0
                a="Focused"
                print("Distracted")
                cv2.putText(frame, "Status {}".format(a), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, "No.of.Distraction{}".format(TOTAL), (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, "EAR {}".format(ear_avg), (10, 120), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
        if (TOTAL>=10):
            war.play()
           cv2.putText(frame, "Warning {}".format("Stop the Vechile"), (160, 200), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow("L2R presents", frame)
        key = cv2.waitKey(1) & 0xFF
        # When key 'Q' is pressed, exit
        if key is ord('q'):
            break
cap.release()
out.release()
# destroy all windows
cv2.destroyAllWindows()

