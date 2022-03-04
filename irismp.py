import cv2 as cv
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
#right eye
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
#Iris Part
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]

#video
cap = cv.VideoCapture(0)
#image
# cap = cv.imread("face115.jpeg")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
   while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame,1)
    rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    img_h,img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_points=np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        # cv.polylines(frame,[mesh_points[LEFT_EYE]],True,(0,255,0),1,cv.LINE_AA)
        # cv.polylines(frame,[mesh_points[RIGHT_EYE]],True,(0,255,0),1,cv.LINE_AA)
        #iris
        # cv.polylines(frame,[mesh_points[LEFT_IRIS]],True,(0,255,0),1,cv.LINE_AA)
        # cv.polylines(frame,[mesh_points[RIGHT_IRIS]],True,(0,255,0),1,cv.LINE_AA)
        (l_cx,l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx,l_cy],dtype=np.int32)
        center_right = np.array([r_cx,r_cy],dtype=np.int32)
        cv.circle(frame,center_left,int(l_radius),(255,0,255),1,cv.LINE_AA)
        cv.circle(frame,center_right,int(r_radius),(255,0,255),1,cv.LINE_AA)
        #mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv.circle(mask, center_left, int(l_radius), (255,0,0), -1, cv.LINE_AA)
        cv.circle(mask, center_right, int(r_radius), (255,0,0), -1, cv.LINE_AA)
 
        #coloring mask on video
        # into_hsv =cv.cvtColor(lframe,cv.COLOR_BGR2HSV)
        # L_limit=np.array([98,50,50]) # setting the blue lower limit
        # U_limit=np.array([139,255,255]) # setting the blue upper limit
        # l_mask=cv.inRange(into_hsv,L_limit,U_limit)
        # colored =cv.bitwise_and(lframe,lframe,mask=l_mask)

    # cv.imshow('colored mask',colored)
    cv.imshow('Mask', mask)
    cv.imshow('img',frame)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
