import cv2 as cv
import face_recognition
import numpy as np
import csv
import datetime

if __name__=='__main__':
    try:
        video_capture=cv.VideoCapture(0)

        deep_img=face_recognition.load_image_file('faces/deep.jpg')
        deep_encoding=face_recognition.face_encodings(deep_img)[0]

        known_face_encoding=[deep_encoding]
        known_face_name=['deep']

        employees=known_face_name.copy()

        face_locations=[]
        face_encodings=[]

        time=datetime.datetime.now()
        current_date=time.strftime("%Y-%m-%d")

        f=open(f"{current_date}.csv","a+",newline="")
        lnwriter=csv.writer(f)

        while True:
            _,frame=video_capture.read()
            small_frame=cv.resize(frame,(0,0),fx=0.25,fy=0.25)
            rgb_small=cv.cvtColor(small_frame,cv.COLOR_BGR2RGB)

            recognize_faces=face_recognition.face_locations(rgb_small)
            face_encodings=face_recognition.face_encodings(rgb_small,face_locations)

            cv.imshow('Attendence',frame)

            for face_encoding in face_encodings:
                matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
                face_distance=face_recognition.face_distance(known_face_encoding,face_encodings)

                best_match_index=np.argmin(face_distance)

                if(matches[best_match_index]):
                    name=known_face_name[best_match_index]
                
                cv.imshow('Attendence',frame)     

                if cv.waitKey(1) & 0xFF==ord('q'):
                    break
                                              
            video_capture.release()
            cv.destroyAllWindows()
            f.close()
 




    except Exception as e:
        print(f"Some Error Occured {e}")



 