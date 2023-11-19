#packages used for the project
import face_recognition
import cv2
import numpy as np
import serial 
import time
import os 
#setting up the file to hold the unknown person's video 
filename = 'video.avi'

#setting the parameters for the video
frames_per_second = 24.0

my_res = '480p' 

def change_res(video_stream, width, height):
    video_stream.set(3,width)
    video_stream.set(4,height)
    
Dimensions = {
    "480p":(640, 480),
    "720p":(1280, 720),
    "1080p":(1920, 1080),
    "4k":(3840, 2160)}

def get_dims(video_stream, res='1080p'):
    width, height = Dimensions['480p']
    if res in Dimensions:
        width, height = Dimensions[res]
    change_res(video_stream, width, height)
    return width, height

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

#hog training model is used as it runs better on lower end hardware
Model= "hog"
#video stream variable for the webcam
video_stream= cv2.VideoCapture(0)
#obtaining the data from the webcam when an unknown person is detected
dims = get_dims(video_stream, res=my_res) 

video_type_cv2= get_video_type(filename)

out = cv2.VideoWriter(filename, video_type_cv2,frames_per_second,dims)
#connecting the arduino to python
arduino = serial.Serial('COM6', 9600, timeout=.1)

time.sleep(2)

print("connected to arduino")
#passing the images of the people who will be known
hurtado_image = face_recognition.load_image_file("hurtado.jpg")

hurtado_face_encoding = face_recognition.face_encodings(hurtado_image)[0]

musk_image = face_recognition.load_image_file("Bill.jpg")

musk_face_encoding = face_recognition.face_encodings(musk_image)[0]
#an array of the known people's encodings and a string for their names
known_face_encodings = [
    hurtado_face_encoding,
    musk_face_encoding
]
known_face_names = [
    "Miguel Hurtado",
    "Bill Gates"
]

locations = []

encodings = []

face_names = []

process_this_frame = True

while True:
    
    ret, image = video_stream.read()
    #resize the image
    re_image = cv2.resize(image,(0,0),fx=0.25,fy=0.25)
    
    #convert the image to rgb
    
    rgb_re_image= cv2.cvtColor(re_image,cv2.COLOR_BGR2RGB)
    
    if process_this_frame:
        #finding the face locations and encodings
        locations= face_recognition.face_locations(rgb_re_image,model=Model)
        encodings= face_recognition.face_encodings(rgb_re_image,locations)
        
        face_names = []
        
        for face_encoding in encodings:
            #seeing if the face that is detected has a match
            match = face_recognition.compare_faces(known_face_encodings,face_encoding)
            
            name = "unknown"
            #if statement is used to determine what happens when it is a match or not
            if True in match:
                print("Door is opening (for 5 seconds)")
                #A '0' is sent to the arduino to close the door
                arduino.write(b'0')
                #5 seconds for the person to open the door
                time.sleep(5)
                print("Door is closing")
            elif name == "unknown":
                print("Unknown Person Detected")
                out.write(image)
                
            #A '1' is sent to the arduino to close the door
            arduino.write(b'1')
            distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            #The code below is for drawing a box around the persons face
            best_match = np.argmin(distance)
            
            if match[best_match]:
                name=known_face_names[best_match]
                #finding the persons name
            face_names.append(name)
    process_this_frame = not process_this_frame
         #using the location of the face and the name to make a box around the face
    for (top, right, bottom, left), name in zip(locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        #rectangle is made around the face with the text of the persons name beneath it
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #shows the feed of the webcam to the user
    cv2.imshow('Video', image)
    #user can exit the program by pressing the q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
out.release()
cv2.destroyAllWindows()
        
