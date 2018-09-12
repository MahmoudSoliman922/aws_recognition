import face_recognition
import cv2
import requests
import boto3
import threading
import os
# aws rekognition client declaration
client = boto3.client('rekognition')
# aws s3 bucket client declaration
s3Client = boto3.client('s3')
# bucket instance
s3 = boto3.resource('s3')

## Testing Area

import os, sys

# Open a file
path = "Faces/"
dirs = os.listdir( path )
known_people_name = []
known_people_encodings = []
# This would print all the files and directories
i = 0
for file in dirs:
    known_people_name.append(file)
    temp_image = face_recognition.load_image_file("Faces/"+file)
    temp_encoding = face_recognition.face_encodings(temp_image)[0]
    known_people_encodings.append(temp_encoding)
    i+=1
    print(i)

## End of testing area
class people():
    val = {
        'name':'Unknown',
        'mood':'Unknown',
        'reactions':{
            'happy': '0',
            'sad': '0',
            'angry': '0',
            'calm': '0',
            'disgusted': '0',
            'confused': '0',
            'surprised': '0'
        }
    }
class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return

def emotionRecognition(passed_small_frame):
    cv2.imwrite(filename="temp.jpg",img = passed_small_frame)
    temp_image = "temp.jpg"
    with open(temp_image, 'rb') as image:
        # response = client.recognize_celebrities(Image={'Bytes': image.read()})
        response = client.detect_faces(Image={'Bytes': image.read()},
            Attributes=[
                'ALL'
            ]
            )
        image.close()
    maxEmotion = 0
    if response['FaceDetails'] != []:
        if response['FaceDetails'][0]['Emotions']:
            for item in response['FaceDetails'][0]['Emotions']:
                if item['Confidence'] > maxEmotion:
                    maxEmotion = item['Confidence']
                    people.val['mood'] = item['Type']
                    people.val['reactions'][item['Type'].lower()] = int(item['Confidence'])
                else:
                    people.val['reactions'][item['Type'].lower()] = int(item['Confidence'])
        else :
            print("No Emotions found!")
    else:
        people.val = {
            'name': 'Unknown',
            'mood':'Unknown',
                'reactions':{
                'happy': '0',
                'sad': '0',
                'angry': '0',
                'calm': '0',
                'disgusted': '0',
                'confused': '0',
                'surprised': '0'
            }
        }
        print("No faces found!")

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)

# # Load a sample picture and learn how to recognize it.
# mahmoud_image = face_recognition.load_image_file("Faces/Mahmoud.jpg")
# mahmoud_face_encoding = face_recognition.face_encodings(mahmoud_image)[0]

# # Load a sample picture and learn how to recognize it.
# salah_image = face_recognition.load_image_file("Faces/Abo_Salah.jpg")
# salah_face_encoding = face_recognition.face_encodings(salah_image)[0]

# # Load a sample picture and learn how to recognize it.
# ibrahim_image = face_recognition.load_image_file("Faces/Ahmed_Ibrahim.jpg")
# ibrahim_face_encoding = face_recognition.face_encodings(ibrahim_image)[0]

# # Load a sample picture and learn how to recognize it.
# ayman_image = face_recognition.load_image_file("Faces/Ayman.jpg")
# ayman_face_encoding = face_recognition.face_encodings(ayman_image)[0]

# # Load a sample picture and learn how to recognize it.
# shibob_image = face_recognition.load_image_file("Faces/Shibob.jpg")
# shibob_face_encoding = face_recognition.face_encodings(shibob_image)[0]

# # Load a sample picture and learn how to recognize it.
# mashhour_image = face_recognition.load_image_file("Faces/Mashhour.jpg")
# mashhour_face_encoding = face_recognition.face_encodings(mashhour_image)[0]


# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     mahmoud_face_encoding,
#     salah_face_encoding,
#     ibrahim_face_encoding,
#     ayman_face_encoding,
#     shibob_face_encoding,
#     mashhour_face_encoding

# ]
# known_face_names = [
#     "Mahmoud",
#     "Salah",
#     "Ahmed",
#     "Ayman",
#     "Shibob",
#     "Mashhour"
# ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    emotionRec = ThreadWithReturnValue(target=emotionRecognition,args = (small_frame,))
    emotionRec.start()
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_people_encodings, face_encoding)
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_people_name[first_match_index]
                people.val['name'] = name
            # face_names.append(name)

    process_this_frame = not process_this_frame
    emotion = emotionRec.join()
    print(people.val)
    r = requests.post("http://dev.getsooty.com:5000/mobile/deeplens", data = people.val)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
