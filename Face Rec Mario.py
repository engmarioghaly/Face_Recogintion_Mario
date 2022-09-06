import face_recognition
import cv2
import numpy as np


# Load a sample picture and learn how to recognize it.
hager =  face_recognition.load_image_file("hager.png")
hager_face_encoding = face_recognition.face_encodings(hager)[0]


Mario =  face_recognition.load_image_file("Mario.JPG")
Mario_face_encoding = face_recognition.face_encodings(Mario)[0]

Ronza =  face_recognition.load_image_file("ronza.JPG")
Ronza_face_encoding = face_recognition.face_encodings(Ronza)[0]


Dalia =  face_recognition.load_image_file("dalia.JPG")
Dalia_face_encoding = face_recognition.face_encodings(Dalia)[0]

Nour =  face_recognition.load_image_file("nour.JPG")
Nour_face_encoding = face_recognition.face_encodings(Nour)[0]

Eman =  face_recognition.load_image_file("eman.JPG")
Eman_face_encoding = face_recognition.face_encodings(Eman)[0]



video = cv2.VideoCapture(0)

known_face_encodings = [
    hager_face_encoding,
    Mario_face_encoding,
    Ronza_face_encoding,
    Dalia_face_encoding,
    Nour_face_encoding ,
    Eman_face_encoding
    
]

known_face_names = [
    "Hager",
    "Mario",
    "Ronza",
    "Dalia",
    "Nour",
    "Eman"
]

while True:

    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # face-reconition uses RGB

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Not in this class"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()

     
