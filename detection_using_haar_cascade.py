import argparse
import os

import cv2

import logger
from utils import create_outvideo_folder

logging = logger.myLogger(__name__)

person_cascade = cv2.CascadeClassifier(os.path.join('data/cascades/haarcascade_fullbody.xml'))
face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')


# eye_cascade = cv2.CascadeClassifier('data/cascades/frontalEyes35x16.xml')


def run_detection(given_video, write_video):
    cap = cv2.VideoCapture(given_video)
    out_video = None
    if write_video:
        fps = 20
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_video = cv2.VideoWriter()
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_video.open('data/outvideos/HC_output.avi', fourcc, fps, size, True)
    while True:
        r, frame = cap.read()
        if r:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            people = person_cascade.detectMultiScale(gray_frame)
            for (x, y, w, h) in people:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, str(len(faces)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("preview", frame)
            if write_video:
                out_video.write(frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break
    if write_video:
        out_video.release()
    logging.info(
        "Stopped HC object Detection; Check video in data/outvideos folder" if write_video else "Stopped Object Detection; Bye!")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Object Detection on Video: Live/Saved Using Haar-cascade',
                                     add_help=True)
    parser.add_argument('-v', '--video', default="live", type=str,
                        help='video on which you want to run object detection; default is live which opens a live webcam stream. Other option is to provide path to a video file.')
    parser.add_argument('-w', '--write', default="false", type=str,
                        help='set True if you want to save the output video')
    args = parser.parse_args()
    given_video = None
    if args.video != "live":
        given_video = args.video
        if os.path.isfile(given_video):
            logging.info("Opening the given video path: " + str(given_video))
        else:
            logging.info("No video found in given path, please check; running detection on webcam.")
            given_video = 0
    else:
        logging.info("Running detection on live webcam video; Press q to close window ")
        given_video = 0

    write_video = False
    if args.write.lower() == "true":
        write_video = True

    create_outvideo_folder()
    run_detection(given_video, write_video)
