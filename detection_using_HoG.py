import argparse
import os

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

import logger
from utils import create_outvideo_folder

logging = logger.myLogger(__name__)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def run_detection(given_video, write_video):
    cap = cv2.VideoCapture(given_video)
    if write_video:
        fps = 20
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_video = cv2.VideoWriter()
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_video.open('data/outvideos/HoG_output.avi', fourcc, fps, size, True)
    while True:
        r, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cv2.imshow("HoG", frame)
        if write_video:
            out_video.write(frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break
    if write_video:
        out_video.release()
    logging.info(
        "Stopped HoG object Detection; Check video in data/outvideos folder" if write_video else "Stopped Object Detection; Bye!")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Object Detection on Video: Live/Saved Using HoG',
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
