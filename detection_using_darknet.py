import argparse
import warnings

import cv2
from PIL import Image

import logger
from models import *
from utils import *

warnings.filterwarnings("ignore", category=UserWarning)

logging = logger.myLogger(__name__)


def run_detection(given_video, model_loader, color, only_persons, save_frames, write_video):
    total_frame_count = 0
    img_size = model_loader.get_imgsize()
    cap = cv2.VideoCapture(given_video)
    if write_video:
        fps = 20
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_video = cv2.VideoWriter()
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_video.open('data/outvideos/YOLOv3_output.avi', fourcc, fps, size, True)
    while (True):
        total_frame_count += 1
        ret, img = cap.read()
        piling = Image.fromarray(img)
        detections = model_loader.detect_from_image(piling)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        person_count = 0
        if detections is not None:
            tracked_objects = detections.cpu()
            for x1, y1, x2, y2, obj_id, prob, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                cls = model_loader.get_classes()[int(cls_pred)]
                if int(cls_pred) == 0:
                    person_count += 1
                if only_persons and int(cls_pred) != 0:
                    continue
                cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                cv2.rectangle(img, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
                cv2.putText(img, cls + "-" + str(prob.item()), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
        cv2.putText(img, "PersonCount-" + str(person_count), (int(pad_x), int(pad_y)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255), thickness=2)

        cv2.imshow('Running Detection', img)
        if write_video:
            out_video.write(img)
        if save_frames:
            name = os.path.join("data", "frames", "frame%d.jpg" % total_frame_count)
            cv2.imwrite(name, img)
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break
    if write_video:
        out_video.release()
        logging.info("Check output video in data/outvideos folder")

    logging.info(
        "Stopped Object Detection; Check frames in data/frames folder" if save_frames else "Stopped Object Detection; Bye!")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Object Detection on Video: Live/Saved', add_help=True)
    parser.add_argument('-v', '--video', default="live", type=str,
                        help='video on which you want to run object detection; default is live which opens a live webcam stream. Other option is to provide path to a video file.')
    parser.add_argument('-c', '--config', default="config", type=str,
                        help='Path to the folder with yolov3.cfg, yolov3.weights and coco.names files')
    parser.add_argument('-o', '--only', default="false", type=str,
                        help='set True if you want only Person detection; default false')
    parser.add_argument('-s', '--save', default="false", type=str,
                        help='set True if you want to save frames of the video')
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

    config_path = os.path.join(args.config, 'yolov3.cfg')
    weights_path = os.path.join(args.config, 'yolov3.weights')
    class_path = os.path.join(args.config, 'coco.names')

    only_persons = args.only
    if only_persons.lower() == "false":
        only_persons = False
    else:
        only_persons = True

    save_frames = args.save
    if save_frames.lower() == "false":
        save_frames = False
    else:
        save_frames = True

    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4
    bounding_boxes_color = (0, 255, 0)
    logging.info("Loading pretrained YOLO-V3 Darknet model")
    model_loader = ModelLoader(img_size, weights_path, class_path, config_path, conf_thres, nms_thres)

    write_video = False
    if args.write.lower() == "true":
        write_video = True
    create_outvideo_folder()
    run_detection(given_video, model_loader, bounding_boxes_color, only_persons, save_frames, write_video)
