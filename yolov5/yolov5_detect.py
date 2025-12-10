# detect.py

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER, check_file, check_img_size, check_requirements, colorstr, cv2,
    increment_path, non_max_suppression, scale_boxes, xyxy2xywh
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights="yolov5s.pt",  # Model path or triton URL
    source=0,  # Source for inference (webcam, image, video, etc.)
    data="data/coco128.yaml",  # Dataset YAML path
    imgsz=(640, 640),  # Inference image size (height, width)
    conf_thres=0.25,  # Confidence threshold for detections
    iou_thres=0.45,  # Intersection over union threshold for NMS
    device="",  # Device for inference, e.g., '0' for GPU or 'cpu'
    view_img=True,  # Show results
    save_img=True,  # Save inference images
    save_txt=False,  # Save results to *.txt
    save_csv=False,  # Save results in CSV format
    project="runs/detect",  # Directory to save results
    name="exp",  # Subdirectory for results
    line_thickness=3,  # Thickness of bounding box lines
):
    # Process source input (image, video, webcam, etc.)
    source = str(source)
    save_dir = increment_path(Path(project) / name, exist_ok=False)
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    # Load dataset
    dataset = LoadStreams(source, img_size=imgsz, stride=stride) if isinstance(source, str) and source.isdigit() else LoadImages(source, img_size=imgsz, stride=stride)

    # Inference and post-processing
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255

        # Run inference
        pred = model(im)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

        # Process detections
        for det in pred:  # Per image
            annotator = Annotator(im0s, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls)))

            # Show and save results
            im0s = annotator.result()
            if view_img:
                cv2.imshow(str(path), im0s)
                cv2.waitKey(1)

            if save_img:
                cv2.imwrite(str(save_dir / Path(path).name), im0s)

    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="Path to model weights")
    parser.add_argument("--source", type=str, default="data/images", help="Input source (image/video/webcam/etc.)")
    parser.add_argument("--device", default="", help="Device for inference (e.g., 'cpu', '0', etc.)")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="Inference size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--view-img", action="store_true", help="Display results")
    parser.add_argument("--save-txt", action="store_true", help="Save results to text files")
    parser.add_argument("--save-csv", action="store_true", help="Save results in CSV format")
    parser.add_argument("--save-img", action="store_true", help="Save inference images")
    parser.add_argument("--project", default="runs/detect", help="Project directory for saving results")
    parser.add_argument("--name", default="exp", help="Subdirectory for results")
    parser.add_argument("--line-thickness", type=int, default=3, help="Bounding box line thickness")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
