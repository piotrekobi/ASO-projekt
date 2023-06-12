import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms import functional as F


def load_sequence(sequence_folder):
    img_folder = os.path.join(sequence_folder, "img1")
    img_files = sorted(
        [
            os.path.join(img_folder, f)
            for f in os.listdir(img_folder)
            if f.endswith(".jpg") and not f.startswith("._")
        ]
    )

    images = []
    for img_file in img_files:
        img = cv2.imread(img_file)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Unable to load image at {img_file}")

    det_file = os.path.join(sequence_folder, "det/det.txt")
    detections = []
    if os.path.exists(det_file):
        data = np.loadtxt(det_file, delimiter=",")
        frame_indices = data[:, 0].astype(int)
        max_frame_index = frame_indices[-1]
        for i in range(1, int(max_frame_index) + 1):
            mask = frame_indices == i
            frame_data = data[mask]
            detections.append(frame_data)
    else:
        detections = [None] * len(images)

    return images, detections


def draw_detections(image, detections):
    """
    Draw bounding boxes on an image.
    :param image: The image to draw on.
    :param detections: Detections for the image. Each row of detections is expected
    to be in the format (frame_index, id, bb_left, bb_top, bb_width, bb_height, _, _, _, _), where
    bb_left, bb_top, bb_width, and bb_height are the coordinates and size of the bounding box.
    """
    image_copy = image.copy()

    for detection in detections:
        bb_left = int(detection[0])
        bb_top = int(detection[1])
        bb_width = int(detection[2] - detection[0])
        bb_height = int(detection[3] - detection[1])
        print(bb_left, bb_top, bb_width, bb_height)
        cv2.rectangle(
            image_copy,
            (bb_left, bb_top),
            (bb_left + bb_width, bb_top + bb_height),
            (0, 255, 0),
            1,
        )

    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(image_copy)
    plt.show()


def get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



class MOT20Dataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.sequence_folders = sorted(
            [
                f.path
                for f in os.scandir(self.root_folder)
                if f.is_dir() and not f.name.startswith(".")
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        sequence_folder = self.sequence_folders[index]
        images, detections = load_sequence(sequence_folder)

        max_images_per_sequence = 10 
        images = images[:max_images_per_sequence]
        detections = detections[:max_images_per_sequence]

        images = [self.transform(image) for image in images]

        targets = []
        for detection in detections:
            boxes = detection[:, 2:6]
            boxes[:, 2:] += boxes[:, :2]

            target = {}
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.ones((boxes.shape[0],), dtype=torch.int64)
            targets.append(target)

        return {"images": images, "targets": targets}

    def __len__(self):
        return len(self.sequence_folders)