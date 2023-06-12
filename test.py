import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import draw_detections, get_detection_model, MOT20Dataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset = MOT20Dataset(os.path.join("MOT20", "train"))
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

model = get_detection_model(2)
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()

sample_sequence_index = 0
sample_sequence = dataset[sample_sequence_index]
sample_images = sample_sequence["images"]
sample_targets = sample_sequence["targets"]

for image, target in zip(sample_images, sample_targets):
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image)

    boxes = predictions[0]["boxes"].detach().cpu().numpy()
    scores = predictions[0]["scores"].detach().cpu().numpy()
    labels = predictions[0]["labels"].detach().cpu().numpy()

    keep = scores >= 0.7
    print(scores)
    boxes = boxes[keep]
    labels = labels[keep]

    if len(boxes) == 0:
        print("No objects detected in the image")
        continue

    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    draw_detections(image, boxes)
