import os
import torch
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import get_detection_model, MOT20Dataset

torch.cuda.empty_cache()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset = MOT20Dataset(os.path.join("MOT20", "train"))
data_loader = DataLoader(
    dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x
)

model = get_detection_model(2)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(
    params, lr=0.0001, momentum=0.8, weight_decay=0.0005
) 

lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        imgs = [torch.stack([img.to(device) for img in dic["images"]]) for dic in batch]
        targets = [
            [{k: v.to(device) for k, v in t.items()} for t in dic["targets"]]
            for dic in batch
        ]

        for img, target in zip(imgs, targets):
            loss_dict = model(img, target)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

    lr_scheduler.step()

    print(f"Epoch #{epoch + 1} loss: {epoch_loss}")

torch.save(model.state_dict(), "model.pth")
