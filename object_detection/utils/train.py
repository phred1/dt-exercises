#!/usr/bin/env python3
import numpy as np
import os
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import utils
from torch.utils.data import Dataset
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import transforms as T
import cv2
MODEL_PATH="../exercise_ws/src/obj_det/include/model"
ENV="real"
TRAIN_DATA=f"../dataset/{ENV}_training"
CHECKPOINTS=f"../checkpoints/{ENV}"
EVAL=f"./eval/{ENV}"
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class DuckieDataset(Dataset):
    def __init__(self, path, tranforms):
        self.path = path
        self.tranforms = tranforms
        self.data = list(sorted(os.listdir(path)))

    def __getitem__(self, idx):
        item_path = os.path.join(self.path, self.data[idx])
        data = np.load(item_path)
        img = data[f"arr_{0}"]
        boxes = data[f"arr_{1}"]
        classes = data[f"arr_{2}"]
        img = torch.as_tensor(img, dtype=torch.float32)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        return img, target, image_id
    
    def get_transform(self, train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def __len__(self):
        return len(self.data)


def get_model(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def evaluation(model, val_loader, device, epoch):
    images, targets, image_ids = next(iter(val_loader))
    target = {}
    if len(targets["boxes"].shape) > 1:
        target["boxes"] = torch.squeeze(targets["boxes"], 0)
    target["boxes"] = target["boxes"].cpu().numpy().astype(np.int32)
    image = torch.squeeze(images,0)
    image_eval = image.permute(2, 0, 1).to(device)
    sample = image.cpu().numpy()
    model.eval()
    cpu_device = torch.device("cpu")

    outputs = model([image_eval])
    print(outputs[0]["boxes"].shape)
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    keep = torchvision.ops.nms(outputs[0]["boxes"],outputs[0]["scores"], 0.01)
    print(len(keep))
    predictions = outputs[0]["boxes"][keep].cpu().detach().numpy().astype(np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    for pred in predictions:
        cv2.rectangle(sample,
            (int(pred[0]), int(pred[1])),
            (int(pred[2]), int(pred[3])),
            (0, 0, 220), 1)

    for box in target["boxes"]:
        cv2.rectangle(sample,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (220, 0, 0), 1)
        print(box)
    ax.set_axis_off()
    cv2.imwrite(f"{EVAL}/epoch_{epoch}.png",cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
    # sample = sample.astype(np.float32)
    # sample /= 255.
    # cv2.imshow("eval",cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

def main():

    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of f"{MODEL_PATH}/weights`!\

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = DuckieDataset(TRAIN_DATA, None)

    batch_size = 1
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # our dataset has two classes only - background and person
    num_classes = 5
    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00001,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    loss_hist = Averager()
    itr = 1
    for epoch in range(num_epochs):
        print(f"Epoch_{epoch}")
        # update the learning rate
        lr_scheduler.step()
        loss_hist.reset()
        model.train()
    
        for images, targets, image_ids in train_loader:
            target = {}
            if len(targets["boxes"].shape) > 1:
                target["boxes"] = torch.squeeze(targets["boxes"], 0)
            target["boxes"] = target["boxes"].to(device)
            if len(targets["labels"].shape) > 1:
                target["labels"] = torch.squeeze(targets["labels"], 0)
            target["labels"] = target["labels"].to(device)
            
            image = torch.squeeze(images)
            image = image.permute(2, 0, 1).to(device)

            loss_dict = model([image], [target])
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:

                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
    
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()


        print("That's it!")
        evaluation(model, validation_loader, device, epoch)
        torch.save(model.state_dict(), f"{CHECKPOINTS}/epoch_{epoch}")
    pass

if __name__ == "__main__":

    main()