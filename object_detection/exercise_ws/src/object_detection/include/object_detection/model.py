
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torch import nn
import numpy as np

class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        self.model = Model()

        self.checkpoint = torch.load(model_file)
        self.model.model.load_state_dict(self.checkpoint)
        self.model.model.eval()

        self.device = None
        if torch.cuda.is_available():
            print("GPU FOUND!")
            self.device = torch.device('cuda')
        else: 
            raise NoGPUAvailable()
        self.model.model.to(self.device)


    def predict(self, batch_or_image):
        img = torch.as_tensor(batch_or_image, dtype=torch.float32)
        img = img.permute(2, 0, 1).to(self.device)
        outputs = self.model.model([img])
        keep = torchvision.ops.nms(outputs[0]["boxes"],outputs[0]["scores"], 0.01)
        box = outputs[0]["boxes"][keep].cpu().detach().numpy().astype(np.int32)
        label = outputs[0]["labels"][keep].cpu().detach().numpy().astype(np.int32)
        score = outputs[0]["scores"][keep].cpu().detach().numpy().astype(np.int32)

        return [box], [label], [score]

class Model(nn.Module): 
    def __init__(self):
        super(Model, self).__init__()
        self.model = self.get_model(5)

    def get_model(self, num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model