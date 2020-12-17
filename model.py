"""model.py:
    
    contains StandardNet, SimNet and their components together with a PretrainNet that is used to pretrain the feature extractors of SimNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 5)   # 1st conv layer INPUT [1 x 28 x 28] OUTPUT [64 x 12 x 12]
        self.conv2 = nn.Conv2d(64, 64, 5)  # 2nd conv layer INPUT [64 x 12 x 12] OUTPUT [64 x 4 x 4]
        self.conv3 = nn.Conv2d(64, 128, 5) # 3rd conv layer INPUT [64 x 4 x 4] OUTPUT [128 x 1 x 1]
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05) # batch normalisation
        self.pool = nn.MaxPool2d(2, 2, padding=1)
         
    def forward(self, x):
        output = self.pool(F.relu(self.bn1(self.conv1(x))))
        output = self.pool(F.relu(self.bn1(self.conv2(output))))
        output = F.relu(self.conv3(output))
        output = output.view(-1, 128 * 1 * 1)
        return output
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 10)
         
    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.log_softmax(self.fc2(output), dim=1)
        return output
    
class SimilarityClassifier(nn.Module):
    def __init__(self):
        super(SimilarityClassifier, self).__init__()
        
        self.U = nn.Parameter(torch.rand(size=(512, 128), requires_grad=True))
        self.V = nn.Parameter(torch.rand(size=(512, 128), requires_grad=True))
        
    def forward(self, encoded_features, prototypes):
                
        factor_1 = torch.matmul(self.U, encoded_features.t()) # size = (512, 32)
        factor_2 = torch.matmul(self.V, prototypes.t()) # size = (512, 10)
        
        h = torch.matmul(factor_1.t(), factor_2) # size = (32, 10)
        h = F.log_softmax(h, dim=1)
        return h
        
                         
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
            RevGrad(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        
    def forward(self, x):
        output = self.layer(x)
        output = output/torch.max(output)
        output = F.softmax(output, dim=1) 
        
        return output[:, 0]
    
class PretrainNet(nn.Module):
    def __init__(self):
        super(PretrainNet, self).__init__()
        
        self.encoder = Encoder()
        self.classifier = Classifier()
        
         
    def forward(self, x):

        encode_out = self.encoder(x)
        class_out = self.classifier(encode_out)
        
        return class_out
    
class SimNet(nn.Module):
    def __init__(self):
        super(SimNet, self).__init__()
        
        self.fnet = Encoder()
        self.gnet = Encoder()
        self.classifier = SimilarityClassifier()
        self.disc = Discriminator()
        
         
    def forward(self, x, mode="train", proto_images=None, encoded_proto=None):
        
        if mode == "train":
            encoded_out = self.fnet(x)
            encoded_proto = self.gnet(proto_images)
            class_out = self.classifier(encoded_out, encoded_proto)
            disc_out = self.disc(encoded_out)
            
        if mode == "test":
            encoded_out = self.fnet(x)
            class_out = self.classifier(encoded_out, encoded_proto)
            disc_out = self.disc(encoded_out)
            
        
        return [class_out, disc_out, encoded_proto]
    
class StandardNet(nn.Module):
    def __init__(self):
        super(StandardNet, self).__init__()
        
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.disc = Discriminator()
        
    def forward(self, x):

        encode_out = self.encoder(x)
        class_out = self.classifier(encode_out)
        disc_out = self.disc(encode_out)
        
        return [class_out, disc_out]