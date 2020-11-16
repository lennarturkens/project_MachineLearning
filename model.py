import torch.nn as nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad
import torch


class fNet(nn.Module):
    def __init__(self):
        super(fNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 5)   # 1st conv layer INPUT [1 x 28 x 28] OUTPUT [64 x 12 x 12]
        self.conv2 = nn.Conv2d(64, 64, 5)  # 2nd conv layer INPUT [64 x 12 x 12] OUTPUT [64 x 4 x 4]
        self.conv3 = nn.Conv2d(64, 128, 5) # 3rd conv layer INPUT [64 x 4 x 4] OUTPUT [128 x 1 x 1]
        self.pool = nn.MaxPool2d(2, 2, padding=1)
         
    def forward(self, x):
        output = self.pool(F.relu(self.conv1(x)))
        output = self.pool(F.relu(self.conv2(output)))
        output = F.relu(self.conv3(output))
        output = output.view(-1, 128 * 1 * 1)
        return output
    
    
class gNet(nn.Module):
    def __init__(self):
        super(gNet, self).__init__()
                
        self.conv1 = nn.Conv2d(1, 64, 5)      # 1st conv layer INPUT [1 x 28 x 28]  OUTPUT [64 x 12 x 12]
        self.conv2 = nn.Conv2d(64, 64, 5)     # 2nd conv layer INPUT [64 x 12 x 12] OUTPUT [64 x 4 x 4]
        self.conv3 = nn.Conv2d(64, 128, 5)    # 3rd conv layer INPUT [64 x 4 x 4]   OUTPUT [128 x 1 x 1]
        self.pool = nn.MaxPool2d(2, 2, padding=1)
        
         
    def forward(self, x):
        output = self.pool(F.relu(self.conv1(x)))
        output = self.pool(F.relu(self.conv2(output)))
        output = F.relu(self.conv3(output))
        output = output.view(-1, 128 * 1 *1)
        return output


    def calcproto(self, class_list):

        g_output = []
        
        for class_image in class_list:
            g_output_class = self.forward(class_image)
            g_output.append(g_output_class)
        
        num_classes = len(g_output) # number of classes (10 in Digits experiment)
        num_features = g_output_class.shape[1] # number of features (128 in Digits experiment)
                
        prototypes = torch.Tensor(size=torch.Size([num_classes, num_features]))
        
        class_count = 0
        for g_class in g_output:
            if len(g_class) > 1: # number of images in 1 class bigger than one (not to be confused with number of classes!)
                prototype = torch.mean(g_class, dim=1) / len(g_class) # calculating the prototype for each class while testing
            
            else:
                prototype = g_class # in case of training we only input one image
            
            prototypes[class_count] = prototype
            
            class_count += 1
        
        return prototypes
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.restored = True

        
        self.layer = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1),
            RevGrad()
        )
        
    def forward(self, x):
        output = self.layer(x)
        return output[:, 0]
    
    
class hSim(nn.Module):
    def __init__(self):
        super().__init__()
    

        self.restored = True
        
        self.U = torch.rand(size=(128, 512), requires_grad=True)
        self.V = torch.rand(size=(128, 512), requires_grad=True)
        
    def forward(self, x_batch, mu_c):
        
        h_batch = torch.zeros(size=[32, 10]) # batchsize = 32 maybe 'automate' this size so we don't get in trouble later when we may change the batch size (same goes for number of classes (10))

        for i in range(0, len(x_batch)):
            x = x_batch[i]

            fac1 = torch.matmul(x, self.U)
            fac2 = torch.matmul(mu_c, self.V)
        
            h = torch.matmul(fac2, fac1)
            
            h_batch[i] = F.log_softmax(h, dim=0)
        
        return h_batch   
    
    
class SimNet(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.restored = True
    
        self.fnet = fNet()
        self.gnet = gNet()
        self.disc = Discriminator()
        self.hsim = hSim()
        
        
    def forward(self, f_input, g_input):
        # f_input is batch of images
        # g_input should be a list (in the future maybe change to tensor?) of one (while training --> randomly chosen) or more (while testing --> every from test_batch) image(s) of every class
        
        f_output = self.fnet(f_input)
        prototypes = self.gnet.calcproto(g_input)
        
        class_output = self.hsim(f_output, prototypes)
        disc_output = self.disc(f_output)
        
        return [class_output, disc_output] 