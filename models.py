import torch.nn as nn
import numpy as np

# CIFAR model (architecture from CS 242)
def conv_block(in_channels, out_channels, kernel_size=3, stride=1,
               padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
        )

class ConvNet(nn.Module):
    def __init__(self, activation_grad=False):
        super(ConvNet, self).__init__()
        self.collect_activation_gradients = activation_grad
        
        self.layer0=    conv_block(3, 32)
        self.layer1=    conv_block(32, 32)
        self.layer2=    conv_block(32, 64, stride=2)
        self.layer3=    conv_block(64, 64)
        self.layer4=    conv_block(64, 64)
        self.layer5=    conv_block(64, 128, stride=2)
        self.layer6=    conv_block(128, 128)
        self.layer7=    conv_block(128, 256)
        self.layer8=    conv_block(256, 256)
        
        self.pooler=    nn.AdaptiveAvgPool2d(1)
        
        self.relu  =    nn.ReLU(inplace=False)
        
        self.classifier = nn.Linear(256, 10)
        
        self.activation_outputs = []
        
#         self.model = nn.Sequential(
#             conv_block(3, 32),
#             conv_block(32, 32),
#             conv_block(32, 64, stride=2),
#             conv_block(64, 64),
#             conv_block(64, 64),
#             conv_block(64, 128, stride=2),
#             conv_block(128, 128),
#             conv_block(128, 256),
#             conv_block(256, 256),
#             nn.AdaptiveAvgPool2d(1)
#             )

    def forward(self, x):
        o0 = self.relu(self.layer0(x))
        o1 = self.relu(self.layer1(o0))
        o2 = self.relu(self.layer2(o1))
        o3 = self.relu(self.layer3(o2))
        o4 = self.relu(self.layer4(o3))
        o5 = self.relu(self.layer5(o4))
        o6 = self.relu(self.layer6(o5))
        o7 = self.relu(self.layer7(o6))
        o8 = self.relu(self.layer8(o7))
        
#         o0.retain_grad()
#         o1.retain_grad()
#         o2.retain_grad()
#         o3.retain_grad()
#         o4.retain_grad()
#         o5.retain_grad()
#         o6.retain_grad()
#         o7.retain_grad()
#         o8.retain_grad()
        
#         if self.collect_activation_gradients:
#             self.activation_outputs = [o0, o1, o2,
#                                        o3, o4, o5, 
#                                        o6, o7, o8]
            
#             for o in self.activation_outputs:
#                 o.retain_grad()
        
        h  = self.pooler(o8)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        model_output = self.classifier(h)
        return model_output, (o0, o1, o2, o3, o4, o5, o6, o7, o8)