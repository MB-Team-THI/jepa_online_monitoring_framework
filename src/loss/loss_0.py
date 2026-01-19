import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_0(nn.Module):
    def __init__(self, settings={}):
        super().__init__()
        self.settings = settings
        self.type = settings['type']

        use_cuda = torch.cuda.is_available()        
        self.device = torch.device("cuda" if use_cuda else "cpu")



    def L2_loss(self, y, y_pred):
        return nn.functional.mse_loss(y, y_pred)
    

    def L1_loss(self, y, y_pred):
        return nn.functional.l1_loss(y, y_pred)
    
    
    def forward(self, y, y_pred, cls_tokens=[]):
        assert y_pred.requires_grad, "y_pred must have a gradient"
        assert y.shape == y_pred.shape, "shapes of y and y_pred must be the same"

        loss = 0.0
        if self.type == "L2_loss":
            loss = self.L2_loss(y=y, y_pred=y_pred)

        elif self.type == "L1_loss":
            loss = self.L1_loss(y, y_pred)


        return loss