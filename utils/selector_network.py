import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SelectorNetwork(nn.Module):
    def __init__(self):
        super(SelectorNetwork, self).__init__()
        

        self.fc1 = nn.Linear(768, 256,bias=True)  
        self.fc2 = nn.Linear(256, 32,bias=True)    

        # concat
        self.fc3 = nn.Linear(34, 8,bias=True)      
        self.fc4 = nn.Linear(8, 1)  


        init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        init.kaiming_normal_(self.fc4.weight, nonlinearity='sigmoid')



    def forward(self, x):
        # sep input and later concat 
        x1 = x[:, 2:]   # sentence embedding
        x2 = x[:, :2]   # entropy,ratio
        

        x1 = F.leaky_relu(self.fc1(x1))
        x1 = F.leaky_relu(self.fc2(x1))

        x_combined = torch.cat((x1, x2), dim=1)
        
        x_combined = F.leaky_relu(self.fc3(x_combined))
        output = F.sigmoid(self.fc4(x_combined))

  
        return output
