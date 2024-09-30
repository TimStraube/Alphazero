import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Ein Residualblock besteht aus mehreren Schichten von Neuronen (Conv2d), die in einer sequentiellen Reihenfolge
    angeordnet sind.
    Der Eingabevektor wird direkt an den Ausgabevektor weitergegeben, indem ein sogenannter Residualpfad
    eingeführt wird. (residual = x)
    Dieser Pfad ermöglicht es dem Modell, Informationen aus früheren Schichten direkt an spätere
    Schichten weiterzugeben. Die Architektur des ResBlocks ist unter ./resBlock.PNG zu sehen (Orginal-Paper).
    """
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x