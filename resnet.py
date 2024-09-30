import torch.nn as nn
from resblock import ResBlock

class ResNet(nn.Module):
    """
    Diese Klasse wird verwendt, um ein neuronales Netz für den Alpha-MCTS zu erstellen. Die Entwicklung richtet sich
    am Orginal Paper für Alpha-Zero (ein Screenshot der Architektur befinded unter ./image_net_architecture.PNG).
    """
    def __init__(self, game, num_resBlocks, num_hidden, inputarrays, device):
        super().__init__()
        self.device = device
        """
        Als input für das neuronale Netz werden die States (Abbilder/Matritzen des Spielfeldes) verwendet. Diese States
        ähneln Bildern, welche in RGB-Planes unterteilt werden. In unserem Fall wird das Spielfeld ebenfalls in 3 Planes 
        unterteilt. Die Plane für Spieler 1, die Plane für unbesetzte Felder (0) und für die Plane für den 
        gegnerischen Spieler (-1). Aus diesem Grund wird beim Startblock die hidden-size auf 3 gesetzt. Die Output-Dim
        wird auf den parameter "num_hidden" gesetzt, welcher beim Anlegen des Netzes festgelegt wird.
        Die Kernel_size ist der Parameter, welcher die größe des Filters angibt, welcher über das Eingangssignal gelegt 
        wird. Um die states nicht zu verfälschen wird diese ebenfalls auf 3 gesetzt und padding auf same (1) gestellt.
        """
        self.startBlock = nn.Sequential(
            nn.Conv2d(inputarrays, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        """
        Anschließend folgt wie im Orginal-Paper dargestellt das Backbone, welches aus einer Reihe von resBlocks besteht.
        Wieviele ResBlöcke implementiert werden hängt davon ab, welcher Wert beim Anlegen des Netzes für
        "num_resBlocks" eingetragen wurde.
        """
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        """
        Das Neuronale Netz hat 2 Outputs. Zunächst die policy und zusätzlich der value. Für beide Outputs wird ein
        seperater "Head" entworfen. Der Policy-Head hat die Outputgröße von der Anzahl an möglichen Aktionen, da dieser
        später die action-propability-matrix zurückgeben soll. Der Value-Head soll nur den Value vom state zurückgeben,
        darum ist dessen output-size = 1.
        """
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * game.row_count * game.column_count, game.action_size)
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        self.to(device)
    def forward(self, x):
        """
        In der Forword-Funktion, werden die einzelnen erstellten Blöcke nun miteinander vernetzt
        X ist das Eingangssignal, welches unseren State(das Spielfeld in einer Matritze dargestellt) wiederspiegelt.
        Zunächst werden die 3 Planes (Spieler1,Spieler2 und nichtbesetzte Felder) dem Startblock übergeben.
        Das outputsignal vom Startblock wird nun durch alle resBlocks im Backbone geschleift. Dessen Output wird
        jeweils für den Policy-Head und Value-Head als Eingangssignal verwendet. Beide Heads haben als Output dann
        die genünschten Größen value und die policy des anfänglichen States.
        """
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value