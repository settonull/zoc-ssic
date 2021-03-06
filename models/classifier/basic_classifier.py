from torch import nn
from torchvision import models
import torchvision.transforms as transforms


class BasicClassifier(nn.Module):

    #If our particular model needs to do anything special to transform the image we can specify it here
    #will need to copy this over to eval.py if we do anything special
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    def __init__(self, n_classes=1000, dropout=0.2):
        """Initialize SkipGramDistNet."""
        super(BasicClassifier, self).__init__()
        self.n_classes = n_classes

        self.main = models.resnet18(pretrained=False)

        self.main.fc = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(128, self.n_classes),
                                 nn.LogSoftmax(dim=1))

    def forward(self, img):
        """Forward Pass."""

        #do the work on the image
        x = self.main(img)

        return x
