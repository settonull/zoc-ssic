"""Classifier head for MCB models."""

from torch import nn


class BasicClassifier(nn.Module):

    """Class classifier head for MCB style models."""

    def __init__(self, n_classes=1000, dropout=0.2):
        """Initialize SkipGramDistNet."""
        super(BasicClassifier, self).__init__()
        self.n_classes = n_classes

        self.drop = nn.Dropout(p=dropout)
        self.cls = nn.Linear(self.mcb_model.output_dim, self.n_classes)

    def forward(self, img):
        """Forward Pass."""

        #do the work on the image
        out = img

        pooled_output = self.drop(out)
        # logits: [batch_size, n_classes]
        logits = self.cls(pooled_output)

        return logits
