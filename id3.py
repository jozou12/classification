# id3.py
# -------
# ID3 Decision Tree implementation

import util

class ID3Classifier:
    """
    ID3 Decision Tree classifier.
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "id3"
        self.tree = None
        self.pruning = False

        # You may initialize additional parameters here.

    def setPruning(self, pruning):
        self.pruning = pruning

    def train(self, trainingData, trainingLabels,
              validationData, validationLabels):
        """
        Train the ID3 decision tree using information gain.
        Apply pruning if enabled.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classify(self, testData):
        """
        Return the predicted labels for the test data.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # You may add additional helper methods here.