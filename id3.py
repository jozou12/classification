# id3.py
# -------
# ID3 Decision Tree implementation

import util
import math
# A single node in the ID3 decision tree.
# every node is either an internal node or a leaf
class Node:
    def __init__(self, label, feature = None, children = None):
        # feature defaults to None, so a node created with only a label is a leaf
        self.feature = feature
        self.children = children
        # label is required: leaves need a prediction and internal node
        self.label = label
        # avoid using {} as a default argument, since a mutable default would be
        # shared by every Node; create a fresh empty dict for each node instead
        if self.children is None:
            self.children = {}

    def isLeaf(self):
        return self.feature is None

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
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        # every datum has the same keys (one per pixel), so the first datum gives the full list of candidate features
        featuresList = list(trainingData[0].keys())
        # the root node starts with every training example: [0, 1, ..., n-1]
        allIndices = list(range(len(trainingData)))
        # grow the full tree; the root always has examples, so its parentLabel is never used and None is safe here
        self.tree = self.build(allIndices, featuresList, None)
        if self.pruning:
            pass

    def classify(self, testData):
        """
        Return the predicted labels for the test data.
        """
        "*** YOUR CODE HERE ***"
        # predict each datum independently; the list comprehension keeps the output in the same order as testData, which the evaluator relies on
        return [self.predict(datum) for datum in testData]



    # You may add additional helper methods here.

    #   Return the entropy (in bits) of a set of examples:
    #   H(S) = - sum_c p_c * log2(p_c)
    def entropy(self, labelCounts, total):
        # an empty set has no uncertainty; returning 0 also avoids dividing by 0
        if total == 0:
            return 0
        else:
            result = 0.0
            for label, count in labelCounts.items():
                # labels with count 0 contribute nothing (0 * log 0 is taken as 0), and skipping them avoids math.log(0), which raises an error
                if count == 0:
                    continue
                p = count/total
                result = result - p * math.log(p, 2)
            return result

    # Return the most common label in the list labels.
    def majority(self, labels):
        count = util.Counter()
        for label in labels:
            count[label] += 1
        return count.argMax()

    # Return (feature, gain) for the feature with the highest information
    # gain on the examples in indices:
    # Gain(S, A) = H(S) - sum_{v in {0,1}} |S_v| / |S| * H(S_v)
    def bestFeature(self, indices, features):
        # step1: count labels from parent node
        # labelCounts is the label distribution of the whole node, H(S)
        labelCounts = util.Counter()
        for i in indices:
            labelCounts[self.trainingLabels[i]] += 1
        n = len(indices)
        node_entropy = self.entropy(labelCounts, n)

        # step2: dict scan
        # onCounts[f] is a Counter of labels among the examples where feature f is 1 every feature needs its own empty Counter first, because the outer
        # dict is a plain dict and has no default value
        onCounts = {}
        for f in features:
            onCounts[f] = util.Counter()

        for i in indices:
            for f in features:
                if self.trainingData[i][f] == 1:
                    onCounts[f][self.trainingLabels[i]] += 1

        # step3: compute the information gain of every feature and keep the best gain is always >= 0, so starting at -1 guarantees some feature is chosen
        bestGain = -1
        bestFeature = None
        for f in features:
            onLabelCounts = onCounts[f]
            n1 = onLabelCounts.totalCount()
            offLabelCounts = labelCounts - onCounts[f]
            n0 = n - n1
            onEntropy = self.entropy(onLabelCounts, n1)
            offEntropy = self.entropy(offLabelCounts, n0)
            weightedEntropy = (n1/n) *  onEntropy + (n0/n) * offEntropy
            gain = node_entropy - weightedEntropy
            if gain > bestGain:
                bestGain = gain
                bestFeature = f
        return bestFeature, bestGain

    #   Recursively grow an ID3 (sub)tree and return its root Node.
    def build(self, indices, features, parentLabel):
        # base case 1: no examples reach this branch
        if len(indices) == 0:
            return Node(parentLabel)
        labels = [self.trainingLabels[i] for i in indices]
        majorityLabel = self.majority(labels)
        # base case 2: all examples share one label, the node is pure
        if len(set(labels)) == 1:
            return Node(majorityLabel)
        # base case 3: no features left to split on, predict the majority
        if len(features) == 0:
            return Node(majorityLabel)
        # choose the most informative feature
        bestFeature, bestGain = self.bestFeature(indices, features)
        # partition the examples by their value of the chosen feature
        offIndices = [i for i in indices if self.trainingData[i][bestFeature] == 0]
        onIndices = [i for i in indices if self.trainingData[i][bestFeature] == 1]
        # build a NEW list without the chosen feature; removing it from "features" in place would also change the list used by the sibling subtree and by the caller
        remainingFeatures = [f for f in features if f != bestFeature]
        # internal node: splits on bestFeature and remembers its majority label
        node = Node(majorityLabel, bestFeature)
        # each child gets this node's majority label in case its subset is empty
        node.children[0] = self.build(offIndices, remainingFeatures, majorityLabel)
        node.children[1] = self.build(onIndices,  remainingFeatures, majorityLabel)
        return node
    # Return the predicted label for a single datum by walking the tree from the root down to a leaf.
    def predict(self, datum):
        current = self.tree
        while not current.isLeaf():
            # # datum[...] is this image's value (0 or 1) at that pixel,
            current = current.children[datum[current.feature]]
        return current.label
