# id3.py
# -------
# ID3 Decision Tree implementation

import util
import math
# A single node in the ID3 decision tree.
# every node is either an internal node or a leaf
class Node:
    """
    A single node in the ID3 decision tree.
    An internal node splits on one feature and has two children (one for feature = 0, one for feature = 1).
    A leaf node has no feature and no children, and its label is the prediction.
    """
    def __init__(self, label, feature = None, children = None):
        self.feature = feature # storing the feature (pixel) this node splits on, None means the node is a leaf
        self.children = children # storing the child nodes as a dict {0: child node, 1: child node}
        self.label = label  # storing the prediction for a leaf, or the majority label for an internal node (used when pruning)
        if self.children is None:
            self.children = {}  # creating a new empty dict for each node instead of using {} as the default argument, since a mutable default argument would be shared by every node

    def isLeaf(self):
        return self.feature is None # a node is a leaf when it has no feature to split on

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
        self.trainingData = trainingData # storing the training data so every helper method can use it
        self.trainingLabels = trainingLabels # storing the training labels
        self.validationData = validationData # storing the validation data for pruning
        self.validationLabels = validationLabels # storing the validation labels for pruning
        featuresList = list(trainingData[0].keys()) # getting all features (pixels) from the first sample, since every sample has the same features
        allIndices = list(range(len(trainingData))) # getting the indices of all training samples [0, 1, ..., n-1]
        self.tree = self.build(allIndices, featuresList, None) # building the full tree, the root always has samples so its parent label is never used
        # print("nodes before pruning:", self.countNodes(self.tree)) # temp
        if self.pruning:
            validationIndices = list(range(len(self.validationData))) # getting the indices of all validation samples
            self.prune(self.tree, validationIndices) # pruning starts from the root with every validation sample
        #     print("nodes after pruning:", self.countNodes(self.tree)) # temp

    def classify(self, testData):
        """
        Return the predicted labels for the test data.
        """
        return [self.predict(datum) for datum in testData]  # predict each datum independently; the list comprehension keeps the output in the same order as testData, which the evaluator relies on



    # You may add additional helper methods here.

    #   Return the entropy (in bits) of a set of examples:
    #   H(S) = - sum_c p_c * log2(p_c)
    def entropy(self, labelCounts, total):
        """
        Returns the entropy of a set of samples: H(S) = - sum_c p_c * log2(p_c)
        labelCounts is a Counter of how many samples have each label, and total is the number of samples.
        The entropy is 0 when every sample has the same label, and highest when all labels are equally common.
        """    
        if total == 0:
            return 0 # an empty set has no uncertainty, returning 0 also avoids dividing by 0
        else:
            result = 0.0 # storing the entropy
            for label, count in labelCounts.items():
                # skipping labels with count 0, since 0 * log(0) is taken as 0 and math.log(0) gives an error
                if count == 0:
                    continue
                p = count / total # calculating the proportion of samples with this label
                result = result - p * math.log(p, 2) # subtracting p * log2(p) because of the minus sign in the formula
            return result

    # Return the most common label in the list labels.
    def majority(self, labels):
        """
        Returns the most common label in the list labels (labels must not be empty).
        Used as the prediction of a leaf, and stored on internal nodes for pruning.
        """
        count = util.Counter() # counting how many times each label appears
        for label in labels:
            count[label] += 1 # incrementing the count of the label, Counter starts missing keys at 0
        return count.argMax() # returning the label with the highest count

    def bestFeature(self, indices, features):
        """
        Returns (feature, gain) for the feature with the highest information gain on the samples in indices:
        Gain(S, A) = H(S) - sum_{v in {0,1}} |S_v| / |S| * H(S_v)
        indices are the training samples at the current node (must not be empty),
        and features are the candidate features not used yet on this path.
        """
       # step1: counting the labels of all samples at the current node to get H(S)
        labelCounts = util.Counter() # counting how many samples have each label at this node
        for i in indices:
            labelCounts[self.trainingLabels[i]] += 1
        n = len(indices) # getting the total number of samples at this node
        node_entropy = self.entropy(labelCounts, n) # calculating the entropy of the node before splitting

        # step2: scanning the samples only once to count the labels where each feature is on (= 1)
        onCounts = {} # storing a Counter of labels for every feature, onCounts[feature][label]
        for f in features:
            onCounts[f] = util.Counter()  # creating an empty Counter for each feature, since a plain dict has no default value

        for i in indices: # going through every sample at this node
            for f in features: # going through every candidate feature
                if self.trainingData[i][f] == 1: # if the feature is on then increase the count for the label of sample i
                    onCounts[f][self.trainingLabels[i]] += 1

        # step3: calculating the information gain of every feature and keeping the best one
        bestGain = -1 # gain is always >= 0, so starting at -1 makes sure some feature is chosen
        bestFeature = None
        for f in features:
            onLabelCounts = onCounts[f]  # getting the label counts of the samples where the feature is on
            n1 = onLabelCounts.totalCount() # getting the number of samples where the feature is on
            offLabelCounts = labelCounts - onCounts[f] # getting the label counts where the feature is off by subtraction, no need to scan again
            n0 = n - n1 # getting the number of samples where the feature is off
            onEntropy = self.entropy(onLabelCounts, n1) # calculating the entropy of the "on" subset
            offEntropy = self.entropy(offLabelCounts, n0) # calculating the entropy of the "off" subset
            weightedEntropy = (n1/n) *  onEntropy + (n0/n) * offEntropy # calculating the expected entropy after splitting, weighted by subset size
            gain = node_entropy - weightedEntropy # calculating how much the split reduces the entropy
            if gain > bestGain:  # checking if the current feature gives a better gain, strict ">" keeps the first feature on ties
                bestGain = gain
                bestFeature = f
        return bestFeature, bestGain

    def build(self, indices, features, parentLabel):
        """
        Recursively builds the ID3 (sub)tree for the samples in indices and returns its root node.
        features are the features not used yet on this path,
        and parentLabel is the majority label of the parent node (used when no samples reach this node).
        """
         # base case 1: no samples reach this branch, so use the majority label of the parent
        if len(indices) == 0:
            return Node(parentLabel)
        labels = [self.trainingLabels[i] for i in indices] # getting the labels of all samples at this node
        majorityLabel = self.majority(labels) # getting the most common label at this node
 
        # base case 2: all samples have the same label, so the node is pure
        if len(set(labels)) == 1:
            return Node(majorityLabel)
 
        # base case 3: no features left to split on, so predict the majority label
        if len(features) == 0:
            return Node(majorityLabel)
 
        bestFeature, bestGain = self.bestFeature(indices, features) # choosing the feature with the highest information gain
 
        offIndices = [i for i in indices if self.trainingData[i][bestFeature] == 0] # getting the samples where the best feature is off
        onIndices = [i for i in indices if self.trainingData[i][bestFeature] == 1] # getting the samples where the best feature is on
 
        # creating a NEW list without the best feature, removing it from "features" directly
        # would also change the list used by the other child and by the parent
        remainingFeatures = [f for f in features if f != bestFeature]
 
        node = Node(majorityLabel, bestFeature) # creating an internal node that splits on the best feature
        node.children[0] = self.build(offIndices, remainingFeatures, majorityLabel) # building the subtree for feature = 0
        node.children[1] = self.build(onIndices,  remainingFeatures, majorityLabel) # building the subtree for feature = 1
        return node # returning the root of this subtree
  
    def predict(self, datum):
        """
        Returns the predicted label for one sample by walking the tree from the root down to a leaf.
        """
        current = self.tree  # starting from the root of the tree
        while not current.isLeaf():
            current = current.children[datum[current.feature]] # going to the child that matches the sample's value (0 or 1) of the node's feature
        return current.label # returning the label of the leaf as the prediction

    def prune(self, node, indices):
        """
        Reduced-error pruning: prunes the subtree rooted at node using the validation samples in indices
        (the validation samples that reach this node), and returns the number of validation errors of the
        subtree after pruning. Children are pruned first (post-order), then the node is turned into a leaf
        if that does not increase the validation errors.
        """
        leafErrors = len([i for i in indices if self.validationLabels[i] != node.label]) # counting the errors if this node were a leaf
        if node.isLeaf(): # a leaf cannot be pruned further, so return its errors
            return leafErrors

        offIndices = [i for i in indices if self.validationData[i][node.feature] == 0] # getting the validation samples that go to child 0
        onIndices = [i for i in indices if self.validationData[i][node.feature] == 1] # getting the validation samples that go to child 1

        subtreeErrors = self.prune(node.children[1], onIndices) + self.prune(node.children[0], offIndices) # pruning both children first and adding up their errors
        
        if leafErrors <= subtreeErrors: # checking if turning this node into a leaf does not make the validation errors worse
            node.feature = None  # removing the feature so the node becomes a leaf
            node.children = {} # removing the children of the node
            return leafErrors
        else:
            return subtreeErrors  # keeping the subtree and returning its errors

    def countNodes(self, node):
        """
        Returns the number of nodes in the subtree rooted at node (used to compare tree size with and without pruning).
        """
        if node.isLeaf():  # a leaf counts as one node
            return 1
        else:
            return 1 + self.countNodes(node.children[0]) + self.countNodes(node.children[1])  # counting this node plus the nodes of both subtrees