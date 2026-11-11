# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    labels_counter = util.Counter() # counting the number of training samples for each label
    features_counter = util.Counter() # counting how many times each feature is on for each label
    total_training_samples = len(trainingData) # getting the total number of training data

    # going through every training sample
    for index in range(total_training_samples):
      current_sample = trainingData[index] # getting the current sample/image
      current_label = trainingLabels[index] # getting the label for the sample/image

      labels_counter[current_label] += 1 # incrementing the total count of training samples with the label

      # going through every feature in the current sample/image
      for feature in self.features:
        if current_sample[feature] == 1: # if the feature is on then increase the count for the specific label
          features_counter[(feature, current_label)] += 1

    self.prior_probability = util.Counter() # creating a counter to store P(y) for each label
    
    # calculating the prior probability for each label
    for label in self.legalLabels:
      total_label_count = labels_counter[label] # getting the total label count
      self.prior_probability[label] = total_label_count / total_training_samples # calculating P(y) = total label count / total samples

    # initializing variables to keep track of the best smoothing factor
    best_accuracy = -1
    best_smoothing_factor = None
    best_conditional_probability = None

    # trying out every smoothing factor k starting from the smallest value
    for k in sorted(kgrid):
      conditional_probabilities = util.Counter() # storing the conditional probabilities for the current smoothing factor k

      # going through every label
      for label in self.legalLabels:
        current_label_count = labels_counter[label] # getting the number of training samples with the current label

        # going through all the features of the current label
        for feature in self.features:
          current_feature_count = features_counter[(feature, label)] # getting the total number of times this feature is on for the label

          numerator = current_feature_count + k # adding k to the feature count for Laplace smoothing
          denominator = current_label_count + 2 * k # adding 2k since the feature has two possible values (0 or 1)
          current_conditional_probability = numerator / denominator # calculating P(feature = 1 | label)

          conditional_probabilities[(feature, label)] = current_conditional_probability # storing the condiitonal probability for the specific feature and label
      
      self.conditionalProb = conditional_probabilities # storing the probablities of the current smoothing factor k

      validation_accuracy = self.calculate_validation_accuracy(validationData, validationLabels) # calculating validation accuracy for the current smoothing factor k

      # checking if the current smoothing factor k gives better accuracy
      if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy # updating best_accuracy
        best_smoothing_factor = k # updating best_smoothing_factor
        best_conditional_probability = conditional_probabilities # updating best_conditional_probability

    self.k = best_smoothing_factor # saving the best smoothing factor k
    self.conditionalProb = best_conditional_probability # saving the conditional probability associated with the best k

  def calculate_validation_accuracy(self, validation_data, validation_labels):
    predicted_labels = self.classify(validation_data) # predicting the labels for all images using current model
    
    total_correct_predictions = 0 # storing the total correct predictions
    total_validation_labels = len(validation_labels) # getting the length of the total validation labels

    # going through every validation labels
    for index in range(total_validation_labels):
      predicted_label = predicted_labels[index] # getting the predicted label by the classifer
      actual_label = validation_labels[index] # getting the actual correct label

      if predicted_label == actual_label: # checking if the predicted label equals the actual label
        total_correct_predictions += 1 # if accurate then increase the total correct predictions

    validation_accuracy = total_correct_predictions / total_validation_labels # calculating the accuracy by the total correct predictions / total images
    return validation_accuracy # returning validation accuracy

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()

    # going through every possible label (0-9)
    for label in self.legalLabels:
      prior_probability = self.prior_probability[label] # getting the prior probability of the label P(y)
      log_probability = math.log(prior_probability) # calculating log P(y)

      # going through every feature in the image
      for feature in self.features:
        feature_number = datum[feature] # getting the feature number where 0 = off and 1 = on
        probability_feature_on = self.conditionalProb[(feature, label)] # getting P(feature = 1 | label) from the trained model
        
        # if the pixel is on then P(feature = 1 | label)
        if feature_number == 1:
          feature_probability = probability_feature_on
        else: # else the pixel is off so P(feature = 0 | label)
          feature_probability = 1 - probability_feature_on

        log_probability += math.log(feature_probability) # adding log P(feature | label) to the total log probability
      
      logJoint[label] = log_probability # storing the final log probability for the label

    return logJoint # returning all log probabilities for all possible labels

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
    
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
