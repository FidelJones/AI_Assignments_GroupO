#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    features: FeatureVector = {}
    for word in x.split():
        features[word] = features.get(word, 0) + 1
    return features
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and
      validationExamples to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    for _ in range(numEpochs):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            margin = dotProduct(weights, phi) * y
            if margin < 1:
                # Subgradient update for hinge loss: w += eta * y * phi
                increment(weights, eta * y, phi)

        # Evaluate after each epoch
        trainError = evaluatePredictor(
            trainExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)
        )
        validationError = evaluatePredictor(
            validationExamples, lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)
        )
        print(("Epoch done: train error = %s, validation error = %s" % (trainError, validationError)))
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified
      correctly by |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # note that there is intentionally flexibility in how you define phi.
    # y should be 1 or -1 as classified by the weight vector.
    # IMPORTANT: In the case that the score is 0, y should be set to 1.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi: Dict[str, int] = {}
        if len(weights) > 0:
            # Choose a random subset of features from the weight vector
            keys = list(weights.keys())
            num_active = random.randint(1, max(1, len(keys)))
            active = random.sample(keys, num_active)
            for k in active:
                phi[k] = random.randint(1, 3)
        score = dotProduct(phi, weights)
        y = 1 if score >= 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that 1 <= n <= len(x).
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        # Remove spaces and tabs, then count character n-grams
        stripped = ''.join(ch for ch in x if ch not in (' ', '\t'))
        features: Dict[str, int] = {}
        if n <= 0 or n > len(stripped):
            return features
        for i in range(len(stripped) - n + 1):
            gram = stripped[i:i + n]
            features[gram] = features.get(gram, 0) + 1
        return features
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # for debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.
    '''
    import random
    from util import dotProduct, increment 

    # --- 1. Initialization ---
    
    centers = random.sample(examples, K)
    
    example_sq_norms = [dotProduct(x, x) for x in examples]

    assignments: List[int] = [] 
    total_loss = float('inf') 

    for epoch in range(maxEpochs):
        
        center_sq_norms = [dotProduct(c, c) for c in centers]

        new_assignments: List[int] = []
        new_total_loss = 0.0
        
        for i, x in enumerate(examples):
            min_dist_sq = float('inf')
            best_center_idx = -1
            
            for j in range(K):
                mu_j = centers[j]
                
                dot_prod = dotProduct(x, mu_j)
                
                dist_sq = example_sq_norms[i] + center_sq_norms[j] - 2 * dot_prod
                
                dist_sq = max(0.0, dist_sq)
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_center_idx = j
            
            new_assignments.append(best_center_idx)
            new_total_loss += min_dist_sq

        if assignments == new_assignments:
            return centers, assignments, new_total_loss
        
        assignments = new_assignments
        total_loss = new_total_loss
        
        new_centers_sum: List[Dict[str, float]] = [{} for _ in range(K)]
        cluster_counts = [0] * K
        
        for i, z in enumerate(assignments):
            increment(new_centers_sum[z], 1.0, examples[i])
            cluster_counts[z] += 1
            
        next_centers: List[Dict[str, float]] = [{} for _ in range(K)]
        
        for j in range(K):
            count = cluster_counts[j]
            if count > 0:
                scale = 1.0 / count
                
                for feature, total_value in new_centers_sum[j].items():
                    next_centers[j][feature] = total_value * scale
            else:
                next_centers[j] = centers[j] 
        centers = next_centers 
    
    return centers, assignments, total_loss
    # END_YOUR_CODE
