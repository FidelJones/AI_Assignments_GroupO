import math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Dict, Any, Optional, Iterable, Set
import gymnasium as gym
import numpy as np

import util
from util import ContinuousGymMDP, StateT, ActionT
from custom_mountain_car import CustomMountainCarEnv

############################################################
# Problem 3a
# Implementing Value Iteration on Number Line (from Problem 1)
def valueIteration(succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]], discount: float, epsilon: float = 0.001):
    '''
    Given transition probabilities and rewards, computes and returns V and
    the optimal policy pi for each state.
    - succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
    - Returns: Dictionary mapping each state to an action.
    '''
    # Define a mapping from states to Set[Actions] so we can determine all the actions that can be taken from s.
    # You may find this useful in your approach.
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        # Return Q(state, action) based on V(state)
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        q = 0.0
        for (nextState, prob, reward) in succAndRewardProb.get((state, action), []):
            q += prob * (reward + discount * V[nextState])
        return q
        # END_YOUR_CODE

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        # Return the policy given V.
        # Remember the policy for a state is the action that gives the greatest Q-value.
        # IMPORTANT: if multiple actions give the same Q-value, choose the largest action number for the policy. 
        # HINT: We only compute policies for states in stateActions.
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        policy = {}
        for state in stateActions:
            best_a = None
            best_q = -float('inf')
            for action in stateActions[state]:
                q = computeQ(V, state, action)
                # tie-breaker: choose largest action when Q equal
                if (q > best_q) or (math.isclose(q, best_q) and (best_a is None or action > best_a)):
                    best_q = q
                    best_a = action
            # best_a might still be None if no actions for state; skip in that case
            if best_a is not None:
                policy[state] = best_a
        return policy
        # END_YOUR_CODE

    print('Running valueIteration...')
    V = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
    numIters = 0
    while True:
        newV = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
        # update V values using the computeQ function above.
        # repeat until the V values for all states do not change by more than epsilon.
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        delta = 0.0
        # update only for states that have available actions
        for state in stateActions:
            best = -float('inf')
            for action in stateActions[state]:
                q = computeQ(V, state, action)
                if q > best:
                    best = q
            # If state has no actions, best remains -inf; treat as 0 (terminal)
            if best == -float('inf'):
                best = 0.0
            newV[state] = best
            delta = max(delta, abs(newV[state] - V[state]))
        numIters += 1
        V = newV
        if delta < epsilon:
            break
        # END_YOUR_CODE
    V_opt = V
    print(("valueIteration: %d iterations" % numIters))
    return computePolicy(V_opt)

############################################################
# Problem 3b
# Model-Based Monte Carlo

# Runs value iteration algorithm on the number line MDP
# and prints out optimal policy for each state.
def run_VI_over_numberLine(mdp: util.NumberLineMDP):
    succAndRewardProb = {
        (-mdp.n + 1, 1): [(-mdp.n + 2, 0.2, mdp.penalty), (-mdp.n, 0.8, mdp.leftReward)],
        (-mdp.n + 1, 2): [(-mdp.n + 2, 0.3, mdp.penalty), (-mdp.n, 0.7, mdp.leftReward)],
        (mdp.n - 1, 1): [(mdp.n - 2, 0.8, mdp.penalty), (mdp.n, 0.2, mdp.rightReward)],
        (mdp.n - 1, 2): [(mdp.n - 2, 0.7, mdp.penalty), (mdp.n, 0.3, mdp.rightReward)]
    }

    for s in range(-mdp.n + 2, mdp.n - 1):
        succAndRewardProb[(s, 1)] = [(s+1, 0.2, mdp.penalty), (s - 1, 0.8, mdp.penalty)]
        succAndRewardProb[(s, 2)] = [(s+1, 0.3, mdp.penalty), (s - 1, 0.7, mdp.penalty)]

    pi = valueIteration(succAndRewardProb, mdp.discount)
    return pi


class ModelBasedMonteCarlo(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, calcValIterEvery: int = 10000,
                 explorationProb: float = 0.2,) -> None:
        self.actions = actions
        self.discount = discount
        self.calcValIterEvery = calcValIterEvery
        self.explorationProb = explorationProb
        self.numIters = 0

        # (state, action) -> {nextState -> ct} for all nextState
        self.tCounts = defaultdict(lambda: defaultdict(int))
        # (state, action) -> {nextState -> totalReward} for all nextState
        self.rTotal = defaultdict(lambda: defaultdict(float))

        self.pi = {} # Optimal policy for each state. state -> action

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # Should return random action if the given state is not in self.pi.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always follow the policy if available.
    # HINT: Use random.random() (not np.random()) to sample from the uniform distribution [0, 1]
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e6: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        # If not exploring or exploration draw fails, follow policy if available, else random
        if explore and random.random() < explorationProb:
            return random.choice(self.actions)
        # follow learned policy if available, else return random action
        return self.pi.get(state, random.choice(self.actions))
        # END_YOUR_CODE

    # We will call this function with (s, a, r, s'), which is used to update tCounts and rTotal.
    # For every self.calcValIterEvery steps, runs value iteration after estimating succAndRewardProb.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):

        self.tCounts[(state, action)][nextState] += 1
        self.rTotal[(state, action)][nextState] += reward

        if self.numIters % self.calcValIterEvery == 0:
            # Estimate succAndRewardProb based on self.tCounts and self.rTotal.
            # Hint 1: prob(s, a, s') = (counts of transition (s,a) -> s') / (total transtions from (s,a))
            # Hint 2: Reward(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
            # Then run valueIteration and update self.pi.
            succAndRewardProb = defaultdict(list)
            # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
            for (s_a), nexts in list(self.tCounts.items()):
                total = float(sum(nexts.values()))
                if total <= 0.0:
                    continue
                s, a = s_a
                for nextState, ct in nexts.items():
                    prob = ct / total
                    avg_reward = self.rTotal[(s, a)][nextState] / ct
                    succAndRewardProb[(s, a)].append((nextState, prob, avg_reward))
            # Run value iteration on the estimated model
            try:
                self.pi = valueIteration(succAndRewardProb, self.discount)
            except Exception as e:
                # If value iteration fails for some reason, keep existing policy
                print("ModelBasedMonteCarlo: valueIteration failed:", e)
            # END_YOUR_CODE

############################################################
# Problem 4a
# Performs Tabular Q-learning. Read util.RLAlgorithm for more information.
class TabularQLearning(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        - intialQ: the value for intializing Q values.
        '''
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT 1: You can access Q-value with self.Q[state, action]
    # HINT 2: Use random.random() to sample from the uniform distribution [0, 1]
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        if explore and random.random() < explorationProb:
            return random.choice(self.actions)
        # choose action with highest Q; tie-breaker: largest action number
        best_a = None
        best_q = -float('inf')
        for a in self.actions:
            q = self.Q[(state, a)]
            if (q > best_q) or (math.isclose(q, best_q) and (best_a is None or a > best_a)):
                best_q = q
                best_a = a
        return best_a
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |Q|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update the Q values using self.getStepSize() 
    # HINT 1: The target V for the current state is a combination of the immediate reward
    # and the discounted future value.
    # HINT 2: V for terminal states is 0
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        alpha = self.getStepSize()
        cur = self.Q[(state, action)]
        if terminal:
            target = reward
        else:
            # V(nextState) = max_a Q(nextState, a)
            best_next = -float('inf')
            for a in self.actions:
                best_next = max(best_next, self.Q[(nextState, a)])
            target = reward + self.discount * best_next
        self.Q[(state, action)] = cur + alpha * (target - cur)
        # END_YOUR_CODE

############################################################
# Problem 4b: Fourier feature extractor

def fourierFeatureExtractor(
        state: StateT,
        maxCoeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), maxCoeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]
    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    # Below, implement the fourier feature extractor as similar to the doc string provided.
    # The return shape should be 1 dimensional ((maxCoeff+1)^(len(state)),).
    #
    # HINT: refer to util.polynomialFeatureExtractor as a guide for
    # doing efficient arithmetic broadcasting in numpy.

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    s = np.array(state, dtype=float)
    scale_arr = np.array(scale, dtype=float)
    # ensure scale matches state shape
    if scale_arr.shape != s.shape:
        scale_arr = np.ones_like(s) * scale_arr
    # create all coefficient combinations (cartesian product of 0..maxCoeff for each dimension)
    dims = len(s)
    # number of features:
    total = (maxCoeff + 1) ** dims
    coeffs_list = np.indices([maxCoeff + 1] * dims).reshape(dims, -1).T  # shape (total, dims)
    # compute dot product coeffs . (s * scale)
    scaled_state = s * scale_arr
    dots = np.dot(coeffs_list, scaled_state)
    features = np.cos(np.pi * dots)
    return features
    # END_YOUR_CODE

############################################################
# Problem 4c: Q-learning with Function Approximation
# Performs Function Approximation Q-learning. Read util.RLAlgorithm for more information.
class FunctionApproxQLearning(util.RLAlgorithm):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, explorationProb=0.2):
        '''
        - featureDim: the dimensionality of the output of the feature extractor
        - featureExtractor: a function that takes a state and returns a numpy array representing the feature.
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        '''
        self.featureDim = featureDim
        self.featureExtractor = featureExtractor
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.W = np.random.standard_normal(size=(featureDim, len(actions)))
        self.numIters = 0

    def getQ(self, state: np.ndarray, action: int) -> float:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = np.array(self.featureExtractor(state)).reshape(-1)
        # ensure W shape consistent
        return float(np.dot(phi, self.W[:, action]))
        # END_YOUR_CODE

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT: This function should be the same as your implementation for 4a.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        if explore and random.random() < explorationProb:
            return random.choice(self.actions)
        # choose best action by Q; tie-breaker: largest action
        best_a = None
        best_q = -float('inf')
        for a in self.actions:
            q = self.getQ(state, a)
            if (q > best_q) or (math.isclose(q, best_q) and (best_a is None or a > best_a)):
                best_q = q
                best_a = a
        return best_a
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.005 * (0.99)**(self.numIters / 500)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update W using self.getStepSize()
    # HINT 1: this part will look similar to 4a, but you are updating self.W
    # HINT 2: review the function approximation module for the update rule
    def incorporateFeedback(self, state: np.ndarray, action: int, reward: float, nextState: np.ndarray, terminal: bool) -> None:
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        alpha = self.getStepSize()
        phi = np.array(self.featureExtractor(state)).reshape(-1)
        cur_q = float(np.dot(phi, self.W[:, action]))
        if terminal:
            target = reward
        else:
            # target uses max_a' Q(nextState, a')
            best_next = -float('inf')
            for a in self.actions:
                best_next = max(best_next, self.getQ(nextState, a))
            target = reward + self.discount * best_next
        error = target - cur_q
        # gradient step for the column corresponding to 'action'
        self.W[:, action] = self.W[:, action] + alpha * error * phi
        # END_YOUR_CODE

############################################################
# Problem 5c: Constrained Q-learning

class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, force: float, gravity: float,
                 max_speed: Optional[float] = None,
                 explorationProb=0.2):
        super().__init__(featureDim, featureExtractor, actions,
                         discount, explorationProb)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action that is valid.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
        # Define helper to check validity of an action under max_speed constraint.
        # NOTE: This uses a reasonably common approximate dynamic for mountain car:
        #   new_vel = v + (action - 1) * force - gravity * cos(3 * pos)
        # where action is assumed in {0,1,2} mapping to accelerations -force, 0, +force via (action-1).
        # If your CustomMountainCarEnv uses a different dynamic, replace this calculation accordingly.
        def action_valid_for_state(s, a):
            if self.max_speed is None:
                return True
            pos = float(s[0])
            vel = float(s[1])
            accel = (a - 1) * self.force
            # gravity effect approximation (common in mountain car): -gravity * cos(3*pos)
            gravity_effect = - self.gravity * math.cos(3 * pos)
            new_vel = vel + accel + gravity_effect
            # check against max_speed threshold (absolute)
            return abs(new_vel) <= float(self.max_speed)

        valid_actions = [a for a in self.actions if action_valid_for_state(state, a)]
        # if no valid actions (rare), fall back to all actions
        if len(valid_actions) == 0:
            valid_actions = list(self.actions)

        if explore and random.random() < explorationProb:
            return random.choice(valid_actions)

        # choose best among valid actions with tie-breaker largest action
        best_a = None
        best_q = -float('inf')
        for a in valid_actions:
            q = self.getQ(state, a)
            if (q > best_q) or (math.isclose(q, best_q) and (best_a is None or a > best_a)):
                best_q = q
                best_a = a
        return best_a
        # END_YOUR_CODE

############################################################
# This is helper code for comparing the predicted optimal
# actions for 2 MDPs with varying max speed constraints
gym.register(
    id="CustomMountainCar-v0",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

mdp1 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)
mdp2 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)

# This is a helper function for 5c. This function runs
# ConstrainedQLearning, then simulates various trajectories through the MDP
# and compares the frequency of various optimal actions.
def compare_MDP_Strategies(mdp1: ContinuousGymMDP, mdp2: ContinuousGymMDP):
    rl1 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        mdp1.env.force,
        mdp1.env.gravity,
        10000,
        explorationProb=0.2,
    )
    rl2 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        mdp2.env.force,
        mdp2.env.gravity,
        0.065,
        explorationProb=0.2,
    )
    sampleKRLTrajectories(mdp1, rl1)
    sampleKRLTrajectories(mdp2, rl2)

def sampleKRLTrajectories(mdp: ContinuousGymMDP, rl: ConstrainedQLearning):
    accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
    for n in range(100):
        traj = util.sample_RL_trajectory(mdp, rl)
        accelerate_left = traj.count(0)
        no_accelerate = traj.count(1)
        accelerate_right = traj.count(2)

    print(f"\nRL with MDP -> start state:{mdp.startState()}, max_speed:{rl.max_speed}")
    print(f"  *  total accelerate left actions: {accelerate_left}, total no acceleration actions: {no_accelerate}, total accelerate right actions: {accelerate_right}")
