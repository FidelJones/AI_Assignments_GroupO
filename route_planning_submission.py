from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch


# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Please read the docstring for `State` in `util.py` for more details and code.
#   > Please read the docstrings for in `mapUtil.py`, especially for the CityMap class

########################################################################################
# Problem 2a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(self.startLocation)
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.endTag in self.cityMap.tags[state.location]
        # END_YOUR_CODE

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        """
        Note we want to return a list of *3-tuples* of the form:
            (actionToReachSuccessor: str, successorState: State, cost: float)
        Our action space is the set of all named locations, where a named location 
        string represents a transition from the current location to that new location.
        """
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        successors = []
        current_loc = state.location
        for neighbor, dist in self.cityMap.distances[current_loc].items():
            action = neighbor
            next_state = State(neighbor)
            cost = dist
            successors.append((action, next_state, cost))
        return successors
        # END_YOUR_CODE


########################################################################################
# Problem 2b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. 

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "parking_entrance", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    startLocation = locationFromTag("landmark=hoover_tower", cityMap)
    endTag = "landmark=gates"
    # END_YOUR_CODE
    return ShortestPathProblem(startLocation, endTag, cityMap)
   


########################################################################################
# Problem 3a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`. Note that tags 
    from the `startLocation` count towards covering the set of tags.

    Hint: naively, your `memory` representation could be a list of all locations visited.
    However, that would be too large of a state space to search over! Think 
    carefully about what `memory` should represent.
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        start_tags = self.cityMap.tags[self.startLocation]
        visited = tuple(sorted(tag for tag in self.waypointTags if tag in start_tags))
        return State(self.startLocation, visited)
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        if self.endTag not in self.cityMap.tags[state.location]:
            return False
        return set(state.memory) == set(self.waypointTags)
        # END_YOUR_CODE

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
        successors = []
        current_loc = state.location
        visited = set(state.memory)

        for neighbor, dist in self.cityMap.distances[current_loc].items():
            neighbor_tags = self.cityMap.tags[neighbor]
            new_visited = visited | {t for t in self.waypointTags if t in neighbor_tags}
            next_state = State(neighbor, tuple(sorted(new_visited)))
            successors.append((neighbor, next_state, dist))

        return successors
        # END_YOUR_CODE


########################################################################################
# Problem 3c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem with waypoints using the map of Stanford, 
    specifying your own `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 2b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    startLocation = locationFromTag("landmark=hoover_tower", cityMap)
    waypointTags = [
        "landmark=coupa_green_library", 
        "amenity=food", 
        "landmark=oval"
    ]
    endTag = "landmark=gates"
    # END_YOUR_CODE
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)
    
    


########################################################################################
# Problem 4a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.
# See util.py for the class definitions and methods of Heuristic and SearchProblem.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def __init__(self):
            # Store original problem and heuristic
            self.originalProblem = problem
            self.heuristic = heuristic
            
        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return self.originalProblem.startState()
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return self.originalProblem.isEnd(state)
            # END_YOUR_CODE

        def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
            successors = []
            # heuristic value at current state
            h_s = self.heuristic.evaluate(state)

            for action, nextState, cost in self.originalProblem.actionSuccessorsAndCosts(state):
                h_s_next = self.heuristic.evaluate(nextState)
                # modified cost for UCS so that UCS's accumulated cost = g(next) + h(next)
                new_cost = cost + h_s_next - h_s
                successors.append((action, nextState, new_cost))

            return successors
            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 4b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        self.end_locations_geo = [
            cityMap.geoLocations[loc]
            for loc, tags in cityMap.tags.items()
            if endTag in tags
        ]
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        if len(self.end_locations_geo) == 0:
            return 0.0

        current_geo = self.cityMap.geoLocations[state.location]

        min_dist = min(
            computeDistance(current_geo, end_geo) for end_geo in self.end_locations_geo
        )
        return min_dist
        # END_YOUR_CODE


########################################################################################
# Problem 4c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `startLocation`.
        # Solving this reversed shortest path problem will give us our heuristic,
        # as it estimates the minimal cost of reaching an end state from each state
        self.endTag = endTag
        self.cityMap = cityMap
        
        class ReverseShortestPathProblem(SearchProblem):
            def startState(self) -> State:
                """
                Return special "END" state
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                return State("!VIRTUAL_END!")
                # END_YOUR_CODE

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False), 
                UCS will exhaustively compute costs to *all* other states.
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                return False
                # END_YOUR_CODE

            def actionSuccessorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                # If current location is the special "END" state, 
                # return all the locations with the desired endTag and cost 0 
                # (i.e, we connect the special location "END" with cost 0 to all locations with endTag)
                # Else, return all the successors of current location and their corresponding distances according to the cityMap
                # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
                if state.location == "!VIRTUAL_END!":
                    successors = []
                    for loc, tags in cityMap.tags.items():
                        if endTag in tags:
                            successors.append((loc, State(loc), 0.0))
                    return successors

                successors = []
                if state.location in cityMap.distances:
                    for neighbor, dist in cityMap.distances[state.location].items():
                        successors.append((neighbor, State(neighbor), dist))
                return successors
                # END_YOUR_CODE

        # Call UCS.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`isEnd` always returns False), will exhaustively
        # compute costs to *all* other states.

        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        ucs = UniformCostSearch()
        ucs.solve(ReverseShortestPathProblem())
        # END_YOUR_CODE

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `endTag`), we can retrieve `ucs.pastCosts`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!

        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        self.pastCosts = ucs.pastCosts
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.pastCosts.get(state.location, float("inf"))
        # END_YOUR_CODE