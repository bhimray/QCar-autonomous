import math
import heapq
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Callable, Set, Any


# =========================
# Utilities
# =========================

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(order=True)
class PQItem:
    # heapq uses the first field for ordering
    priority: Tuple[float, float]
    node: int = field(compare=False)


# =========================
# Graph / Roadmap Adapter
# =========================

class RoadmapAdapter:
    """
    Adapter for Quanser SDCSRoadMap-like objects.

    You must provide:
    - neighbors(node): -> List[int]
    - edge_cost(u, v): -> float (base cost, usually distance)
    - node_xy(node): -> (x, y) world coordinates (for heuristic)
    - nearest_node(x, y): -> node id (for replanning start state)

    The default implementations try common naming patterns and can be edited.
    """

    def __init__(self, roadmap: Any):
        self.roadmap = roadmap
        self._node_xy_cache: Dict[int, Tuple[float, float]] = {}
        self._neighbors_cache: Dict[int, List[int]] = {}

    # ---- YOU MAY NEED TO EDIT THESE 4 METHODS depending on your SDCSRoadMap API ----

    def node_xy(self, node: int) -> Tuple[float, float]:
        if node in self._node_xy_cache:
            return self._node_xy_cache[node]

        # Common Quanser pattern: roadmap.get_node_pose(node) -> [x,y,psi] or [[...]]
        pose = self.roadmap.get_node_pose(node)
        # squeeze-like handling:
        if hasattr(pose, "squeeze"):
            pose = pose.squeeze()

        x = float(pose[0])
        y = float(pose[1])
        self._node_xy_cache[node] = (x, y)
        return (x, y)

    def neighbors(self, node: int) -> List[int]:
        """
        Try typical roadmap methods. If your roadmap has a different API,
        replace this with the correct call.
        """
        if node in self._neighbors_cache:
            return self._neighbors_cache[node]

        # Possible APIs:
        # - roadmap.get_node_neighbors(node)
        # - roadmap.get_successors(node)
        # - roadmap.graph adjacency dictionary
        nbrs = None

        if hasattr(self.roadmap, "get_node_neighbors"):
            nbrs = self.roadmap.get_node_neighbors(node)
        elif hasattr(self.roadmap, "get_successors"):
            nbrs = self.roadmap.get_successors(node)
        elif hasattr(self.roadmap, "graph"):
            # If roadmap.graph[node] returns neighbors
            nbrs = self.roadmap.graph.get(node, [])
        else:
            raise AttributeError(
                "RoadmapAdapter.neighbors: Could not find a neighbor API. "
                "Please implement neighbors(node) for your SDCSRoadMap version."
            )

        # Normalize to python list[int]
        nbrs = list(map(int, nbrs))
        self._neighbors_cache[node] = nbrs
        return nbrs

    def edge_cost(self, u: int, v: int) -> float:
        """
        Base cost: distance between node poses.
        If your roadmap has per-edge lengths, use those instead.
        """
        pu = self.node_xy(u)
        pv = self.node_xy(v)
        return euclid(pu, pv)

    def nearest_node(self, x: float, y: float, candidate_nodes: Optional[List[int]] = None) -> int:
        """
        Finds closest roadmap node to current pose.
        If your roadmap provides a fast lookup, replace this.
        """
        if candidate_nodes is None:
            # If roadmap has get_num_nodes
            if hasattr(self.roadmap, "get_num_nodes"):
                nodes = list(range(int(self.roadmap.get_num_nodes())))
            elif hasattr(self.roadmap, "N"):
                nodes = list(range(int(self.roadmap.N)))
            else:
                # Fallback: try to infer from cache; user should adapt
                nodes = list(self._node_xy_cache.keys())
                if not nodes:
                    raise AttributeError(
                        "RoadmapAdapter.nearest_node: No way to enumerate nodes. "
                        "Provide candidate_nodes or implement node enumeration."
                    )
        else:
            nodes = candidate_nodes

        best = nodes[0]
        best_d = float("inf")
        p = (x, y)
        for n in nodes:
            d = euclid(self.node_xy(n), p)
            if d < best_d:
                best_d = d
                best = n
        return best


# =========================
# ARA* Planner (Graph)
# =========================

class ARAStarPlanner:
    """
    ARA* on a directed graph with dynamic edge costs.

    - epsilon starts >1 and decreases toward 1
    - supports incremental repair via INCONS + OPEN reuse (classic ARA*)
    - you can update edge penalties/blocks then call replan()
    """

    def __init__(
        self,
        graph: RoadmapAdapter,
        epsilon_start: float = 3.0,
        epsilon_min: float = 1.0,
        epsilon_step: float = 0.5,
    ):
        self.G = graph

        self.eps_start = float(epsilon_start)
        self.eps_min = float(epsilon_min)
        self.eps_step = float(epsilon_step)

        # dynamic cost modifiers
        self.edge_penalty: Dict[Tuple[int, int], float] = {}  # multiplicative >=1
        self.edge_blocked: Set[Tuple[int, int]] = set()

        # ARA* search bookkeeping
        self.g: Dict[int, float] = {}
        self.parent: Dict[int, Optional[int]] = {}
        self.OPEN: List[PQItem] = []
        self.INCONS: Set[int] = set()
        self.CLOSED: Set[int] = set()

        self.start: Optional[int] = None
        self.goal: Optional[int] = None

    # ---- dynamic environment interface ----

    def block_edge(self, u: int, v: int):
        self.edge_blocked.add((u, v))

    def unblock_edge(self, u: int, v: int):
        self.edge_blocked.discard((u, v))

    def set_edge_penalty(self, u: int, v: int, penalty: float):
        self.edge_penalty[(u, v)] = max(1.0, float(penalty))

    def clear_edge_penalty(self, u: int, v: int):
        self.edge_penalty.pop((u, v), None)

    def cost(self, u: int, v: int) -> float:
        if (u, v) in self.edge_blocked:
            return float("inf")
        base = self.G.edge_cost(u, v)
        mult = self.edge_penalty.get((u, v), 1.0)
        return base * mult

    # ---- heuristic ----

    def h(self, n: int) -> float:
        # admissible Euclidean-to-goal heuristic
        assert self.goal is not None
        return euclid(self.G.node_xy(n), self.G.node_xy(self.goal))

    def key(self, n: int, eps: float) -> Tuple[float, float]:
        # key = (g(n) + eps*h(n), g(n))  (tie-break on smaller g)
        return (self.g.get(n, float("inf")) + eps * self.h(n), self.g.get(n, float("inf")))

    # ---- ARA* internals ----

    def _init_search(self, start: int, goal: int):
        self.start, self.goal = int(start), int(goal)

        self.g = {self.start: 0.0}
        self.parent = {self.start: None}

        self.OPEN = []
        self.INCONS = set()
        self.CLOSED = set()

    def _push_open(self, node: int, eps: float):
        heapq.heappush(self.OPEN, PQItem(priority=self.key(node, eps), node=node))

    def _pop_open(self) -> int:
        return heapq.heappop(self.OPEN).node

    def _rebuild_open(self, eps: float):
        # rebuild OPEN priorities when epsilon changes
        nodes = [item.node for item in self.OPEN]
        self.OPEN = []
        for n in nodes:
            self._push_open(n, eps)

        # move INCONS to OPEN
        for n in self.INCONS:
            self._push_open(n, eps)
        self.INCONS.clear()

        # clear CLOSED for next improvePath phase
        self.CLOSED.clear()

    def _improve_path(self, eps: float):
        """
        improvePath from ARA*:
        Continue while OPEN's best key < key(goal) OR goal not discovered yet.
        """
        assert self.start is not None and self.goal is not None

        # Ensure goal exists in g dict for key comparison
        if self.goal not in self.g:
            self.g[self.goal] = float("inf")
            self.parent[self.goal] = None

        def best_open_key() -> Tuple[float, float]:
            if not self.OPEN:
                return (float("inf"), float("inf"))
            return self.OPEN[0].priority

        # push start if OPEN empty (first time)
        if not self.OPEN:
            self._push_open(self.start, eps)

        while best_open_key() < self.key(self.goal, eps):
            if not self.OPEN:
                break  # no more nodes to expand

            s = self._pop_open()
            if s in self.CLOSED:
                continue  # skip stale PQ entries

            self.CLOSED.add(s)

            # expand
            for sp in self.G.neighbors(s):
                c = self.cost(s, sp)
                if math.isinf(c):
                    continue

                gs = self.g.get(s, float("inf"))
                gsp = self.g.get(sp, float("inf"))
                tentative = gs + c

                if tentative < gsp:
                    self.g[sp] = tentative
                    self.parent[sp] = s

                    if sp not in self.CLOSED:
                        self._push_open(sp, eps)
                    else:
                        self.INCONS.add(sp)

    def _extract_path(self) -> Optional[List[int]]:
        """
        Follow parents from goal back to start.
        """
        assert self.start is not None and self.goal is not None

        if self.g.get(self.goal, float("inf")) == float("inf"):
            return None

        path = []
        n = self.goal
        while n is not None:
            path.append(n)
            n = self.parent.get(n, None)
        path.reverse()

        if path and path[0] == self.start:
            return path
        return None

    # ---- public planning API ----

    def plan_anytime(self, start: int, goal: int, max_improvements: int = 10) -> Dict[str, Any]:
        """
        Runs ARA* from epsilon_start down toward epsilon_min.
        Returns best path found and intermediate solutions.
        """
        self._init_search(start, goal)

        eps = self.eps_start
        solutions = []

        # First improvement round
        self._improve_path(eps)
        path = self._extract_path()
        solutions.append((eps, path, self.g.get(self.goal, float("inf"))))

        # Decrease epsilon and repair
        improve_count = 1
        while eps > self.eps_min and improve_count < max_improvements:
            eps = max(self.eps_min, eps - self.eps_step)
            self._rebuild_open(eps)
            self._improve_path(eps)
            path = self._extract_path()
            solutions.append((eps, path, self.g.get(self.goal, float("inf"))))
            improve_count += 1

        # best is the last one (lowest eps) if feasible; else pick best feasible
        best = None
        for (e, p, cost) in solutions:
            if p is not None:
                best = (e, p, cost)
        return {
            "best": best,
            "solutions": solutions,
        }

    def replan(self, start: int, goal: int, current_eps: float = 2.0, max_improve: int = 3) -> Dict[str, Any]:
        """
        A lighter-weight anytime repair. In practice for replanning while moving:
        - Use a moderate epsilon (like 2.0) for fast response
        - Optionally refine toward 1.0 if time allows
        """
        self._init_search(start, goal)
        eps = max(self.eps_min, float(current_eps))

        solutions = []
        self._improve_path(eps)
        path = self._extract_path()
        solutions.append((eps, path, self.g.get(self.goal, float("inf"))))

        # small number of refinements
        improve = 1
        while eps > self.eps_min and improve < max_improve:
            eps = max(self.eps_min, eps - self.eps_step)
            self._rebuild_open(eps)
            self._improve_path(eps)
            path = self._extract_path()
            solutions.append((eps, path, self.g.get(self.goal, float("inf"))))
            improve += 1

        best = None
        for (e, p, cost) in solutions:
            if p is not None:
                best = (e, p, cost)

        return {"best": best, "solutions": solutions}


# =========================
# Replanning Manager (moving vehicle)
# =========================

class ReplanningManager:
    """
    Handles:
    - current route (node path)
    - generating waypointSequence via roadmap.generate_path(nodeSequence)
    - dynamic updates: blocking edges / adding penalties
    - replanning triggers

    You provide:
    - pose_provider(): -> (x, y, psi, v)
    - obstacle_ahead_checker(node_path): -> list of blocked edges or a boolean
      (THIS is where your LiDAR/camera logic plugs in)
    """

    def __init__(
        self,
        roadmap: Any,
        planner: ARAStarPlanner,
        pose_provider: Callable[[], Tuple[float, float, float, float]],
        obstacle_edge_detector: Callable[[List[int]], List[Tuple[int, int]]],
    ):
        self.roadmap = roadmap
        self.adapter = planner.G
        self.planner = planner

        self.pose_provider = pose_provider
        self.obstacle_edge_detector = obstacle_edge_detector

        self.current_node_path: Optional[List[int]] = None
        self.current_waypoints = None

        # Simple "stuck" logic
        self.last_progress_time = 0.0
        self.last_pose_xy: Optional[Tuple[float, float]] = None

    def compute_initial_plan(self, start_node: int, goal_node: int) -> List[int]:
        out = self.planner.plan_anytime(start_node, goal_node, max_improvements=6)
        best = out["best"]
        if best is None or best[1] is None:
            raise RuntimeError("No path found by ARA*.")
        node_path = best[1]
        self._set_path(node_path)
        return node_path

    def _set_path(self, node_path: List[int]):
        self.current_node_path = node_path
        # Quanser roadmap expects nodeSequence to generate a path of waypoints
        self.current_waypoints = self.roadmap.generate_path(node_path)

    def maybe_replan(self, goal_node: int, current_time_sec: float):
        """
        Call periodically in your main loop.
        Replans if:
        - obstacle blocks edges along the planned path
        - stuck / no progress
        """

        if not self.current_node_path:
            return

        # 1) Detect obstacle-caused blocked edges (user plugs in perception)
        blocked_edges = self.obstacle_edge_detector(self.current_node_path)

        # Apply blocks/penalties
        for (u, v) in blocked_edges:
            self.planner.block_edge(u, v)

        # 2) Stuck detection (very basic placeholder)
        x, y, psi, v = self.pose_provider()
        if self.last_pose_xy is None:
            self.last_pose_xy = (x, y)
            self.last_progress_time = current_time_sec
        else:
            moved = euclid((x, y), self.last_pose_xy)
            if moved > 0.30:  # moved 30 cm
                self.last_pose_xy = (x, y)
                self.last_progress_time = current_time_sec

        stuck = (current_time_sec - self.last_progress_time) > 2.0 and v < 0.05

        need_replan = bool(blocked_edges) or stuck
        if not need_replan:
            return

        # 3) Choose new start node as nearest roadmap node to current pose
        start_node = self.adapter.nearest_node(x, y)

        # 4) Replan with moderate epsilon for speed (anytime)
        out = self.planner.replan(start_node, goal_node, current_eps=2.5, max_improve=3)
        best = out["best"]
        if best is None or best[1] is None:
            # If no new route found, do not wipe current plan;
            # your behavior layer should stop safely.
            return

        new_node_path = best[1]
        self._set_path(new_node_path)


# =========================
# Example: How you would hook this into Quanser
# =========================

def example_usage():
    """
    Replace pose_provider() and obstacle_edge_detector() with your QCar/QLabs logic.
    """

    # 1) Roadmap setup
    from hal.products.mats import SDCSRoadMap  # Quanser import
    roadmap = SDCSRoadMap()

    adapter = RoadmapAdapter(roadmap)
    planner = ARAStarPlanner(adapter, epsilon_start=3.0, epsilon_min=1.0, epsilon_step=0.5)

    # 2) Pose provider: get estimated pose from your GPS/EKF
    def pose_provider():
        # TODO: replace with your estimator output
        # return (x, y, psi, v)
        return (0.0, 0.0, 0.0, 0.0)

    # 3) Obstacle-to-edge detector:
    #    given the current node path, decide which edges are blocked.
    def obstacle_edge_detector(node_path: List[int]) -> List[Tuple[int, int]]:
        # TODO: use LiDAR/camera to detect obstacles ahead,
        # then map obstacle(s) to the next few edges in node_path.
        #
        # For now, return empty list = no obstacles
        return []

    manager = ReplanningManager(
        roadmap=roadmap,
        planner=planner,
        pose_provider=pose_provider,
        obstacle_edge_detector=obstacle_edge_detector,
    )

    # 4) Plan to goal
    start_node = 0
    goal_node = 20
    node_path = manager.compute_initial_plan(start_node, goal_node)
    waypointSequence = manager.current_waypoints

    # Your existing initial pose
    initialPose = roadmap.get_node_pose(start_node)
    if hasattr(initialPose, "squeeze"):
        initialPose = initialPose.squeeze()

    # 5) Main loop (pseudo)
    t = 0.0
    dt = 0.05
    for k in range(1000):
        # ... run your controller to follow manager.current_waypoints ...
        # ... update LiDAR/camera perception ...
        manager.maybe_replan(goal_node=goal_node, current_time_sec=t)
        t += dt


if __name__ == "__main__":
    example_usage()
