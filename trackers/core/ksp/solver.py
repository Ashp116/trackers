from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import supervision as sv
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from pyvis.network import Network

def path_to_pixel_positions(
    path: List[Union[str, Tuple[int, int, int]]],
    frame_height: int,
    frame_width: int,
    grid_rows: int = 97,
    grid_cols: int = 102
) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Convert path from (t, row, col) to pixel positions (x, y) for each frame t.
    Excludes 'SOURCE' and 'SINK'.
    
    Returns a list of (t, (x, y)) tuples.
    """
    cell_height = frame_height / grid_rows
    cell_width = frame_width / grid_cols

    pixel_path = []

    for node in path:
        if node == "SOURCE" or node == "SINK":
            continue
        t, row, col = node
        x = int((col + 0.5) * cell_width)
        y = int((row + 0.5) * cell_height)
        pixel_path.append((x, y))

    return pixel_path

def visualize_graph_pyvis(G: nx.DiGraph, output_file: str = "berclaz_graph.html"):
    net = Network(height="800px", width="100%", directed=True)
    net.toggle_physics(False)  # Disable physics/no gravity

    # Add nodes
    for node in G.nodes:
        if node == "SOURCE":
            net.add_node("SOURCE", label="SOURCE", color="green", shape="box", level=0)
        elif node == "SINK":
            net.add_node("SINK", label="SINK", color="red", shape="box", level=999)
        else:
            t, row, col = node
            label = f"t={t}, r={row}, c={col}"
            net.add_node(str(node), label=label, title=label, color="lightblue", level=t)

    # Add edges (convert float32 to float)
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        # Convert weight to native float if needed
        if hasattr(weight, "item"):
            weight = float(weight.item())
        else:
            weight = float(weight)

        net.add_edge(str(u), str(v), value=1.0 / (1.0 + abs(weight)), title=f"weight={weight:.2f}")

    net.show(output_file, notebook=False)

def show_occupancy_overlay_sequence(frames: list[np.ndarray],
                                    occupancy_maps: list[np.ndarray],
                                    alpha: float = 0.5,
                                    title: str = "Occupancy Overlay"):
    assert len(frames) == len(occupancy_maps), "Frames and maps must be of equal length"
    num_frames = len(frames)
    index = [0]  # Mutable so the inner function can modify it

    fig, ax = plt.subplots(figsize=(10, 8))
    image = None
    overlay = None

    def draw_frame(i):
        ax.clear()
        frame = frames[i]
        occupancy_map = occupancy_maps[i]

        grid_rows, grid_cols = occupancy_map.shape
        frame_height, frame_width = frame.shape[:2]
        cell_height = frame_height / grid_rows
        cell_width = frame_width / grid_cols

        ax.imshow(frame[..., ::-1])  # Convert BGR to RGB if needed
        ax.imshow(occupancy_map, cmap='Reds', alpha=alpha,
                  extent=[0, frame_width, frame_height, 0], interpolation='nearest')

        # Draw grid lines
        for r in range(grid_rows + 1):
            y = r * cell_height
            ax.axhline(y, color='white', linestyle='--', linewidth=0.5, alpha=0.4)
        for c in range(grid_cols + 1):
            x = c * cell_width
            ax.axvline(x, color='white', linestyle='--', linewidth=0.5, alpha=0.4)

        ax.set_title(f"{title} ({i+1}/{num_frames})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            index[0] = (index[0] + 1) % num_frames
            draw_frame(index[0])
        elif event.key == 'left':
            index[0] = (index[0] - 1) % num_frames
            draw_frame(index[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw_frame(index[0])
    plt.tight_layout()
    plt.show()

@dataclass(frozen=True)
class TrackNode:
    frame_id: int
    det_idx: int
    class_id: int
    position: tuple
    bbox: np.ndarray
    confidence: float

    def __hash__(self):
        pass

    def __eq__(self, other: Any):
        pass

    def __str__(self):
        pass

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


class KSPSolver:
    def __init__(
        self,
        path_overlap_penalty: float = 40,
        iou_weight: float = 0.9,
        dist_weight: float = 0.1,
        size_weight: float = 0.1,
        conf_weight: float = 0.1,
        entry_weight: float = 2.0,
        exit_weight: float = 2.0,
    ):
        self._pMaps = []
        self._frameSize = ()
        self._frames = []

        self.source = "SOURCE"
        self.sink = "SINK"
        pass

    def frame_size(self, height, width):
        self._frameSize = (width, height)

    def reset(self) -> None:
        pass

    def append_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        grid_rows, grid_cols = 97, 102
        frame_height, frame_width = frame.shape[:2]

        cell_height = frame_height / grid_rows
        cell_width = frame_width / grid_cols

        occupancy_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        kernel_size = 9
        sigma = 1.5
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        gaussian = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        gaussian /= gaussian.sum()
        k_half = kernel_size // 2

        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            col = int(cx / cell_width)
            row = int(cy / cell_height)

            for i in range(-k_half, k_half + 1):
                for j in range(-k_half, k_half + 1):
                    r, c = row + i, col + j
                    if 0 <= r < grid_rows and 0 <= c < grid_cols:
                        occupancy_map[r, c] += gaussian[i + k_half, j + k_half]

        occupancy_map = np.clip(occupancy_map, 0, 1)
        self._pMaps.append(occupancy_map)
        self._frames.append(frame)


    def _get_center(self, bbox: np.ndarray) -> np.ndarray:
        pass

    def set_entry_exit_regions(self, regions: List[Tuple[int, int, int, int]]) -> None:
        pass

    def set_border_entry_exit(
        self,
        use_border: bool = True,
        borders: Optional[Set[str]] = None,
        margin: int = 40,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        pass

    def _in_door(self, node: TrackNode) -> bool:
        pass

    def _edge_cost(self, nodeU: TrackNode, nodeV: TrackNode) -> float:
        pass

    def _build_graph(self):
        """
        Build a space-time graph for offline tracking using Berclaz's formulation.
        Each node is a (t, row, col) grid cell with occupancy.
        Edges are weighted by average negative log-odds of occupancy.
        """
        G = nx.DiGraph()
        num_frames = len(self._pMaps)
        grid_rows, grid_cols = self._pMaps[0].shape

        source_node = 'SOURCE'
        sink_node = 'SINK'
        G.add_node(source_node)
        G.add_node(sink_node)

        radius_scale = 2

        spatial_offsets = [
            (dy * radius_scale, dx * radius_scale)
            for dy in [-1, 0, 1]
            for dx in [-1, 0, 1]
        ]

        epsilon = 1e-6  # Avoid division by zero and log(0)
        detection_nodes = []

        # Step 1: Add detection nodes with log-odds as weight
        for t, occ_map in enumerate(self._pMaps):
            frame_nodes = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    p = np.clip(occ_map[row, col], epsilon, 1 - epsilon)
                    if p > 0:
                        node = (t, row, col)
                        log_odds = -np.log(p / (1 - p))  # Cost
                        G.add_node(node, weight=log_odds)
                        frame_nodes.append(node)
            detection_nodes.append(frame_nodes)

        # Step 2: Add temporal edges with cost = avg of log-odds between adjacent frames
        for t in range(num_frames - 1):
            curr_nodes = detection_nodes[t]
            next_nodes_set = set(detection_nodes[t + 1])  # Fast lookup

            for node in curr_nodes:
                _, row, col = node
                w1 = G.nodes[node]['weight']

                for dr, dc in spatial_offsets:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                        next_node = (t + 1, nr, nc)
                        if next_node in next_nodes_set:
                            w2 = G.nodes[next_node]['weight']
                            edge_weight = w1
                            G.add_edge(node, next_node, weight=edge_weight)

        # Step 3: Connect SOURCE first frame nodes (weight = 0)
        for node in detection_nodes[0]:
            G.add_edge(source_node, node, weight=0)

        # Step 4: Connect last frame nodes SINK (weight = 0)
        for node in detection_nodes[-1]:
            G.add_edge(node, sink_node, weight=0)

        # Store the graph
        self.graph = G

    def extend_graph(self, path):
        for i in range(len(path) - 1):
            vi = path[i]
            vj = path[i + 1]

            if not self.graph.has_edge(vi, vj):
                continue

            # Split node vj (except source/sink)
            if vj not in (self.source, self.sink):
                vj_in = (vj, 'in')
                vj_out = (vj, 'out')

                # Add in/out nodes if not already
                if not self.graph.has_node(vj_in):
                    self.graph.add_node(vj_in)
                if not self.graph.has_node(vj_out):
                    self.graph.add_node(vj_out)

                # Connect in → out with zero-cost edge
                if not self.graph.has_edge(vj_in, vj_out):
                    self.graph.add_edge(vj_in, vj_out, weight=0)

                # Redirect incoming edges to vj → vj_in
                for u in list(self.graph.predecessors(vj)):
                    edge_data = self.graph.get_edge_data(u, vj)
                    self.graph.add_edge(u, vj_in, **edge_data)
                    self.graph.remove_edge(u, vj)

                # Redirect outgoing edges from vj → vj_out
                for u in list(self.graph.successors(vj)):
                    edge_data = self.graph.get_edge_data(vj, u)
                    self.graph.add_edge(vj_out, u, **edge_data)
                    self.graph.remove_edge(vj, u)

                # Remove old node vj if no edges remain
                if self.graph.degree(vj) == 0:
                    self.graph.remove_node(vj)

                # Update vj to refer to vj_in  vj_out in next steps
                vj = vj_in
                next_vj = vj_out
            else:
                next_vj = vj

            # Reverse edge vi→ vj to vj→ vi with negative cost
            edge_data = self.graph.get_edge_data(vi, vj)
            cost = edge_data.get('weight', 1)
            self.graph.remove_edge(vi, vj)
            self.graph.add_edge(next_vj, vi, weight=-cost)

    def solve(self, k: Optional[int] = None) -> List[List[TrackNode]]:
        show_occupancy_overlay_sequence(self._frames, self._pMaps)
        self._build_graph()
        print(len(self._pMaps))

        path = nx.bellman_ford_path(self.graph, "SOURCE", "SINK", "weight")
        cost = nx.bellman_ford_path_length(self.graph, "SOURCE", "SINK", "weight")
        
        pl = [path]
        costs = [cost]

        for i in range(2):
            if i != 0:
                if costs[-1] >= costs[-2]:
                    break
            
            self.extend_graph(pl[-1])
            pstarl = nx.bellman_ford_path(self.graph, "SOURCE", "SINK", "weight")
            coststar1 = nx.bellman_ford_path_length(self.graph, "SOURCE", "SINK", "weight")
            pl.append(pstarl)
            costs.append(coststar1)
        
        print(len(pl))
        print(pl)
        print(pl[-1] == pl[-2])
        return [path_to_pixel_positions(p, self._frameSize[1], self._frameSize[0]) for p in pl]
        
