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
    grid_rows: int = 56,
    grid_cols: int = 61
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

def show_occupancy_overlay(frame: np.ndarray, occupancy_map: np.ndarray, alpha: float = 0.5, title: str = "Occupancy Overlay"):
    grid_rows, grid_cols = occupancy_map.shape
    frame_height, frame_width = frame.shape[:2]

    cell_height = frame_height / grid_rows
    cell_width = frame_width / grid_cols

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame[..., ::-1])  # Convert BGR to RGB if using OpenCV

    # Overlay occupancy map
    ax.imshow(occupancy_map, cmap='Reds', alpha=alpha,
              extent=[0, frame_width, frame_height, 0], interpolation='nearest')

    # Draw grid lines (optional)
    for r in range(grid_rows + 1):
        y = r * cell_height
        ax.axhline(y, color='white', linestyle='--', linewidth=0.5, alpha=0.4)

    for c in range(grid_cols + 1):
        x = c * cell_width
        ax.axvline(x, color='white', linestyle='--', linewidth=0.5, alpha=0.4)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
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
        pass

    def frame_size(self, height, width):
        self._frameSize = (width, height)

    def reset(self) -> None:
        pass

    def append_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        grid_rows, grid_cols = 56, 61
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
                        log_odds = -np.log10(p / (1 - p))  # Cost
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
                            edge_weight = (w1 + w2) / 2
                            G.add_edge(node, next_node, weight=edge_weight)

        # Step 3: Connect SOURCE → first frame nodes (weight = 0)
        for node in detection_nodes[0]:
            G.add_edge(source_node, node, weight=0)

        # Step 4: Connect last frame nodes → SINK (weight = 0)
        for node in detection_nodes[-1]:
            G.add_edge(node, sink_node, weight=0)

        # Store the graph
        self.graph = G

    def solve(self, k: Optional[int] = None) -> List[List[TrackNode]]:
        self._build_graph()
        print(len(self._pMaps))

        path = nx.bellman_ford_path(self.graph, "SOURCE", "SINK", "weight")
        return (path_to_pixel_positions(path, self._frameSize[1], self._frameSize[0]))
