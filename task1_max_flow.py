import logging
from colorama import init, Fore
from tabulate import tabulate
import matplotlib.pyplot as plt
from time import time

import networkx as nx  # For graph visualization

# ==============================
# Initialize colorama and logger
# ==============================
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


class EdmondKarpMaxFlow:
    """
    Class that implements the Edmond-Karp algorithm for computing
    the maximum flow in a directed graph (with explicit tracking of flow).
    """

    def __init__(self):
        """
        Constructor that initializes:
         - self.graph[u][v]: capacity from u to v
         - self.flow[u][v]: actual flow used from u to v
         - self.V: set of vertices
        """
        self.graph = {}
        self.flow = {}
        self.V = set()

    def add_edge(self, u, v, capacity):
        """
        Add an edge (u -> v) with a given capacity to the graph.
        Also initialize the flow on this edge as 0.
        """
        if u not in self.graph:
            self.graph[u] = {}
        self.graph[u][v] = capacity

        if u not in self.flow:
            self.flow[u] = {}
        self.flow[u][v] = 0  # Initialize flow to 0

        # Keep track of vertices
        self.V.add(u)
        self.V.add(v)

    def _bfs_find_path(self, source, sink, parent, step_log=False):
        """
        Private helper function to find a path with available capacity
        in the residual graph using BFS.
        'parent' is used to store the path (who leads to whom).
        Returns True if a path is found, otherwise False.

        If step_log=True, it will print BFS steps to the console for demonstration.
        """
        visited = set()
        queue = [source]
        visited.add(source)

        if step_log:
            logger.info(
                Fore.CYAN + f"[BFS] Starting BFS from {source} to find a path to {sink}"
            )

        while queue:
            u = queue.pop(0)

            # Iterate over possible neighbors in the residual graph
            for v in self.graph.get(u, {}):
                # residual capacity = self.graph[u][v] - self.flow[u][v]
                # BUT here we keep it simpler by referencing "capacity - used flow" directly
                residual_capacity = self.graph[u][v] - self.flow[u][v]

                if residual_capacity > 0 and v not in visited:
                    visited.add(v)
                    parent[v] = u
                    if step_log:
                        logger.info(
                            f"[BFS] Found edge with capacity left: {u} -> {v} "
                            f"(residual={residual_capacity})"
                        )
                    queue.append(v)

                    if v == sink:
                        if step_log:
                            logger.info(
                                Fore.MAGENTA + "[BFS] Reached the sink, path found!"
                            )
                        return True

        if step_log:
            logger.info("[BFS] No path found in this BFS.")
        return False

    def edmonds_karp(self, source, sink, verbose=False):
        """
        Edmond-Karp implementation to find the maximum flow from source to sink.
        Returns the max flow value.

        If verbose=True, we log BFS steps and path augmentation details.
        """
        parent = {}
        max_flow = 0
        start_time = time()
        iteration_count = 0

        while self._bfs_find_path(source, sink, parent, step_log=verbose):
            iteration_count += 1
            # Find the bottleneck (minimum residual capacity) along the found path
            path_flow = float("inf")
            s = sink
            while s != source:
                u = parent[s]
                residual_capacity = self.graph[u][s] - self.flow[u][s]
                path_flow = min(path_flow, residual_capacity)
                s = u

            # Update flows along the path
            v = sink
            path_edges_str = []
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow  # add flow forward
                # Create reverse edge if needed
                if v not in self.flow:
                    self.flow[v] = {}
                if u not in self.flow[v]:
                    self.flow[v][u] = 0
                self.flow[v][u] -= path_flow  # reverse flow

                path_edges_str.append(f"{u}->{v}")
                v = u

            max_flow += path_flow

            if verbose:
                path_edges_str.reverse()  # so it prints from source to sink
                logger.info(
                    Fore.GREEN + f"[Iteration {iteration_count}] Path found: "
                    f"{' -> '.join(path_edges_str)}, "
                    f"bottleneck = {path_flow}"
                )

        elapsed_time = time() - start_time
        logger.info(
            Fore.GREEN + f"Edmond-Karp completed. Max flow = {max_flow}, "
            f"time = {elapsed_time:.6f}s, iterations = {iteration_count}"
        )
        return max_flow


class LogisticsFlowTask:
    """
    High-level class to demonstrate the usage of EdmondKarpMaxFlow
    for a logistics network scenario (Task 1).
    """

    def __init__(self, verbose=False):
        """
        Initializes the solver and sets up the example graph from the assignment.
        `verbose` will turn on BFS step logging if True.
        """
        self.flow_solver = EdmondKarpMaxFlow()
        self.source = "SuperSource"
        self.sink = "SuperSink"
        self.terminals = ["Terminal 1", "Terminal 2"]
        self.storages = ["Склад 1", "Склад 2", "Склад 3", "Склад 4"]
        self.shops = [
            "Магазин 1",
            "Магазин 2",
            "Магазин 3",
            "Магазин 4",
            "Магазин 5",
            "Магазин 6",
            "Магазин 7",
            "Магазин 8",
            "Магазин 9",
            "Магазин 10",
            "Магазин 11",
            "Магазин 12",
            "Магазин 13",
            "Магазин 14",
        ]
        self.verbose = verbose
        self._build_graph()
        self.max_flow_value = 0

    def _build_graph(self):
        """
        Builds the logistic network graph:
        Adds a SuperSource -> Terminal edges,
        Terminal -> Storage edges,
        Storage -> Shop edges,
        Shop -> SuperSink edges.
        """
        # Connect SuperSource to terminals with large capacity
        self.flow_solver.add_edge(self.source, "Terminal 1", 9999)
        self.flow_solver.add_edge(self.source, "Terminal 2", 9999)

        # Terminal 1 edges
        self.flow_solver.add_edge("Terminal 1", "Склад 1", 25)
        self.flow_solver.add_edge("Terminal 1", "Склад 2", 20)
        self.flow_solver.add_edge("Terminal 1", "Склад 3", 15)

        # Terminal 2 edges
        self.flow_solver.add_edge("Terminal 2", "Склад 3", 15)
        self.flow_solver.add_edge("Terminal 2", "Склад 4", 30)
        self.flow_solver.add_edge("Terminal 2", "Склад 2", 10)

        # Склад 1 -> Магазини
        self.flow_solver.add_edge("Склад 1", "Магазин 1", 15)
        self.flow_solver.add_edge("Склад 1", "Магазин 2", 10)
        self.flow_solver.add_edge("Склад 1", "Магазин 3", 20)

        # Склад 2 -> Магазини
        self.flow_solver.add_edge("Склад 2", "Магазин 4", 15)
        self.flow_solver.add_edge("Склад 2", "Магазин 5", 10)
        self.flow_solver.add_edge("Склад 2", "Магазин 6", 25)

        # Склад 3 -> Магазини
        self.flow_solver.add_edge("Склад 3", "Магазин 7", 20)
        self.flow_solver.add_edge("Склад 3", "Магазин 8", 15)
        self.flow_solver.add_edge("Склад 3", "Магазин 9", 10)

        # Склад 4 -> Магазини
        self.flow_solver.add_edge("Склад 4", "Магазин 10", 20)
        self.flow_solver.add_edge("Склад 4", "Магазин 11", 10)
        self.flow_solver.add_edge("Склад 4", "Магазин 12", 15)
        self.flow_solver.add_edge("Склад 4", "Магазин 13", 5)
        self.flow_solver.add_edge("Склад 4", "Магазин 14", 10)

        # Connect shops to SuperSink
        for shop in self.shops:
            self.flow_solver.add_edge(shop, self.sink, 9999)

    def run_flow_calculation(self):
        """
        Runs Edmond-Karp to compute the max flow from SuperSource to SuperSink.
        Stores the result in self.max_flow_value.
        """
        self.max_flow_value = self.flow_solver.edmonds_karp(
            self.source, self.sink, verbose=self.verbose
        )
        return self.max_flow_value

    def _flow_decomposition_for(self, start_node, end_node):
        """
        Decompose how much flow goes from start_node to end_node using
        the final flow graph (self.flow_solver.flow).

        We'll do a repeated BFS on the *flow* graph (edges where flow>0),
        each time extracting a path. Summing these path flows
        will give the total flow from start_node to end_node.
        """

        # Make a copy of flows to track the usage as we find paths
        residual_flow = {}
        for u in self.flow_solver.flow:
            residual_flow[u] = {}
            for v in self.flow_solver.flow[u]:
                f = self.flow_solver.flow[u][v]
                if f > 0:
                    residual_flow[u][v] = f
                # else not needed => no edge

        total_flow = 0

        while True:
            parent = {}
            found_flow = self._bfs_find_path_in_flow_graph(
                start_node, end_node, parent, residual_flow
            )
            if found_flow <= 0:
                break
            total_flow += found_flow
            # Update residual_flow
            # walk backward from end_node to start_node
            v = end_node
            while v != start_node:
                u = parent[v]
                residual_flow[u][v] -= found_flow
                if residual_flow[u][v] == 0:
                    del residual_flow[u][v]
                # Add reverse if needed
                if v not in residual_flow:
                    residual_flow[v] = {}
                if u not in residual_flow[v]:
                    residual_flow[v][u] = 0
                residual_flow[v][u] += found_flow
                v = u

        return total_flow

    def _bfs_find_path_in_flow_graph(self, source, sink, parent, flow_graph):
        """
        BFS in the 'flow_graph', which has edges only if flow>0,
        to find any path from source to sink.
        Returns the bottleneck flow on that path (>=1) if found, else 0.
        """
        visited = set([source])
        queue = [source]
        parent.clear()

        while queue:
            u = queue.pop(0)
            if u in flow_graph:
                for v in flow_graph[u]:
                    if v not in visited and flow_graph[u][v] > 0:
                        visited.add(v)
                        parent[v] = u
                        if v == sink:
                            # reconstruct path to find min
                            path_flow = float("inf")
                            s = sink
                            while s != source:
                                rcap = flow_graph[parent[s]][s]
                                path_flow = min(path_flow, rcap)
                                s = parent[s]
                            return path_flow
                        queue.append(v)
        return 0

    def build_terminal_shop_table(self):
        """
        Build a complete table:
         - Terminal
         - Shop
         - Actual Flow (units)
        using the final flows from the algorithm.

        We'll do a decomposition for each (Terminal, Shop) pair.
        """
        rows = []
        for terminal in self.terminals:
            for shop in self.shops:
                if shop.startswith("Магазин"):
                    f = self._flow_decomposition_for(terminal, shop)
                    rows.append([terminal, shop, f])
        return rows

    def generate_final_report(self):
        """
        1) Print a full table (Terminal, Shop, Flow).
        2) Print answers to the 4 questions from the assignment.
        3) Visualize the network in NetworkX with capacity and final flow.
        """
        logger.info(Fore.CYAN + "\n=== 1) FULL TERMINAL-SHOP FLOW TABLE ===")

        # Build full table
        table_data = self.build_terminal_shop_table()
        print(
            tabulate(
                table_data, headers=["Terminal", "Shop", "Flow"], tablefmt="github"
            )
        )

        # -------------------------------
        # 2) Answers to the 4 questions
        # -------------------------------
        logger.info(Fore.CYAN + "\n=== 2) ANSWERS TO THE FOUR QUESTIONS ===")

        # Q1. Which terminals deliver the largest flow to shops?
        # We'll sum all flows for each terminal from the table we just built.
        terminal_total = {}
        for row in table_data:
            t, s, f = row
            terminal_total[t] = terminal_total.get(t, 0) + f

        # Sort by flow desc
        sorted_terminals = sorted(
            terminal_total.items(), key=lambda x: x[1], reverse=True
        )
        # The highest flow value (it could be the same for multiple terminals).
        top_flow_value = sorted_terminals[0][1]

        # Collect all terminals that share this top flow.
        top_terminals = [
            term for term, flow in sorted_terminals if flow == top_flow_value
        ]

        logger.info(
            Fore.YELLOW + "Q1: Which terminal(s) have the highest total flow to shops?"
        )
        for term, val in sorted_terminals:
            logger.info(f"  - {term}: total flow = {val}")
        logger.info(
            Fore.YELLOW
            + f"Answer: The top terminal(s) with flow {top_flow_value} unit(s): "
            + ", ".join(top_terminals)
            + "\n"
        )

        # Q2. Which routes have the smallest capacity and how does that affect overall flow?
        #    Let's check edges with minimal capacity. We'll find the minimum capacity among all edges.
        all_edges = []
        for u in self.flow_solver.graph:
            for v, cap in self.flow_solver.graph[u].items():
                all_edges.append((u, v, cap))

        min_cap = min(e[2] for e in all_edges if e[2] > 0)
        smallest_edges = [(u, v, c) for (u, v, c) in all_edges if c == min_cap]

        logger.info(
            Fore.YELLOW
            + "Q2: Which routes have the smallest capacity, and how does it affect the overall flow?"
        )
        logger.info(
            f"  - The smallest capacity in the network is {min_cap}. Edges with this capacity:"
        )
        for u, v, c in smallest_edges:
            logger.info(f"    {u} -> {v}, capacity = {c}")
        logger.info(
            "  - These edges can become potential bottlenecks if the flow wants to exceed this limit.\n"
        )

        # Q3. Which shops got the fewest items, and can we increase their supply by boosting capacity on certain routes?
        #    We'll sum flow for each shop in the table_data as well.
        shop_total = {}
        for row in table_data:
            t, s, f = row
            shop_total[s] = shop_total.get(s, 0) + f

        sorted_shops = sorted(shop_total.items(), key=lambda x: x[1])  # ascending
        fewest_shop = sorted_shops[0]

        logger.info(
            Fore.YELLOW
            + "Q3: Which shops received the least goods? Can we increase their supply by increasing capacity?"
        )
        for s, val in sorted_shops:
            logger.info(f"  - {s}: total inflow = {val}")
        logger.info(
            Fore.YELLOW
            + f"Answer: The shop with the least flow is {fewest_shop[0]} with {fewest_shop[1]} units."
        )
        logger.info(
            "  If we want to increase supply, we could investigate the path(s) leading to that shop and increase capacity on the critical edges.\n"
        )

        # Q4. Are there bottlenecks that can be removed to improve the network's efficiency?
        #    We'll look for edges fully used: flow == capacity (or close to capacity).
        #    Because we store capacity in self.flow_solver.graph and final flow in self.flow_solver.flow,
        #    we compare them.
        logger.info(
            Fore.YELLOW
            + "Q4: Are there bottlenecks that can be removed to improve efficiency?"
        )
        fully_used_edges = []
        for u in self.flow_solver.graph:
            for v in self.flow_solver.graph[u]:
                cap = self.flow_solver.graph[u][v]
                used = self.flow_solver.flow[u].get(v, 0)
                if abs(cap - used) < 1e-9:  # or cap == used for integer flows
                    # means it's fully used
                    fully_used_edges.append((u, v, cap, used))

        if fully_used_edges:
            logger.info("  The following edges are fully utilized (flow == capacity):")
            for u, v, c, f in fully_used_edges:
                logger.info(f"    {u} -> {v}, capacity={c}, flow={f}")
            logger.info(
                "  These edges may represent bottlenecks; increasing capacity here might raise overall flow."
            )
        else:
            logger.info(
                "  No edges are fully utilized, so there's no direct bottleneck.\n"
            )

        # -----------
        # 3) Graph Visualization
        # -----------
        logger.info(Fore.CYAN + "\n=== 3) NETWORK VISUALIZATION WITH NETWORKX ===")
        self.visualize_graph()

    def visualize_graph(self):
        """
        Build a NetworkX DiGraph with 'capacity' and 'flow' attributes,
        then draw it with edge labels.
        """
        G = nx.DiGraph()

        # Add all vertices
        for v in self.flow_solver.V:
            G.add_node(v)

        # Add edges with capacity and flow
        for u in self.flow_solver.graph:
            for v, cap in self.flow_solver.graph[u].items():
                fl = self.flow_solver.flow[u].get(v, 0)
                G.add_edge(u, v, capacity=cap, flow=fl)

        # Choose a layout
        pos = nx.spring_layout(G, k=2.0, seed=42)  # You can tweak for better spacing

        # Draw nodes
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="#FFD700")
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

        # Draw edges
        edge_list = list(G.edges(data=True))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for (u, v, d) in edge_list],
            edge_color="gray",
            arrows=True,
        )

        # Build edge label dict: "capacity(flow)"
        edge_labels = {}
        for u, v, d in edge_list:
            c = d["capacity"]
            f = d["flow"]
            edge_labels[(u, v)] = f"{c}({f})"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue")
        plt.title("Logistics Network: capacity(flow)")
        plt.axis("off")
        plt.show()


def main():
    """
    Main function to run Task 1 (Max Flow in logistics) with
    all required outputs: BFS step logs, final table, analysis, and NetworkX graph.
    """
    logger.info(Fore.CYAN + "=== Starting Task 1: Max Flow Computation ===")

    # Set verbose=True to see BFS steps
    logistics = LogisticsFlowTask(verbose=True)
    max_flow = logistics.run_flow_calculation()
    logger.info(Fore.YELLOW + f"Max Flow from SuperSource to SuperSink: {max_flow}")

    # Generate the final report, including full table and answers to 4 questions
    logistics.generate_final_report()


if __name__ == "__main__":
    main()
