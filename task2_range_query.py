import logging
from colorama import init, Fore
from tabulate import tabulate
import matplotlib.pyplot as plt

import csv
from BTrees.OOBTree import OOBTree
from timeit import timeit

# ==============================
# Initialize colorama and logger
# ==============================
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


class RangeQueryPerformance:
    """
    Class that compares the performance of OOBTree vs dict for range queries by Price.
    """

    def __init__(self, csv_path):
        """
        Constructor that loads items from a CSV file and initializes both data structures.
        OOBTree is keyed by Price (float) to allow range queries by price.
        dict is keyed by ID (string).
        """
        self.csv_path = csv_path
        # OOBTree: key = Price, value = list of items that have this Price
        self.tree = OOBTree()
        # dict: key = ID, value = item dict
        self.dct = {}
        self.items = self._load_csv_data()

        # Populate both structures
        for item in self.items:
            self.add_item_to_tree(item)
            self.add_item_to_dict(item)

    def _load_csv_data(self):
        """
        Loads items from CSV. Each row must have ID, Name, Category, Price (float).
        """
        items = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["Price"] = float(row["Price"])
                    items.append(row)
        except FileNotFoundError:
            logger.error(Fore.RED + f"CSV file not found: {self.csv_path}")
        except Exception as e:
            logger.error(Fore.RED + f"Error loading CSV: {e}")
        return items

    def add_item_to_tree(self, item):
        """
        Add an item to OOBTree, grouped by Price.
        If there's already a list under this price, append;
        otherwise create a new list.
        """
        price = item["Price"]
        if price not in self.tree:
            self.tree[price] = []
        self.tree[price].append(
            {
                "ID": item["ID"],
                "Name": item["Name"],
                "Category": item["Category"],
                "Price": item["Price"],
            }
        )

    def add_item_to_dict(self, item):
        """
        Add an item to the dict, keyed by item["ID"].
        """
        key = item["ID"]
        self.dct[key] = {
            "Name": item["Name"],
            "Category": item["Category"],
            "Price": item["Price"],
        }

    def range_query_tree(self, min_price, max_price):
        """
        Perform a range query in the OOBTree using items(min_price, max_price).
        This will return an iterator over (price, list_of_items) within [min_price, max_price].
        """
        result = []
        # OOBTree.items(minKey, maxKey, excludemin=False, excludemax=False)
        for price, items_list in self.tree.items(min_price, max_price):
            # Here, price is in [min_price, max_price]
            for itm in items_list:
                result.append((price, itm))
        return result

    def range_query_dict(self, min_price, max_price):
        """
        Perform a range query in a regular dict with a linear search by price.
        """
        result = []
        for k, v in self.dct.items():
            price = v["Price"]
            if min_price <= price <= max_price:
                result.append((k, v))
        return result

    def measure_performance(self, min_price, max_price, repeats=100):
        """
        Measure and compare the execution time for OOBTree and dict range queries,
        using the timeit library for 100 repeats.
        Returns a dictionary with the times.
        """

        # Define wrapper functions to call our queries
        def query_tree():
            self.range_query_tree(min_price, max_price)

        def query_dict():
            self.range_query_dict(min_price, max_price)

        # Measure the time for OOBTree, 100 repeats
        tree_time = timeit(
            "query_tree()", globals={"query_tree": query_tree}, number=repeats
        )

        # Measure the time for dict, 100 repeats
        dict_time = timeit(
            "query_dict()", globals={"query_dict": query_dict}, number=repeats
        )

        logger.info(
            Fore.GREEN + f"Total range_query time for OOBTree: {tree_time:.6f} seconds"
        )
        logger.info(
            Fore.BLUE + f"Total range_query time for Dict:    {dict_time:.6f} seconds"
        )

        return {"tree_time": tree_time, "dict_time": dict_time}

    def print_performance_table(self, times):
        """
        Print a small table comparing the range query times via 'tabulate'.
        """
        data = [
            ["OOBTree", f"{times['tree_time']:.6f}"],
            ["Dict", f"{times['dict_time']:.6f}"],
        ]
        print(
            tabulate(data, headers=["Structure", "Total Time (s)"], tablefmt="github")
        )

    def plot_performance(self, times):
        """
        Plot a simple bar chart to visualize the performance results.
        'times' should be a dict with keys 'tree_time' and 'dict_time'.
        """
        labels = ["OOBTree", "Dict"]
        values = [times["tree_time"], times["dict_time"]]

        plt.figure(figsize=(5, 4))
        plt.bar(labels, values, color=["#4CAF50", "#F44336"])
        plt.title("Range Query Performance Comparison")
        plt.ylabel("Time (seconds)")
        plt.show()


def main():
    """
    Main function to run Task 2: Range Query Performance.
    """
    logger.info(Fore.CYAN + "=== Starting Task 2: Range Query Performance ===")

    # Adjust CSV path if needed
    csv_file_path = "generated_items_data.csv"
    perf_tester = RangeQueryPerformance(csv_file_path)

    # For demonstration, let's pick some price range
    min_price, max_price = 10.0, 20.0

    # Measure performance (with timeit for 100 repeats)
    times = perf_tester.measure_performance(min_price, max_price, repeats=100)

    # Print table with final times
    perf_tester.print_performance_table(times)

    # Plot chart
    perf_tester.plot_performance(times)


if __name__ == "__main__":
    main()
