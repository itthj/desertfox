"""
Burkhard-Keller Metric Tree (BK-Tree) for DesertFox.
Enables sub-linear O(log N) perceptual duplicate filtering (dHash / pHash).
Exploits triangle inequality in Hamming distance metric spaces.
"""

from typing import Dict, List, Optional, Tuple, Any

class BKTreeNode:
    def __init__(self, item_hash: str, payload: Optional[Dict[str, Any]] = None):
        self.item_hash = item_hash
        self.payload = payload or {}
        self.int_hash = int(item_hash, 16)
        self.children: Dict[int, 'BKTreeNode'] = {}  # Distance -> Child node

class BKTree:
    """
    Metric tree for efficient perceptual hash lookups with Hamming distance.
    Reduces duplicate search from O(N) to O(log N).
    """

    def __init__(self):
        self.root: Optional[BKTreeNode] = None
        self.size = 0

    @staticmethod
    def hamming_distance(val1: int, val2: int) -> int:
        """Fast bitwise XOR population count."""
        return (val1 ^ val2).bit_count()

    def insert(self, hex_hash: str, payload: Optional[Dict[str, Any]] = None):
        """Inserts a 64-bit or 16-char hex hash into the metric tree."""
        new_node = BKTreeNode(hex_hash, payload)
        if self.root is None:
            self.root = new_node
            self.size = 1
            return

        current = self.root
        while True:
            dist = self.hamming_distance(new_node.int_hash, current.int_hash)
            if dist == 0:
                # Exact perceptual match already indexed
                return
            if dist not in current.children:
                current.children[dist] = new_node
                self.size += 1
                break
            current = current.children[dist]

    def search(self, query_hex: str, max_distance: int = 4) -> List[Tuple[str, int, Dict[str, Any]]]:
        """
        Finds all indexed images with Hamming distance <= max_distance.
        Exploits triangle inequality: |d(root, query) - k| <= max_distance.
        """
        if self.root is None:
            return []

        query_int = int(query_hex, 16)
        results: List[Tuple[str, int, Dict[str, Any]]] = []
        candidates = [self.root]

        while candidates:
            node = candidates.pop()
            d = self.hamming_distance(query_int, node.int_hash)
            if d <= max_distance:
                results.append((node.item_hash, d, node.payload))

            # Triangle inequality pruning
            low = d - max_distance
            high = d + max_distance
            for dist_k, child in node.children.items():
                if low <= dist_k <= high:
                    candidates.append(child)

        return results
