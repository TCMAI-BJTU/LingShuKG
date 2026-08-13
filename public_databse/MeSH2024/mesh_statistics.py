# -*- coding: utf-8 -*-
"""Compute tree statistics for MeSH symptom subtree C23.888."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


class ResultStatistic:
    def __init__(self):
        print("MeSH Statistics")
        tree_list = self.get_mesh_tree_ls()

        self.all_tree_list = tree_list
        self.tree_set_ls = list(set(tree_list))

        self.num_of_roots()
        self.num_of_classes()
        self.num_of_synonyms()
        self.num_of_leaves()
        self.max_depth()
        self.avg_depth()
        self.avg_width()
        self.fanout_ness_factor()

    def get_mesh_tree_ls(self):
        df = pd.read_csv(ROOT / "d2024.csv")
        df.dropna(subset=["MN"], inplace=True)
        df = df[df["MN"].str.contains("C23.888")]
        df = df[["ENTRY", "MH", "MN"]]

        tree_ls = []
        self.entry_ls = []
        for entry, mh, mn in zip(df["ENTRY"], df["MH"], df["MN"]):
            mn = mn.split("||")
            try:
                entry = entry.split("||")
            except Exception:
                entry = []
            self.entry_ls.extend(entry)
            for m in mn:
                if m.startswith("C23.888"):
                    m = m.replace("C23.888.", "")
                    tree_ls.append(m)
        return tree_ls

    def num_of_roots(self):
        # Top-level concepts under the subtree.
        cnt = sum(1 for tree in self.tree_set_ls if len(tree) == 3)
        print("num_of_roots ", cnt)

    def num_of_classes(self):
        print("num_of_classes ", len(self.tree_set_ls))

    def num_of_synonyms(self):
        res = [i.split("|")[0] for i in self.entry_ls]
        print("num_of_synonyms ", len(set(res)))

    def num_of_leaves(self):
        parent_list = []
        self.leaves_list = []
        for tree in self.tree_set_ls:
            try:
                parent = tree[:-4]
            except Exception:
                parent = "aa"
            parent_list.append(parent)
        cnt = 0
        for tree in self.tree_set_ls:
            if tree not in parent_list:
                cnt += 1
                self.leaves_list.append(tree)
        print("num_of_leaves ", cnt)

    def max_depth(self):
        deepest = max(self.tree_set_ls, key=lambda x: len(x))
        print("max_depth ", len(deepest.split(".")))

    def avg_depth(self):
        # Average depth over leaf concepts only.
        cnt_list = [len(tree.split(".")) for tree in self.leaves_list]
        print("avg_depth ", np.mean(cnt_list))

    def avg_width(self):
        # Approximate mean number of nodes per depth level.
        width_dict = defaultdict(int)
        for tree in self.tree_set_ls:
            depth = len(tree.split("."))
            width_dict[depth] += 1
        avg_width = sum(width_dict.values()) / len(width_dict)
        print("avg_width ", avg_width)

    def fanout_ness_factor(self):
        parent2children_dict = defaultdict(list)
        for tree in self.tree_set_ls:
            parent = tree[:-4]
            if parent == "":
                continue
            parent2children_dict[parent].append(tree)
        cnt_list = [len(children) for children in parent2children_dict.values()]
        print("fanout_ness_factor ", np.mean(cnt_list))


if __name__ == "__main__":
    ResultStatistic()
