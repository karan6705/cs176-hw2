"""
CS 176 HW2, Question 5: Board-game recommendation network (GD Contest 2023, top-40 games).

Parts
  (a) fans_liked graph from the JSON records
  (b) node2vec embeddings drawn in 2D
  (c) bipartite affiliation graphs (game-designer, game-category, game-mechanic)
  (d) projected graphs of (c)
  (e) Problem: what actually drives "fans also liked", and can a hybrid
      recommender predict held-out fans_liked links?

Run:  python boardgames_analysis.py boardgames_40.json
Outputs go to ./figs and ./results.json
"""
import json
import os
import sys
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

SEED = 42
rng = np.random.default_rng(SEED)
DATA = sys.argv[1] if len(sys.argv) > 1 else "boardgames_40.json"
os.makedirs("figs", exist_ok=True)
results = {}

PALETTE = ["#2a78d6", "#e5793b", "#1d9e75", "#c2457a", "#7f77dd", "#8a8a85", "#b58b17", "#5aa9c9"]


def short(t, n=22):
    return t if len(t) <= n else t[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Load with pandas
# ---------------------------------------------------------------------------
df = pd.read_json(DATA)
df["rating_value"] = df["rating"].apply(lambda r: r["rating"])
df["num_reviews"] = df["rating"].apply(lambda r: r["num_of_reviews"])
df["fans_liked"] = df["recommendations"].apply(lambda r: r["fans_liked"])
df["categories"] = df["types"].apply(lambda t: [c["name"] for c in t["categories"]])
df["mechanics"] = df["types"].apply(lambda t: [m["name"] for m in t["mechanics"]])
df["designers"] = df["credit"].apply(lambda c: [d["name"] for d in c["designer"]])
df = df.sort_values("rank").reset_index(drop=True)
ids = set(df["id"])
title = dict(zip(df["id"], df["title"]))
rank = dict(zip(df["id"], df["rank"]))

results["n_games"] = len(df)
results["n_fans_liked_total"] = int(df["fans_liked"].apply(len).sum())

# ---------------------------------------------------------------------------
# (a) fans_liked graph
# ---------------------------------------------------------------------------
G = nx.DiGraph()
for _, r in df.iterrows():
    G.add_node(r["id"], title=r["title"], rank=r["rank"], rating=r["rating_value"])
for _, r in df.iterrows():
    for t in r["fans_liked"]:
        if t in ids and t != r["id"]:
            G.add_edge(r["id"], t)

U = G.to_undirected()
recip = sum(1 for u, v in G.edges() if G.has_edge(v, u)) // 2
results["a"] = {
    "directed_edges": G.number_of_edges(),
    "undirected_edges": U.number_of_edges(),
    "reciprocal_pairs": recip,
    "reciprocity": round(nx.reciprocity(G), 3),
    "density_directed": round(nx.density(G), 3),
    "avg_clustering_undirected": round(nx.average_clustering(U), 3),
    "isolated": [title[n] for n in G if G.degree(n) == 0],
}
indeg = sorted(G.in_degree(), key=lambda x: -x[1])
outdeg = sorted(G.out_degree(), key=lambda x: -x[1])
results["a"]["top_in_degree"] = [(title[n], d) for n, d in indeg[:6]]
results["a"]["lowest_in_degree"] = [(title[n], d) for n, d in indeg[-5:]]
results["a"]["top_out_degree"] = [(title[n], d) for n, d in outdeg[:5]]
pr = nx.pagerank(G, alpha=0.85)
results["a"]["top_pagerank"] = [(title[n], round(p, 4)) for n, p in sorted(pr.items(), key=lambda x: -x[1])[:6]]

comms = nx.community.louvain_communities(U, seed=SEED, resolution=1.0)
comms = sorted(comms, key=lambda c: -len(c))
comm_of = {n: i for i, c in enumerate(comms) for n in c}
results["a"]["communities"] = [[title[n] for n in sorted(c, key=lambda x: rank[x])] for c in comms]
results["a"]["modularity"] = round(nx.community.modularity(U, comms), 3)

# Community profile: most common mechanics / categories in each community
prof = []
for c in comms:
    sub = df[df["id"].isin(c)]
    mech = Counter(m for ms in sub["mechanics"] for m in ms).most_common(3)
    cat = Counter(m for ms in sub["categories"] for m in ms).most_common(3)
    prof.append({"size": len(c), "top_mechanics": mech, "top_categories": cat})
results["a"]["community_profiles"] = prof

pos = nx.spring_layout(U, seed=SEED, k=0.55, iterations=400)
fig, ax = plt.subplots(figsize=(15, 12))
mutual = [(u, v) for u, v in G.edges() if G.has_edge(v, u) and u < v]
oneway = [(u, v) for u, v in G.edges() if not G.has_edge(v, u)]
nx.draw_networkx_edges(G, pos, edgelist=oneway, ax=ax, edge_color="#9a9a95", alpha=0.45,
                       arrows=True, arrowsize=9, width=0.7, connectionstyle="arc3,rad=0.08",
                       node_size=[200 + 70 * G.in_degree(n) for n in G])
nx.draw_networkx_edges(U, pos, edgelist=mutual, ax=ax, edge_color="#3d3d3a", alpha=0.6, width=1.3)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[200 + 70 * G.in_degree(n) for n in G],
                       node_color=[PALETTE[comm_of[n] % len(PALETTE)] for n in G],
                       edgecolors="white", linewidths=1.2)
for n, (x, y) in pos.items():
    ax.text(x, y + 0.045, f"{rank[n]}. {short(title[n])}", fontsize=8.5, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
for i, c in enumerate(comms):
    ax.scatter([], [], c=PALETTE[i % len(PALETTE)], s=80, label=f"Community {i + 1} ({len(c)} games)")
ax.plot([], [], color="#3d3d3a", lw=1.3, label="Mutual recommendation")
ax.plot([], [], color="#9a9a95", lw=0.8, label="One-way recommendation (arrow)")
ax.legend(loc="lower left", fontsize=9, frameon=False)
ax.set_title("Top-40 BoardGameGeek games: 'fans also liked' graph\n"
             "node size = in-degree (how often other fans recommend it), color = Louvain community",
             fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.savefig("figs/a_fans_liked_graph.png", dpi=170)
plt.close()

# ---------------------------------------------------------------------------
# (b) node2vec embeddings
# ---------------------------------------------------------------------------
def node2vec_embed(graph, dim=32, walk_length=20, num_walks=200, p=1.0, q=0.5, seed=SEED):
    from node2vec import Node2Vec
    g = nx.relabel_nodes(graph, {n: str(n) for n in graph})
    n2v = Node2Vec(g, dimensions=dim, walk_length=walk_length, num_walks=num_walks, p=p, q=q,
                   workers=1, seed=seed, quiet=True)
    model = n2v.fit(window=5, min_count=1, batch_words=64, seed=seed, workers=1, epochs=5)
    return {int(k): model.wv[k] for k in g.nodes()}


emb = node2vec_embed(U)
order = list(df["id"])
X = np.array([emb[n] for n in order])
xy_tsne = TSNE(n_components=2, perplexity=8, random_state=SEED, init="pca").fit_transform(X)
xy_pca = PCA(n_components=2, random_state=SEED).fit_transform(X)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Nearest neighbours in embedding space for a few headline games
nn = {}
for g in [174430, 224517, 342942, 266192]:
    if g in emb:
        sims = sorted(((cos(emb[g], emb[o]), o) for o in order if o != g), reverse=True)[:4]
        nn[title[g]] = [(title[o], round(s, 3)) for s, o in sims]
results["b"] = {"dim": 32, "walk_length": 20, "num_walks": 200, "p": 1.0, "q": 0.5,
                "nearest_neighbours": nn}

# How well do embedding clusters agree with graph communities? (silhouette in 2D t-SNE space)
from sklearn.metrics import silhouette_score
results["b"]["silhouette_tsne_vs_louvain"] = round(float(
    silhouette_score(xy_tsne, [comm_of[n] for n in order])), 3)

fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))
for ax, xy, name in [(axes[0], xy_tsne, "t-SNE"), (axes[1], xy_pca, "PCA")]:
    ax.scatter(xy[:, 0], xy[:, 1], s=[90 + 25 * G.in_degree(n) for n in order],
               c=[PALETTE[comm_of[n] % len(PALETTE)] for n in order], edgecolors="white", linewidths=1)
    for (x, y), n in zip(xy, order):
        ax.annotate(f"{rank[n]}. {short(title[n], 20)}", (x, y), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_title(f"node2vec embeddings (32-d) projected to 2D with {name}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
for i, c in enumerate(comms):
    axes[0].scatter([], [], c=PALETTE[i % len(PALETTE)], s=60, label=f"Graph community {i + 1}")
axes[0].legend(fontsize=9, frameon=False, loc="best")
plt.suptitle("Color = Louvain community from part (a); games that sit close together are reached by similar random walks",
             fontsize=11)
plt.tight_layout()
plt.savefig("figs/b_node2vec_embeddings.png", dpi=160)
plt.close()

# ---------------------------------------------------------------------------
# (c) bipartite affiliation graphs
# ---------------------------------------------------------------------------
def build_bipartite(col):
    B = nx.Graph()
    for _, r in df.iterrows():
        B.add_node(r["id"], bipartite=0, label=r["title"])
        for f in r[col]:
            B.add_node(("F", f), bipartite=1, label=f)
            B.add_edge(r["id"], ("F", f))
    return B


bip = {k: build_bipartite(k) for k in ["designers", "categories", "mechanics"]}
results["c"] = {}
for k, B in bip.items():
    foci = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]
    shared = [n for n in foci if B.degree(n) >= 2]
    results["c"][k] = {
        "n_foci": len(foci),
        "n_shared_foci": len(shared),
        "edges": B.number_of_edges(),
        "largest_foci": sorted(((n[1], B.degree(n)) for n in foci), key=lambda x: -x[1])[:6],
    }


def draw_bipartite(B, fname, focus_name, min_deg):
    games = order
    foci = sorted([n for n, d in B.nodes(data=True) if d["bipartite"] == 1 and B.degree(n) >= min_deg],
                  key=lambda f: (-B.degree(f), f[1]))
    H = B.subgraph(games + foci)
    pos = {g: (0, -i) for i, g in enumerate(games)}
    span = len(games) - 1
    for j, f in enumerate(foci):
        pos[f] = (1, -j * span / max(1, len(foci) - 1))
    fig, ax = plt.subplots(figsize=(13, 15))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.35, edge_color="#7a7a75", width=0.8)
    nx.draw_networkx_nodes(H, pos, nodelist=games, node_color="#2a78d6", node_size=70, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=foci, node_color="#e5793b", node_shape="s",
                           node_size=[40 + 25 * B.degree(f) for f in foci], ax=ax)
    for g in games:
        ax.text(-0.02, pos[g][1], f"{rank[g]}. {short(title[g], 30)}", ha="right", va="center", fontsize=8.5)
    for f in foci:
        ax.text(1.02, pos[f][1], f"{f[1]} ({B.degree(f)})", ha="left", va="center", fontsize=8.5)
    ax.set_xlim(-0.75, 1.6)
    note = "" if min_deg <= 1 else f" (only {focus_name.lower()} shared by at least {min_deg} games are drawn)"
    ax.set_title(f"Affiliation graph: games (left) and {focus_name.lower()} (right){note}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


draw_bipartite(bip["designers"], "figs/c1_bipartite_designers.png", "Designers", 2)
draw_bipartite(bip["categories"], "figs/c2_bipartite_categories.png", "Categories", 2)
draw_bipartite(bip["mechanics"], "figs/c3_bipartite_mechanics.png", "Mechanics", 4)

# ---------------------------------------------------------------------------
# (d) projected graphs
# ---------------------------------------------------------------------------
proj = {k: bipartite.weighted_projected_graph(B, order) for k, B in bip.items()}
results["d"] = {}
for k, P in proj.items():
    w = [d["weight"] for _, _, d in P.edges(data=True)]
    overlap = sum(1 for u, v in P.edges() if U.has_edge(u, v))
    results["d"][k] = {
        "edges": P.number_of_edges(),
        "density": round(nx.density(P), 3),
        "max_shared": max(w) if w else 0,
        "isolated_games": sum(1 for n in P if P.degree(n) == 0),
        "edges_also_in_fans_liked": overlap,
        "share_of_projection_edges_that_are_fans_liked": round(overlap / max(1, P.number_of_edges()), 3),
        "components": nx.number_connected_components(P),
    }
results["d"]["designers"]["pairs"] = sorted(
    [(title[u], title[v], [f[1] for f in set(bip['designers'][u]) & set(bip['designers'][v])])
     for u, v in proj["designers"].edges()], key=lambda x: x[0])


def draw_projection(P, fname, ttl, min_w=1, layout_seed=SEED):
    E = [(u, v) for u, v, d in P.edges(data=True) if d["weight"] >= min_w]
    H = nx.Graph(); H.add_nodes_from(order); H.add_edges_from(E)
    pos = nx.spring_layout(H, seed=layout_seed, k=0.9 / np.sqrt(max(1, len(H))), iterations=500)
    fig, ax = plt.subplots(figsize=(14, 11))
    widths = [0.6 + 0.9 * (P[u][v]["weight"] - min_w) for u, v in E]
    colors = ["#1d9e75" if U.has_edge(u, v) else "#b4b2a9" for u, v in E]
    nx.draw_networkx_edges(H, pos, edgelist=E, width=widths, edge_color=colors, alpha=0.75, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_size=[160 + 40 * H.degree(n) for n in H],
                           node_color=[PALETTE[comm_of[n] % len(PALETTE)] for n in H],
                           edgecolors="white", ax=ax)
    for n, (x, y) in pos.items():
        ax.text(x, y + 0.03, f"{rank[n]}. {short(title[n])}", fontsize=8, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    ax.plot([], [], color="#1d9e75", lw=2, label="Pair is also linked in fans_liked")
    ax.plot([], [], color="#b4b2a9", lw=2, label="Pair is not linked in fans_liked")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(ttl, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def draw_designer_projection(P, fname):
    # Designer projection is a matching of 9 pairs plus 22 isolated games, so a spring layout
    # stacks each pair on top of itself. Lay the pairs out on a grid instead.
    E = sorted(P.edges(), key=lambda e: min(rank[e[0]], rank[e[1]]))
    iso = [n for n in order if P.degree(n) == 0]
    fig, ax = plt.subplots(figsize=(14, 8.5))
    for i, (u, v) in enumerate(E):
        if rank[u] > rank[v]:
            u, v = v, u
        row, col = divmod(i, 3)
        x0, y0 = col * 4.6, -row * 2.2
        shared = sorted(set(bip["designers"][u]) & set(bip["designers"][v]))
        color = "#1d9e75" if U.has_edge(u, v) else "#b4b2a9"
        ax.plot([x0, x0 + 2.6], [y0, y0], color=color, lw=1.5 + 2 * (len(shared) - 1), zorder=1)
        for x, n in [(x0, u), (x0 + 2.6, v)]:
            ax.scatter(x, y0, s=260, c=PALETTE[comm_of[n] % len(PALETTE)], edgecolors="white", zorder=2)
            ax.text(x, y0 + 0.35, f"{rank[n]}. {short(title[n], 24)}", ha="center", fontsize=9)
        ax.text(x0 + 1.3, y0 - 0.45, " and ".join(f[1] for f in shared),
                ha="center", fontsize=8.5, color="#5f5e5a", style="italic")
    ax.text(-0.6, -5.7, "Games with no designer in common with any other top-40 game (isolated in this projection):",
            fontsize=9.5, weight="bold")
    iso_txt = ", ".join(f"{rank[n]}. {title[n]}" for n in iso)
    import textwrap
    for j, line in enumerate(textwrap.wrap(iso_txt, 150)):
        ax.text(-0.6, -6.2 - 0.42 * j, line, fontsize=9)
    ax.plot([], [], color="#1d9e75", lw=2, label="Pair is also linked in fans_liked (all 9 are)")
    for i, c in enumerate(comms):
        ax.scatter([], [], c=PALETTE[i % len(PALETTE)], s=60, label=f"fans_liked community {i + 1}")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    ax.set_xlim(-1.4, 13.2); ax.set_ylim(-9.0, 1.0)
    ax.set_title("Projected graph: games linked when they share a designer (label under each edge = shared designer)\n"
                 "thicker edge = more shared designers", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


draw_designer_projection(proj["designers"], "figs/d1_projection_designers.png")
def draw_category_projection(P, fname):
    # Layout on the full weighted projection (connected, so nothing flies off), then draw
    # 1-shared-category edges faint and 2+ shared-category edges strong.
    D = {u: {} for u in P}
    for u, v, d in P.edges(data=True):
        D[u][v] = D[v][u] = 1.0 / d["weight"]
    pos = nx.kamada_kawai_layout(P, dist=dict(nx.all_pairs_dijkstra_path_length(nx.Graph(
        [(u, v, {"weight": 1.0 / d["weight"]}) for u, v, d in P.edges(data=True)]))))
    weak = [(u, v) for u, v, d in P.edges(data=True) if d["weight"] == 1]
    strong = [(u, v) for u, v, d in P.edges(data=True) if d["weight"] >= 2]
    fig, ax = plt.subplots(figsize=(15, 12))
    nx.draw_networkx_edges(P, pos, edgelist=weak, width=0.4, edge_color="#d3d1c7", alpha=0.6, ax=ax)
    nx.draw_networkx_edges(P, pos, edgelist=strong, ax=ax, alpha=0.85,
                           width=[1.2 + 1.2 * (P[u][v]["weight"] - 2) for u, v in strong],
                           edge_color=["#1d9e75" if U.has_edge(u, v) else "#8a8a85" for u, v in strong])
    nx.draw_networkx_nodes(P, pos, node_size=[150 + 10 * P.degree(n, weight="weight") for n in P],
                           node_color=[PALETTE[comm_of[n] % len(PALETTE)] for n in P], edgecolors="white", ax=ax)
    for n, (x, y) in pos.items():
        ax.text(x, y + 0.035, f"{rank[n]}. {short(title[n])}", fontsize=8, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    ax.plot([], [], color="#d3d1c7", lw=1, label="Share exactly 1 category")
    ax.plot([], [], color="#1d9e75", lw=2.4, label="Share 2+ categories and linked in fans_liked")
    ax.plot([], [], color="#8a8a85", lw=2.4, label="Share 2+ categories, not linked in fans_liked")
    for i, c in enumerate(comms):
        ax.scatter([], [], c=PALETTE[i % len(PALETTE)], s=60, label=f"fans_liked community {i + 1}")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title("Projected graph: games linked when they share a category\n"
                 "(edge width = number of shared categories; node color = fans_liked community from part a)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


draw_category_projection(proj["categories"], "figs/d2_projection_categories.png")

# Combined projection asked for in the prompt: same designer OR same category
combo = nx.Graph(); combo.add_nodes_from(order)
for u, v in combinations(order, 2):
    sd = proj["designers"].has_edge(u, v)
    sc = proj["categories"].has_edge(u, v)
    if sd or sc:
        combo.add_edge(u, v, weight=1, designer=sd, category=sc)
results["d"]["designer_or_category"] = {
    "edges": combo.number_of_edges(),
    "density": round(nx.density(combo), 3),
    "edges_also_in_fans_liked": sum(1 for u, v in combo.edges() if U.has_edge(u, v)),
}

# ---------------------------------------------------------------------------
# (e) Problem: what drives "fans also liked", and can we predict missing links?
# ---------------------------------------------------------------------------
def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


attr = df.set_index("id")
pairs = list(combinations(order, 2))


def pair_feats(u, v):
    a, b = attr.loc[u], attr.loc[v]
    return {
        "mech_jaccard": jacc(a["mechanics"], b["mechanics"]),
        "cat_jaccard": jacc(a["categories"], b["categories"]),
        "same_designer": float(bool(set(a["designers"]) & set(b["designers"]))),
        "coop_both": float("Cooperative Game" in a["mechanics"] and "Cooperative Game" in b["mechanics"]),
        "playtime_gap": abs(np.log((a["minplaytime"] + a["maxplaytime"]) / 2) -
                            np.log((b["minplaytime"] + b["maxplaytime"]) / 2)),
        "year_gap": abs(a["year"] - b["year"]),
        "weight_gap": abs(a["minage"] - b["minage"]),
    }


feat_df = pd.DataFrame([{**pair_feats(u, v), "u": u, "v": v, "linked": int(U.has_edge(u, v))} for u, v in pairs])
results["e"] = {"n_pairs": len(feat_df), "n_linked": int(feat_df["linked"].sum()),
                "base_rate": round(feat_df["linked"].mean(), 3)}

# E1. Homophily: link rate conditional on shared traits
hom = {}
hom["share_designer"] = (round(feat_df.query("same_designer==1")["linked"].mean(), 3),
                         int((feat_df.same_designer == 1).sum()))
hom["no_shared_designer"] = (round(feat_df.query("same_designer==0")["linked"].mean(), 3),
                             int((feat_df.same_designer == 0).sum()))
hom["both_cooperative"] = (round(feat_df.query("coop_both==1")["linked"].mean(), 3), int((feat_df.coop_both == 1).sum()))
q = feat_df["mech_jaccard"].quantile([0.5, 0.9]).values
hom["mech_jaccard_top10pct"] = (round(feat_df[feat_df.mech_jaccard >= q[1]]["linked"].mean(), 3),
                                int((feat_df.mech_jaccard >= q[1]).sum()))
hom["mech_jaccard_bottom50pct"] = (round(feat_df[feat_df.mech_jaccard <= q[0]]["linked"].mean(), 3),
                                   int((feat_df.mech_jaccard <= q[0]).sum()))
hom["cat_jaccard_zero"] = (round(feat_df[feat_df.cat_jaccard == 0]["linked"].mean(), 3),
                           int((feat_df.cat_jaccard == 0).sum()))
hom["cat_jaccard_positive"] = (round(feat_df[feat_df.cat_jaccard > 0]["linked"].mean(), 3),
                               int((feat_df.cat_jaccard > 0).sum()))
results["e"]["homophily_link_rate_and_count"] = hom

# E2. Link prediction with 5-fold edge hold-out. node2vec is retrained on each training graph
#     so held-out edges never leak into the embedding.
meta_cols = ["mech_jaccard", "cat_jaccard", "same_designer", "coop_both", "playtime_gap", "year_gap", "weight_gap"]
pos_pairs = feat_df[feat_df.linked == 1].reset_index(drop=True)
neg_pairs = feat_df[feat_df.linked == 0].reset_index(drop=True)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
neg_folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(neg_pairs))

scores = {k: [] for k in ["mech_jaccard", "cat_jaccard", "same_designer", "common_neighbors",
                          "adamic_adar", "node2vec_cosine", "metadata_LR", "hybrid_LR"]}
p_at_k = {k: [] for k in scores}
coefs = []


def topk_precision(test_df, score_col, k=3):
    hits, tot = 0, 0
    for g in order:
        rows = test_df[(test_df.u == g) | (test_df.v == g)]
        if rows["linked"].sum() == 0:
            continue
        top = rows.sort_values(score_col, ascending=False, kind="mergesort").head(k)
        hits += top["linked"].sum(); tot += k
    return hits / tot if tot else np.nan


for fold, ((tr_p, te_p), (tr_n, te_n)) in enumerate(zip(kf.split(pos_pairs), neg_folds)):
    train_pos, test_pos = pos_pairs.iloc[tr_p], pos_pairs.iloc[te_p]
    train_neg, test_neg = neg_pairs.iloc[tr_n], neg_pairs.iloc[te_n]
    Gtr = nx.Graph(); Gtr.add_nodes_from(order)
    Gtr.add_edges_from(zip(train_pos.u, train_pos.v))
    e_tr = node2vec_embed(Gtr, seed=SEED + fold)

    def struct(d):
        d = d.copy()
        d["common_neighbors"] = [len(list(nx.common_neighbors(Gtr, u, v))) for u, v in zip(d.u, d.v)]
        d["adamic_adar"] = [next(nx.adamic_adar_index(Gtr, [(u, v)]))[2] for u, v in zip(d.u, d.v)]
        d["node2vec_cosine"] = [cos(e_tr[u], e_tr[v]) for u, v in zip(d.u, d.v)]
        return d

    train = struct(pd.concat([train_pos, train_neg]))
    # shuffle so that ties in a score are broken randomly, not by "positives first" order
    test = struct(pd.concat([test_pos, test_neg])).sample(frac=1, random_state=SEED + fold)

    lr_meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(train[meta_cols], train.linked)
    test["metadata_LR"] = lr_meta.predict_proba(test[meta_cols])[:, 1]
    hyb_cols = meta_cols + ["common_neighbors", "adamic_adar", "node2vec_cosine"]
    mu, sd = train[hyb_cols].mean(), train[hyb_cols].std().replace(0, 1)
    lr_h = LogisticRegression(max_iter=2000, class_weight="balanced").fit((train[hyb_cols] - mu) / sd, train.linked)
    test["hybrid_LR"] = lr_h.predict_proba((test[hyb_cols] - mu) / sd)[:, 1]
    coefs.append(dict(zip(hyb_cols, lr_h.coef_[0])))

    for k in scores:
        scores[k].append(roc_auc_score(test.linked, test[k]))
        p_at_k[k].append(topk_precision(test, k, 3))

test_base = len(pos_pairs) / len(feat_df)
results["e"]["link_prediction_auc"] = {k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3)) for k, v in scores.items()}
results["e"]["precision_at_3"] = {k: round(float(np.nanmean(v)), 3) for k, v in p_at_k.items()}
results["e"]["random_precision_at_3"] = round(test_base, 3)
mean_coef = pd.DataFrame(coefs).mean().sort_values(ascending=False)
results["e"]["hybrid_standardized_coefficients"] = {k: round(float(v), 3) for k, v in mean_coef.items()}

# E3. Final recommender trained on the full graph: suggest unlinked pairs with the highest hybrid score
full = feat_df.copy()
full["common_neighbors"] = [len(list(nx.common_neighbors(U, u, v))) for u, v in zip(full.u, full.v)]
full["adamic_adar"] = [next(nx.adamic_adar_index(U, [(u, v)]))[2] for u, v in zip(full.u, full.v)]
full["node2vec_cosine"] = [cos(emb[u], emb[v]) for u, v in zip(full.u, full.v)]
hyb_cols = meta_cols + ["common_neighbors", "adamic_adar", "node2vec_cosine"]
mu, sd = full[hyb_cols].mean(), full[hyb_cols].std().replace(0, 1)
lr_full = LogisticRegression(max_iter=2000, class_weight="balanced").fit((full[hyb_cols] - mu) / sd, full.linked)
full["score"] = lr_full.predict_proba((full[hyb_cols] - mu) / sd)[:, 1]
miss = full[full.linked == 0].sort_values("score", ascending=False).head(10)
results["e"]["top_missing_links"] = [(title[u], title[v], round(s, 3)) for u, v, s in zip(miss.u, miss.v, miss.score)]

# Under-recommended games: high rating rank but low in-degree among the top 40
df["in_degree"] = df["id"].map(dict(G.in_degree()))
df["out_degree"] = df["id"].map(dict(G.out_degree()))
df["pagerank"] = df["id"].map(pr)
results["e"]["rank_vs_indegree_spearman"] = round(float(df[["rank", "in_degree"]].corr(method="spearman").iloc[0, 1]), 3)
results["e"]["rating_vs_indegree_spearman"] = round(float(df[["rating_value", "in_degree"]].corr(method="spearman").iloc[0, 1]), 3)
results["e"]["reviews_vs_indegree_spearman"] = round(float(df[["num_reviews", "in_degree"]].corr(method="spearman").iloc[0, 1]), 3)

# Figures for (e)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
labels = {"mech_jaccard": "Shared mechanics (Jaccard)", "cat_jaccard": "Shared categories (Jaccard)",
          "same_designer": "Same designer", "common_neighbors": "Common neighbours",
          "adamic_adar": "Adamic-Adar", "node2vec_cosine": "node2vec cosine",
          "metadata_LR": "Metadata model (LR)", "hybrid_LR": "Hybrid model (LR)",
          "coop_both": "Both cooperative", "playtime_gap": "Playtime gap (log)",
          "year_gap": "Release-year gap", "weight_gap": "Min-age gap"}
ks = sorted(scores, key=lambda k: np.mean(scores[k]))
means = [np.mean(scores[k]) for k in ks]
errs = [np.std(scores[k]) for k in ks]
cols = ["#2a78d6" if k in ("metadata_LR", "hybrid_LR") else "#b4b2a9" for k in ks]
axes[0].barh([labels[k] for k in ks], means, xerr=errs, color=cols, capsize=3)
axes[0].axvline(0.5, ls="--", c="#5f5e5a", lw=1)
axes[0].text(0.505, -0.6, "random = 0.5", fontsize=9, color="#5f5e5a")
axes[0].set_xlim(0.4, 1.0)
for i, (m, e) in enumerate(zip(means, errs)):
    axes[0].text(m + e + 0.012, i, f"{m:.3f}", va="center", fontsize=9)
axes[0].set_xlabel("ROC-AUC on held-out fans_liked links (5-fold, mean ± sd)")
axes[0].set_title("Which signal predicts a hidden 'fans also liked' link?")
axes[0].spines[["top", "right"]].set_visible(False)

c = mean_coef.sort_values()
axes[1].barh([labels.get(k, k.replace("_", " ")) for k in c.index], c.values,
             color=["#1d9e75" if v > 0 else "#e5793b" for v in c.values])
axes[1].axvline(0, c="#5f5e5a", lw=1)
axes[1].set_xlabel("Standardized logistic-regression coefficient (mean over folds)")
axes[1].set_title("What the hybrid model relies on (positive = makes a link more likely)")
axes[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figs/e1_link_prediction.png", dpi=160)
plt.close()

fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(df["rank"], df["in_degree"], s=np.sqrt(df["num_reviews"]) * 0.9, c="#2a78d6", alpha=0.6, edgecolors="white")
for _, r in df.iterrows():
    ax.annotate(short(r["title"], 18), (r["rank"], r["in_degree"]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("BoardGameGeek rank (1 = best)")
ax.set_ylabel("In-degree inside the top 40 (how many top-40 games' fans recommend it)")
ax.set_title("Being ranked highly is not the same as being recommended by other fans\n(bubble size = number of reviews)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figs/e2_rank_vs_indegree.png", dpi=160)
plt.close()

df[["rank", "title", "year", "rating_value", "num_reviews", "in_degree", "out_degree", "pagerank"]].to_csv("node_metrics.csv", index=False)
json.dump(results, open("results.json", "w"), indent=1, default=str)
print(json.dumps(results, indent=1, default=str))
