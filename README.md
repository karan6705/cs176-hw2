# CS 176 Homework 2, Question 5: Board Games

Analysis of the top-40 BoardGameGeek games (Graph Drawing Contest 2023 dataset).

## Files

- `boardgames_analysis.ipynb` notebook with all parts (a) to (e)
- `boardgames_analysis.py` same code as a script
- `boardgames_40.json` dataset
- `requirements.txt` Python packages
- `figs/` figures used in the report

## How to run

```
pip install -r requirements.txt
python boardgames_analysis.py boardgames_40.json
```

or open `boardgames_analysis.ipynb` and run all cells. Figures are saved to `figs/` and the numbers to `results.json`.

## Parts

- (a) fans_liked graph of the top-40 games
- (b) node2vec embeddings drawn in 2D (t-SNE and PCA)
- (c) bipartite affiliation graphs: games with designers and games with categories
- (d) projected graphs: games connected by a shared designer or a shared category
- (e) predicting missing "fans also liked" links using metadata and graph features
