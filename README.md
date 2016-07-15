# POLARA
Polara is the first recommendation framework that allows a deeper analysis of recommender systems performance, based on the idea of feedback polarity (by analogy with sentiment polarity in NLP).

In addition to standard question of "how good a recommender system is at recommending relevant items", it allows assessing the ability of a recommender system to **avoid irrelevant recommendations** (thus, less likely to disappoint a user). You can read more about this idea in a reasearch paper [Fifty Shades of Ratings: How to Benefit from a Negative Feedback in Top-N Recommendations Tasks](http://arxiv.org/abs/1607.04228). The research results can be easily reproduced with this framework, visit a "fixed state" version of the code at https://github.com/Evfro/fifty-shades (there're also many usage examples).

The framework also features efficient tensor-based implementation of an algorithm, proposed in the paper, that takes full advantage of the polarity-based formulation. Currently, there is an [online demo](http://coremodel.azurewebsites.net) (for test purposes only), that demonstrates the effect of taking into account feedback polarity. 

A special effort was made to make a *recsys for humans*, which stresses on the ease of use of the framework. For example, that's how you build a pure SVD recommender on top of the [Movielens 1M](http://grouplens.org/datasets/movielens/) dataset:

```python
from polara.recommender.data import RecommenderData
from polara.recommender.models import SVDModel
from polara.tools.movielens import get_movielens_data

ml_data = get_movielens_data(get_genres=False)
data_model = RecommenderData(ml_data, 'userid', 'movieid', 'rating')
svd = SVDModel(data_model)
svd.build()
svd.evaluate()
```
