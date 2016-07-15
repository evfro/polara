# POLARA
Polara is the first recommendation framework that allows a deeper analysis of recommender systems performance, based on the idea of feedback polarity (by analogy with sentiment polarity in NLP).

In addition to standard question of "how good a recommender system is at recommending relevant items", it allows assessing the ability of a recommender system to **avoid irrelevant recommendations** (thus, less likely to disappoint a user). You can read more about this idea in a reasearch paper [Fifty Shades of Ratings: How to Benefit from a Negative Feedback in Top-N Recommendations Tasks](http://arxiv.org/abs/1607.04228).

The framework also features efficient tensor-based implementation of an algorithm, proposed in the paper, that takes full advantage of the polarity-based formulation. Currently, there is an [online demo](http://coremodel.azurewebsites.net) (for test purposes only), that demonstrates the effect of taking into account feedback polarity.

A special effort was made to make a *recommender for humans*, which stresses on the ease of use of the framework.
