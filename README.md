# POLARA
Polara is the first recommendation framework that allows a deeper analysis of recommender systems performance, based on the idea of feedback polarity (by analogy with sentiment polarity in NLP).

In addition to standard question of "how good a recommender system is at recommending relevant items", it allows assessing the ability of a recommender system to **avoid irrelevant recommendations** (thus, less likely to disappoint a user). You can read more about this idea in a reasearch paper [Fifty Shades of Ratings: How to Benefit from a Negative Feedback in Top-N Recommendations Tasks](http://arxiv.org/abs/1607.04228). The research results can be easily reproduced with this framework, visit a "fixed state" version of the code at https://github.com/Evfro/fifty-shades (there're also many usage examples).

The framework also features efficient tensor-based implementation of an algorithm, proposed in the paper, that takes full advantage of the polarity-based formulation. Currently, there is an [online demo](http://coremodel.azurewebsites.net) (for test purposes only), that demonstrates the effect of taking into account feedback polarity.


## Prerequisites
**Note:** Currently runs on python 2 only (python 3 support in plans).

The framework heavily depends on `Pandas, Numpy, Scipy` and `Numba` packages. Better performance can be achieved with `mkl` (optional). It's also recommended to use `jupyter notebook` for experimentation. Visualization of results can be done with help of `matplotlib` and optionally `seaborn`. The easiest way to get all those at once is to use the latest [Anaconda distribution](https://www.continuum.io/downloads).

If you use a separate conda environment for testing, the following command can be used to ensure that all required dependencies are in place (see [this](http://conda.pydata.org/docs/commands/conda-install.html) for more info):

`conda install --file conda_req.txt`

Alternatively, a new conda environment with all required packages can be created by:

`conda create -n <your_environment_name> python=2.7 --file conda_req.txt`


## Installation
If you use specific conda environment, don't forget to activate it first with either `source activate <your_environment_name>` (Linux) or  `activate <your_environment_name>` (Windows). Clone this repository to your local machine (`git clone git://github.com/evfro/polara.git`). Once in the root of the newly created local repository, run

`python setup.py install`.


## Usage example
A special effort was made to make a *recsys for humans*, which stresses on the ease of use of the framework. For example, that's how you build a pure SVD recommender on top of the [Movielens 1M](http://grouplens.org/datasets/movielens/) dataset:

```python
from polara.recommender.data import RecommenderData
from polara.recommender.models import SVDModel
from polara.tools.movielens import get_movielens_data
# get data and convert it into appropriate format
ml_data = get_movielens_data(get_genres=False)
data_model = RecommenderData(ml_data, 'userid', 'movieid', 'rating')
# build PureSVD model and evaluate it
svd = SVDModel(data_model)
svd.build()
svd.evaluate()
```

## Creating new recommender models
Basic models can be extended by subclassing `RecommenderModel` class and defining two required methods: `self.build()` and `self.get_recommendations()`. Here's an example of a simple item-to-item recommender model:
```python
from polara.recommender.models import RecommenderModel

class CooccurrenceModel(RecommenderModel):
    def __init__(self, *args, **kwargs):
        super(CooccurrenceModel, self).__init__(*args, **kwargs)
        self.method = 'item-to-item' # pick some meaningful name

    def build(self):
        # build model - calculate item-to-item matrix
        user_item_matrix = self.get_training_matrix()
        # rating matrix product  R^T R  gives cooccurrences count
        i2i_matrix = user_item_matrix.T.dot(user_item_matrix) # gives CSC format
        i2i_matrix.setdiag(0) # exclude "self-links"
        i2i_matrix.eliminate_zeros() # ensure only non-zero elements are stored
        # store matrix for generating recommendations
        self._i2i_matrix = i2i_matrix

    def get_recommendations(self):
        # get test users information and generate top-k recommendations
        test_data, test_shape = self._get_test_data()
        test_matrix, _ = self.get_test_matrix(test_data, test_shape)
        # calculate predicted scores
        i2i_scores = test_matrix.dot(self._i2i_matrix)
        if self.filter_seen:
            # prevent seen items from appearing in recommendations
            self.downvote_seen_items(i2i_scores, test_data)
        # generate top-k recommendations for every test user
        top_recs = self.get_topk_items(i2i_scores)
        return top_recs
```
And the model is ready for evaluation:
```python
i2i = CooccurrenceModel(data_model)
i2i.build()
i2i.evaluate()
```

## Bulk experiments
Here's an example of how to perform **top-*k* recommendations** experiments with *5-fold cross-validation* for several models at once:

```python
from polara.evaluation import evaluation_engine as ee
from polara.recommender.models import NonPersonalized

# define models
i2i = CooccurrenceModel(data_model)
svd = SVDModel(data_model)
popular =  NonPersonalized('mostpopular', data_model)
random = NonPersonalized('random', data_model)
models = [i2i, svd, popular, random]

metrics = ['ranking', 'relevance'] # metrics for evaluation: NDGC, Precision, Recall, etc.
folds = [1, 2, 3, 4, 5] # use all 5 folds for cross-validation
topk_values = [1, 5, 10, 20, 50] # values of k to experiment with

# run experiment
topk_result = {}
for fold in folds:
    data_model.test_fold = fold
    topk_result[fold] = ee.topk_test(models, topk_list=topk_values, metrics=metrics)

# rearrange results into a more friendly representation
# this is just a dictionary of Pandas Dataframes
result = ee.consolidate_folds(topk_result, folds, metrics)
result.keys() # outputs ['ranking', 'relevance']

# calculate average values across all folds for e.g. relevance metrics
result['relevance'].mean(axis=0).unstack() # use .std instead of .mean for standard deviation
```
which results in something like:

| metric/model |item-to-item | SVD | mostpopular | random |
| ---: |:---:|:---:|:---:|:---:|
| *precision* | 0.348212 | 0.600066 | 0.411126 | 0.016159 |
| *recall*    | 0.147969 | 0.304338 | 0.182472 | 0.005486 |
| *miss_rate* | 0.852031 | 0.695662 | 0.817528 | 0.994514 |
| ... | ... | ... | ... | ... |

## Custom pipelines
Polara by default takes care of raw data and helps to organize full evaluation pipeline, that includes splitting data into training, test and evaluation datasets, performing cross-validation and gathering results. However, if you need more control on that workflow, you can easily implement your custom  usage scenario for you own needs.

### Build models without evaluation
If you simply want to build a model on pre-processed data, then you only need to define a training set. This can be easily achived with the following lines of code (assuming you have a pandas dataframe named `train_data` with corresponding "user", "item" and "rating" columns):
```python
data_model = RecommenderData(train_data, 'user', 'item', 'rating')
# mind underscores for training and test attributes,
# this ensures no automated data-processing triggers are activated
data_model._training = data_model._data
data_model._test = None
```
Now you are ready to build your models (as in examples above) and export them to whatever workflow you currently have.

### Cold-start and known-user scenarios
By default polara makes testset and trainset disjoint by users, which allows to evaluate models against both *user cold-start* and more common *known user* scenarios (see the paper mentioned above for explanation).
However in some situations (for example, when polara is used within a larger pipeline) you might want to implement strictly a *known user* scenario to assess the quality of your recommender system on the unseen (held-out) items for those known users. In that case your test users are exactly the same as training users. Assuming that you have prepared both training (`train_data` dataframe) and evaluation data (`test_data` dataframe), the goal can be achieved with the following code:
```python
from collections import namedtuple

data_model = RecommenderData(train_data, 'user', 'item', 'rating')
data_model._training = data_model._data
testset = data_model._data
holdout = test_data
#the number of held-out items in evaluation set must be constant
data_model._holdout_size = #<number of held-out items in evaluation set>
data_model._test = namedtuple('TestData', 'testset holdout')._make([testset, holdout])
```
Now you can build recommender models and evaluate them within polara framework.
