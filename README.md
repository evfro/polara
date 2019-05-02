# POLARA
Polara is the first recommendation framework that allows a deeper analysis of recommender systems performance, based on the idea of feedback polarity (by analogy with sentiment polarity in NLP).

In addition to standard question of "how good a recommender system is at recommending relevant items", it allows assessing the ability of a recommender system to **avoid irrelevant recommendations** (thus, less likely to disappoint a user). You can read more about this idea in a research paper [Fifty Shades of Ratings: How to Benefit from a Negative Feedback in Top-N Recommendations Tasks](http://arxiv.org/abs/1607.04228). The research results can be easily reproduced with this framework, visit a "fixed state" version of the code at https://github.com/Evfro/fifty-shades (there're also many usage examples).
The framework also features efficient tensor-based implementation of an algorithm, proposed in the paper, that takes full advantage of the polarity-based formulation.


## Prerequisites
Current version of Polara supports both Python 2 and Python 3 environments. Future versions are likely to drop support of Python 2 to make a better use of Python 3 features.

The framework heavily depends on `Pandas, Numpy, Scipy` and `Numba` packages. Better performance can be achieved with `mkl` (optional). It's also recommended to use `jupyter notebook` for experimentation. Visualization of results can be done with help of `matplotlib`. The easiest way to get all those at once is to use the latest [Anaconda distribution](https://www.continuum.io/downloads).

If you use a separate `conda` environment for testing, the following command can be issued to ensure that all required dependencies are in place (see [this](http://conda.pydata.org/docs/commands/conda-install.html) for more info):

`conda install --file conda_req.txt`

Alternatively, a new conda environment with all required packages can be created by:

`conda create -n <your_environment_name> python=3.6 --file conda_req.txt`


## Installation
If you use specific `conda` environment, don't forget to activate it first with either `source activate <your_environment_name>` (Linux) or  `activate <your_environment_name>` (Windows). Clone this repository to your local machine (`git clone git://github.com/evfro/polara.git`). Once in the root of the newly created local repository, run

`python setup.py install`.


## Usage example
A special effort was made to make a *recsys for humans*, which stresses on the ease of use of the framework. For example, that's how you build a pure SVD recommender on top of the [Movielens 1M](http://grouplens.org/datasets/movielens/) dataset:

```python
from polara.recommender.data import RecommenderData
from polara.recommender.models import SVDModel
from polara.datasets.movielens import get_movielens_data
# get data and convert it into appropriate format
ml_data = get_movielens_data(get_genres=False)
data_model = RecommenderData(ml_data, 'userid', 'movieid', 'rating')
# build PureSVD model and evaluate it
svd = SVDModel(data_model)
svd.build()
svd.evaluate()
```
Several different scenarios and use cases, which cover many practical aspects, can also be found in the [examples directory](/examples).

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
        # exclude "self-links" and ensure only non-zero elements are stored
        i2i_matrix.setdiag(0)
        i2i_matrix.eliminate_zeros()
        # store matrix for generating recommendations
        self.i2i_matrix = i2i_matrix

    def get_recommendations(self):
        # get test users information and generate top-k recommendations
        test_matrix, test_data = self.get_test_matrix()
        # calculate predicted scores
        i2i_scores = test_matrix.dot(self.i2i_matrix)
        # prevent seen items from appearing in recommendations
        if self.filter_seen:
            self.downvote_seen_items(i2i_scores, test_data)
        # generate top-k recommendations for every test user
        top_recs = self.get_topk_elements(i2i_scores)
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
from polara.recommender.models import PopularityModel, RandomModel

# define models
i2i = CooccurrenceModel(data_model)
svd = SVDModel(data_model)
popular = PopularityModel(data_model)
random = RandomModel(data_model)
models = [i2i, svd, popular, random]

metrics = ['ranking', 'relevance'] # metrics for evaluation: NDGC, Precision, Recall, etc.
folds = [1, 2, 3, 4, 5] # use all 5 folds for cross-validation (default)
topk_values = [1, 5, 10, 20, 50] # values of k to experiment with

# run 5-fold CV experiment
result = ee.run_cv_experiment(models, folds, metrics,
                              fold_experiment=ee.topk_test,
                              topk_list=topk_values)

# calculate average values across all folds for e.g. relevance metrics
scores = result.mean(axis=0, level=['top-n', 'model']) # use .std instead of .mean for standard deviation
scores.xs('recall', level='metric', axis=1).unstack('model')
```
which results in something like:

| **model** | **MP** | **PureSVD** | **RND** | **item-to-item** |
| ---: |:---:|:---:|:---:|:---:|
| **top-n** |
| **1** |  0.017828 |  0.079428 |  0.000055 |  0.024673 |
| **5** |  0.086604 |  0.219408 |  0.001104 |  0.126013 |
| **10** |  0.138546 |  0.300658 |  0.001987 |  0.202134 |
| ... | ... | ... | ... | ... |

## Custom pipelines
Polara by default takes care of raw data and helps to organize full evaluation pipeline, that includes splitting data into training, test and evaluation datasets, performing cross-validation and gathering results. However, if you need more control on that workflow, you can easily implement your custom usage scenario for you own needs.

### Build models without evaluation
If you simply want to build a model on a provided data, then you only need to define a training set. This can be easily achieved with the help of `prepare_training_only` method (assuming you have a pandas dataframe named `train_data` with corresponding "user", "item" and "rating" columns):
```python
data_model = RecommenderData(train_data, 'user', 'item', 'rating')
data_model.prepare_training_only()
```
Now you are ready to build your models (as in examples above) and export them to whatever workflow you currently have.

### Warm-start and known-user scenarios
By default polara makes testset and trainset disjoint by users, which allows to evaluate models against *user warm-start*.
However in some situations (for example, when polara is used within a larger pipeline) you might want to implement strictly a *known user* scenario to assess the quality of your recommender system on the unseen (held-out) items for the known users. The change between these two scenarios as controlled by setting `data_model.warm_start` attribute to `True` or `False`. See [Warm-start and standard scenarios](examples/Warm-start and standard scenarios.ipynb) Jupyter notebook as an example.

### Externally provided test data
If you don't want polara to perform data splitting (for example, when your test data is already provided), you can use the `set_test_data` method of a `RecommenderData` instance. It has a number of input arguments that cover all major cases of externally provided data. For example, assuming that you have new users' preferences encoded in the `unseen_data` dataframe and the corresponding held-out preferences in the `holdout` dataframe, the following command allows to include them into the data model:  
```python
data_model.set_test_data(testset=unseen_data, holdout=holdout, warm_start=True)
```
Polara will automatically perform all required transformations to ensure correct functioning of the evaluation pipeline. To evaluate models you simply call standard methods without any modifications:
```python
svd.build()
svd.evaluate()
```
In this case the recommendations are generated based on the testset and evaluated against the holdout.
See more usage examples in the [Custom evaluation](examples/Custom evaluation.ipynb) notebook.

### Reproducing others work
Polara offers even more options to highly customize experimentation pipeline and tailor it to specific needs. See, for example, [Reproducing EIGENREC results](examples/Reproducing EIGENREC results.ipynb) notebook to learn how Polara can be used to reproduce experiments from the *"[EIGENREC: generalizing PureSVD for effective and efﬁcient top-N recommendations](https://arxiv.org/abs/1511.06033)"* paper.
