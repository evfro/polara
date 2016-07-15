import sys

#INITIALIZATION
#properties that require rebuilding test data:
test_ratio = 0.2 #split 80% of users for training, 20% for test
test_fold = 5 #which fold to use for test data
shuffle_data = False #randomly permute all records in initial data
test_sample = None #sample a fraction of test data;  negative value will sample low-rated items

holdout_size = 3 #number of items hidden from each user
permute_tops = False #affects how top-rated items are sorted in test-set
random_holdout = False #sample evaluation items randomly instead of tops
negative_prediction = False #put negative feedback into evaluation set


#MODELS
switch_positive = 4 #ratings below are treated as negative feedback
predict_negative = False #predict low-rated items
#svd
svd_rank = 10
#coffee
mlrank = (13, 10, 2)
growth_tol = 0.0001
num_iters = 25
show_output = False

#RECOMMENDATIONS
topk = 10 #number of items to return
smooth_prediction = True #mix scores for positive feedback, e.g. 5 -> 4+5
filter_seen = True #prevent seen items from appearing in recommendations


#EVALUATION
not_rated_penalty = 0 #penalize predictions that were not rated by users
ndcg_alternative = True  #use exponential instead of linear relevance contribution in nDCG


#COMPUTATION
test_chunk_size = 1000 #to split tensor decompositions into smaller pieces in memory


def get_config(params):
    this = sys.modules[__name__]
    config = {param:getattr(this, param) for param in params}
    return config
