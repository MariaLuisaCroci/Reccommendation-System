import pandas as pd
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, KFold, GridSearchCV

dataset1_path = r'Part_1\\dataset\\'

### part1.1

reader = Reader(line_format = 'user item rating', sep = ',', rating_scale = [0.5, 5], skip_lines = 1)
data = Dataset.load_from_file(dataset1_path + "ratings.csv",
                              reader)

# splitting dataset into k consecutive folds (without shuffling by default)
# out of the k folds, k-1 sets are used for training while the remaining set is used for testing. 
# n_splits (int) – number of folds
# random_state – determines the RNG that will be used for determining the folds

kf = KFold(n_splits = 5, random_state = 0)  

# defining the list of reccomendation algorithms
algo_list = ['SVD()', 'SlopeOne()', 'NMF()', 'NormalPredictor()', 'KNNBaseline()', 'KNNBasic()', 'KNNWithMeans()', 'KNNWithZScore()', 'BaselineOnly()', 'CoClustering()', 'SVDpp()']

# evaluating RMSE of algorithm in algo_list on 5 splits

benchmark = []

# iterating over all algorithms
for i in range(len(algo_list)):
    
    # perfoming cross validation
    results =    cross_validate(eval(algo_list[i]), data, measures = ['RMSE'], cv = kf, verbose = True, n_jobs = -1)
    
    # getting results & appending algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis = 0)
    tmp = tmp.append(pd.Series([algo_list[i].split(' ')[0].split('.')[-1]], index = ['Algorithm']))
    benchmark.append(tmp)

# results_pd is a dataframe containing test_rmse, fit_time and test_time for each algorithm    
results_pd = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 


### part1.2

# SVD grid of parameter for the GridSearch 

grid_of_parameters = {'n_factors' : [105, 110], 'n_epochs' : [20, 25, 30], 'lr_all' : [0.005, 0.009], 'reg_all' : [0.04, 0.06]}



gs = GridSearchCV(SVD,    # estimator parameter – the algorithm that you want to execute
                  param_grid = grid_of_parameters,    # param_grid – takes the parameter dictionary that we just created as parameter
                  measures = ['rmse'],    # scoring parameter – takes the performance metrics
                  cv = kf, n_jobs = -1) # when -1 n_jobs it use all processors

gs.fit(data)


# KNNBaseline grid of parameter for the GridSearch 

grid_of_parameters =  {'bsl_options' : {'method' : ['als'], 'n_epochs' : [20]},
              'sim_options' : {'name' : ['cosine', 'pearson', 'pearson_baseline'], 'min_support' : [1,5],  'user_based' : [False, True],
                            'k': [20, 40, 100]}
                      }

gs = GridSearchCV(KNNBaseline,
                  param_grid = grid_of_parameters,
                  measures = ['rmse'],
                  cv = kf, n_jobs = -1) # when -1 n_jobs it use all processors


gs.fit(data)
