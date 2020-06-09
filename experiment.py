import sys
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.model_selection import KFold

import torch

np.random.seed(0)
# set to false if no GPU
CUDA = False

EXPERIMENT_TYPE=sys.argv[1]

# algorithms evaluated
algos = ['rf','lreg','our','rcn']
algo_ids = {algos[i]:i for i in range(len(algos))}

# experiment parameters
if EXPERIMENT_TYPE == 'real':
    NUM_TRIALS = 5
    etas = np.linspace(0,0.4,5)
    epss = np.linspace(0.05,0.2,4)
    NUM_ITERS = 2000
    NUM_FLIPS = 5
elif EXPERIMENT_TYPE in ['synthetic-rcn','synthetic-massart']:
    NUM_TRIALS = 50
    etas = np.linspace(0.05,0.45,9)
    epss = [0.05]
    NUM_ITERS = 2000
    NUM_FLIPS = 1
else:
    raise("invalid experiment type")

# shape of output arrays
data_shape = (len(algos),NUM_TRIALS,len(etas))

if EXPERIMENT_TYPE == 'real':
# load adult dataset
    # df = pd.read_csv("../input/uci-adult/adult.csv",header=0)
    df = pd.read_csv("adult.csv",header=0)
    df["income"] = df["income"].map({ "<=50K": -1, ">50K": 1 })
    df.age = df.age.astype(float)
    df = df.drop("fnlwgt",axis=1) #stands for "final weight", it's for importance sampling
    df["educational-num"] = df["educational-num"].astype(float)
    df["hours-per-week"] = df["hours-per-week"].astype(float)
    df_onehot = pd.get_dummies(df, columns=[
        "workclass", "education", "marital-status", "occupation", "relationship",
        "race", "gender", "native-country" ])
    df_onehot.drop("income", axis=1, inplace=True) # remove response variable
    # normalization step
    df_onehot_normalized = (df_onehot-df_onehot.mean())/df_onehot.std()
    y_all = df["income"].values

# synthetic experiment input distribution
# d: dimension of instance
# N: number of points in train set
# frac: 0.25*N is number of points in test set
def mixture_gauss(d,N,frac=0.25):
    total = int(N*(frac+1))
    cov1 = np.eye(d)
    cov2 = np.eye(d)
    cov2[0,0] = 8.
    cov2[0,1] = 0.1
    cov2[1,0] = 0.1
    cov2[1,1] = 0.0024
    vecs = np.zeros((total,d))
    for i in range(total):
        if np.random.uniform() > 0.5:
            vecs[i,:] = np.random.multivariate_normal([0]*d,cov1)
        else:
            vecs[i,:] = np.random.multivariate_normal([0]*d,cov2)
    x_train = vecs[:N,:]
    x_test = vecs[N:,:]
    y_train = (vecs[:N,1]>0).astype(int)*2 - 1
    y_test = (vecs[N:,1]>0).astype(int)*2 - 1
    return x_train, x_test, y_train, y_test

# train-test split (only for real data)
# returns split data as well as subsets of data corresponding to target/complement groups
# complement: true if targeting complement of the target feature
# targ_id: row index for target feature
# xs, ys: data
# train_index: id's of rows of trainset with target feature
# test_index: id's of rows of testset with target feature
def split_data(complement,targ_id,xs,ys,train_index,test_index):
    # make sure halfspace has constant input
    def add_bias(x):
        return np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
    x_train = xs[train_index]
    x_test = xs[test_index]
    y_train = ys[train_index]
    y_test = ys[test_index]

    if EXPERIMENT_TYPE != 'real':
        raise("split data only supported for real-data experiment")

    # we want to ensure that targeted high-income
    # individuals are classified accurately
    if complement:
        targ_group = (x_test[:,targ_id] < 0) & (y_test > 0)
        comp_group = (x_test[:,targ_id] > 0) & (y_test > 0)
    else:
        targ_group = (x_test[:,targ_id] > 0) & (y_test > 0)
        comp_group = (x_test[:,targ_id] < 0) & (y_test > 0)
    targ_x_test = x_test[targ_group,:]
    targ_y_test = y_test[targ_group]
    comp_x_test = x_test[comp_group,:]
    comp_y_test = y_test[comp_group]

    x_train = add_bias(x_train) 
    x_test = add_bias(x_test)
    targ_x_test = add_bias(targ_x_test)
    comp_x_test = add_bias(comp_x_test)

    return (x_train, x_test, y_train, y_test, 
                targ_group, targ_x_test, targ_y_test,
                comp_group, comp_x_test, comp_y_test)

# acc of model on test data x,y
def score(model,xs,ys):
    predictions = model.predict(xs)
    score = model.score(xs,ys)
    return score

# run logistic regression on training data x,y
def logistic(xs,ys,factor = 50.):
    N,d = xs.shape
    # defaulting to standard l2 penalty, which matches the penalty
    # we are using for filtertron right now.
    lreg = LogisticRegression(C=factor / N, max_iter = 200,
                             penalty='l2', solver='liblinear', 
                             fit_intercept=False,
                             tol=0.1)
    return lreg

# acc of halfspace w on points x,y
def acc(x,y,w):
    return np.sum(np.matmul(x,w)*y > 0)/float(len(y))

def leakyrelu(lam,z):
    if z > 0:
        return (1 - lam)*z
    else:
        return lam * z

# leakyrelu loss of w on points x,y
def L(x,y,lam,w):
    prods = -np.matmul(x,w)*y
    return np.average(0.5*prods + (0.5 - lam)*np.abs(prods))

# our algorithm FilterTron run on training data x,y
# T: number of iterations
# lam: leakage
# eps: error parameter (corresponding to lbd on slab mass)
# num_slabs: number of slabs to search over (heuristic to make FilterTron run faster)
# base_stepsize: learning rate
# filter: True if we run the filter, False if we just run gradient descent on LeakyRelu
def our_algo(x,y,T,lam,eps,num_slabs=10,base_stepsize=0.01,filter=True):
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    # random initialization
    N,d = x.shape
    init = np.random.multivariate_normal([0]*d,np.eye(d))
    init /= np.linalg.norm(init)
    w = torch.tensor(init).float()
    w = w/torch.norm(w)

    if CUDA:
        x = x.cuda()
        y = y.cuda()
        w = w.cuda()

    for j in range(T):
        stepsize = base_stepsize
        best_r_loss = -np.inf
        best_in_slab = range(N)
        best_r = None
        # find slab (of mass at least eps) 
        # maximizing leakyrelu loss over that slab 
        if filter:
            prods = -torch.matmul(x,w) * y
            abs_projects = torch.abs(prods)
            losses = 0.5 * prods + (0.5 - lam) * torch.abs(prods)

            sorted_idx = torch.argsort(abs_projects)

            for t in range(int(eps * num_slabs), num_slabs):
                max_idx = int((t * 1.0/num_slabs) * len(abs_projects))
                r = abs_projects[sorted_idx[max_idx]]
                in_slab = sorted_idx[:max_idx]
                loss = torch.mean(losses[in_slab])
                if loss > best_r_loss:
                    best_r_loss = loss
                    best_in_slab = in_slab
                    best_r = r
        else:
            prods_in_slab = -torch.matmul(x,w) * y
            best_r_loss = torch.mean(0.5 * prods_in_slab + (0.5 - lam) * torch.abs(prods_in_slab))

        # restrict to best slab
        if filter:
            x_in_slab = x[best_in_slab,:]
            y_in_slab = y[best_in_slab]
        else:
            x_in_slab = x
            y_in_slab = y

        terms = torch.matmul(x_in_slab,w) * y_in_slab
        factors = (terms > 0) * lam + (terms < 0) * (1.0 - lam)
        factors = factors.float()
        grad = torch.matmul(factors * y_in_slab, x_in_slab)
        grad /= len(y_in_slab)

        # take a gradient step
        w += stepsize*grad
        w /= max(1,torch.norm(w)) # ball version
        #w /= torch.norm(w) # spherical version
    w = w.cpu().numpy()
    return w

def censorfields(feature):
        # figure out which fields to censor if necessary
    if feature == "race_Black":
        vals = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
        demfields = [df_onehot.columns.get_loc('race_%s' % val) for val in vals]
    elif feature == 'gender_Female':
        vals = ['Male', 'Female']
        demfields = [df_onehot.columns.get_loc('gender_%s' % val) for val in vals]
    elif feature == 'native-country_United-States':
        vals = ['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']
        demfields = [df_onehot.columns.get_loc('native-country_%s' % val) for val in vals]
    else:
        raise("target feature not supported")
    return demfields

def experiment(CENSOR, TARGET_FEATURE):
    if not TARGET_FEATURE:
        COMPLEMENT = None
    elif TARGET_FEATURE == 'native-country_United-States':
        # target people whose native country is not the US
        COMPLEMENT = True
        # target the people with the target feature
    else:
        COMPLEMENT = False

    overall = np.zeros(data_shape)

    if EXPERIMENT_TYPE == 'real':
        idx = df_onehot.columns.get_loc(TARGET_FEATURE)
        target = np.zeros(data_shape)
        complement = np.zeros(data_shape)

    # define the Massart adversary for this target feature
    def corrupt(eta,xs,ys):
        def flip(p,x):
            if np.random.random() < p:
                return -x
            else:
                return x
        noisy_ys = np.array(ys)
        if EXPERIMENT_TYPE == 'real':
            for i in range(len(ys)):
                if COMPLEMENT:
                    if xs[i][idx] > 0:
                        noisy_ys[i] = flip(eta,ys[i])
                else:
                    if xs[i][idx] < 0:
                        noisy_ys[i] = flip(eta,ys[i])
            return noisy_ys
        elif EXPERIMENT_TYPE == 'synthetic-rcn':
            for i in range(len(ys)):
                noisy_ys[i] = flip(eta,ys[i])
            return noisy_ys
        elif EXPERIMENT_TYPE == 'synthetic-massart':
            for i in range(len(ys)):
                if xs[i,1]  > 0.3:
                    noisy_ys[i] = flip(eta,ys[i])
            return noisy_ys
        else:
            raise("invalid experiment type")

    # algo: 'our', 'rcn', 'lreg', 'rf'
    # train_and_test_set contains the non-noised train/test data
    # outputs average overall accuracy and average target group accuracy
    def run(algo,train_and_test_set,eta):
        if EXPERIMENT_TYPE == 'real':
            (x_train, x_test, y_train, y_test, 
                targ_group, targ_x_test, targ_y_test, 
                comp_group, comp_x_test, comp_y_test) = train_and_test_set
        else:
            (x_train, x_test, y_train, y_test) = train_and_test_set

        _,dim = x_train.shape

        # if CENSOR and running real-data experiment, hide target fields
        if EXPERIMENT_TYPE != 'real' and CENSOR:
            demfields = censorfields(TARGET_FEATURE)
            train_mask = np.ones(dim, dtype=bool)
            train_mask[demfields] = False
            test_mask = np.ones(dim, dtype=bool)
            test_mask[demfields] = False
            final_x_train = x_train[:,train_mask]
            final_x_test = x_test[:,test_mask]
            final_targ_x_test = targ_x_test[:,test_mask]
            final_comp_x_test = comp_x_test[:,test_mask]
        else:
            final_x_train = x_train
            final_x_test = x_test
            if EXPERIMENT_TYPE == 'real':
                final_targ_x_test = targ_x_test
                final_comp_x_test = comp_x_test

        over_accs = [0]*NUM_FLIPS
        targ_accs = [0]*NUM_FLIPS
        comp_accs = [0]*NUM_FLIPS

        for i in range(NUM_FLIPS):
            print("FLIP #" + str(i))
            y_train_noisy = corrupt(eta,x_train,y_train)
            y_test_noisy = corrupt(eta,x_test,y_test)

            if EXPERIMENT_TYPE == 'real':
                targ_y_test_noisy = y_test_noisy[targ_group]
                comp_y_test_noisy = y_test_noisy[comp_group]

            if algo in ['lreg','rf']:
                if algo == 'lreg':
                    clf = logistic(final_x_train,y_train_noisy)
                else:
                    clf = RandomForestClassifier(max_depth=20, random_state=0)
                clf.fit(final_x_train, y_train_noisy)
                over_acc = score(clf, final_x_test, y_test_noisy)
                if EXPERIMENT_TYPE == 'real':
                    targ_acc = score(clf, final_targ_x_test, targ_y_test_noisy)
                    comp_acc = score(clf, final_comp_x_test, comp_y_test_noisy)
            else:
                if algo == 'our':
                    filt = True
                elif algo == 'rcn':
                    filt = False
                else:
                    raise("Algorithm not supported")

                # crude grid search over parameters
                best_acc = -np.inf
                best_w = None
                for eps in epss:
                    lamb = eta + eps
                    wk = our_algo(final_x_train,y_train_noisy,NUM_ITERS,lamb,eps,
                                           filter=filt,base_stepsize=0.05)
                    acck = acc(x_test,y_test,wk)
                    best_w = None
                    
                    if best_acc < acck:
                        best_w = wk

                    over_acc = best_acc
                    if EXPERIMENT_TYPE == 'real':
                        targ_acc = acc(targ_x_test, targ_y_test,best_w)
                        comp_acc = acc(comp_x_test, comp_y_test,best_w)

            over_accs[i] = over_acc
            if EXPERIMENT_TYPE == 'real':
                targ_accs[i] = targ_acc
                comp_accs[i] = comp_acc
        if EXPERIMENT_TYPE == 'real':
            return np.average(over_accs), np.average(targ_accs), np.average(comp_accs)
        else:
            return np.average(over_accs)

    # driver function for real data experiment
    def real_driver():
        data_x = df_onehot_normalized.to_numpy()
        data_y = y_all
        kf = KFold(n_splits=NUM_TRIALS,shuffle=True,random_state=0)
        trial = 0
        for train_index, test_index in kf.split(data_x):
            print("TRIAL NUMBER: %d" % trial)
            train_and_test_set = split_data(COMPLEMENT,idx,data_x,data_y,train_index,test_index)
            for eta_i,eta in enumerate(etas):
                print(eta)
                for algo in algos:
                    print(algo)
                    over_acc, targ_acc, comp_acc = run(algo,train_and_test_set,eta)
                    overall[algo_ids[algo],trial,eta_i] = over_acc
                    target[algo_ids[algo],trial,eta_i] = targ_acc
                    complement[algo_ids[algo],trial,eta_i] = comp_acc
            trial += 1

        if CENSOR:
            folder = 'real/censor/'
        else:
            folder = 'real/noncensor/'
    
        with open(folder + 'overall%s.json'% TARGET_FEATURE, 'w') as f:
            json.dump(overall.tolist(),f)    
        with open(folder + 'target%s.json'% TARGET_FEATURE, 'w') as f:
            json.dump(target.tolist(),f)
        with open(folder + 'complement%s.json'% TARGET_FEATURE, 'w') as f:
            json.dump(complement.tolist(),f)

    # driver function for synthetic experiment
    def synth_driver():
        for trial in range(NUM_TRIALS):
            print("TRIAL NUMBER: %d" % trial)
            # create train and test set
            train_and_test_set = mixture_gauss(2,1000)
            for eta_i,eta in enumerate(etas):
                print(eta)
                for algo in algos:
                    print(algo)
                    over_acc = run(algo,train_and_test_set,eta)
                    overall[algo_ids[algo],trial,eta_i] = over_acc
    
        if EXPERIMENT_TYPE == 'synthetic-rcn':
            filename = 'gauss/temp_rcn.json'
        elif EXPERIMENT_TYPE == 'synthetic-massart':
            filename = 'gauss/temp_massart.json'
        else:
            raise("no such synthetic experiment")
        
        with open(filename, 'w') as f:
            json.dump(overall.tolist(),f)

    if EXPERIMENT_TYPE == 'real':
        real_driver()
    elif EXPERIMENT_TYPE in ['synthetic-rcn','synthetic-massart']:
        synth_driver()
    else:
        raise("invalid experiment type")

if EXPERIMENT_TYPE == 'real':
    experiment(True, 'race_Black')
    experiment(False, 'race_Black')
    experiment(True, 'gender_Female')
    experiment(False, 'gender_Female')
    experiment(True, 'native-country_United-States')
    experiment(False, 'native-country_United-States')
elif EXPERIMENT_TYPE == 'synthetic-rcn':
    experiment(None,None)
elif EXPERIMENT_TYPE == 'synthetic-massart':
    experiment(None,None)
else:
    raise("invalid experiment type")