# An Information-Theoretic Framework for Unifying Active Learning Problems

## Prerequisites
The dependencies include:

* python:     3.6.5
* gpy:        1.9.8
* gpflow:     1.4.1
* tensorflow: 1.14.0
* tensorflow-probability


## Using The Code

### Level Set Estimation
```
python levelsetestimation.py --function $FUNCTION --criterion $CRITERION --level 0 --numqueries $NQUERIES --numruns 30  --numhyps 1 --noisevar $NOISEVAR  --nparal 1 --ntrain 500 --nysample 5000 --ninit 2
```
where `$FUNCTION` are from functions.py; `$CRITERION` can be `bes`, `straddle`, `dare`; `$NOISEVAR` can be `0.0001`, `0.09`; `$NQUERIES` can be `100`, `200`.


### Bayesian Optimization
```
python bayesianoptimization.py --function $FUNCTION --criterion $CRITERION --numqueries $NQUERIES --numruns 15 --numhyps 1 --noisevar $NOISEVAR --nmax 5 --nfeature 300 --nparal 2 --nsto 10 --ntrain 500 --nysample 3000 --ninit 2
```
where `$FUNCTION` are from functions.py; `$CRITERION` can be `avg_bes_mp` (BES-MP), `pes`, `ucb`, `ei`, `mes`; `$NOISEVAR` can be `0.0001`, `0.09`; `$NQUERIES` can be `100`, `200`.


### Implicit Level Set Estimation
```
python implicitlse.py --function $FUNCTION --criterion $CRITERION --alpha 0.2 --numqueries 200 --numruns 30 --numhyps 1 --nmax 5 --nfeature 300 --noisevar 0.0001 --nparal 1 --ntrain 500 --nysample 3000 --ninit 2
```
where `$FUNCTION` are from functions.py; `$CRITERION` can be `mnes2` (BES-MP), `mnes3` (BES^2-MP).

