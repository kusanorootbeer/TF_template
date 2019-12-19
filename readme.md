# Tensorflow model template

## progress

- [x] set logger
  - [x] git commit hash
  - [x] model parameter add filename and log
- [x] load dataset
  - [x] mnist
- [ ] visualize
  - [ ] tensorboard
- [x] build model
  - [x] VariationalAutoEncoder
  - [x] ConditionalVariationalAutoEncoder
    - [ ] Discrimination
  - [x] MLP Classifier
- [x] model training batch file
- [x] parameter search list up

## for demonstration

    python run.py

## Result

### compare between classifier model and generative model

Accuracy Table

| _model_        | _data restriction_ <br>None | _data restriction_ <br>1000 | _data restriction_ <br>100 | _data restriction_ <br>20 | _data restriction_ <br>10 |
| -------------- | --------------------------- | --------------------------- | -------------------------- | ------------------------- | ------------------------- |
| MLP Classifier | 0.9759                      | 0.9594                      | 0.8827                     | 0.7526                    | 0.6613                    |
| CVAE           |                             |                             |                            |                           |                           |

- data restriction: Number of available training data on each label
  - None: No restriction
  - Number of training data = <img src="https://latex.codecogs.com/gif.latex?n&space;\times" /> num of labels
