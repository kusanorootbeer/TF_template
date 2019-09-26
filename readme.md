# Tensorflow model training template

## idea

- [x] set logger
  - [x] git commit hash
  - [x] model parameter add filename and log
- [x] load dataset
  - [x] mnist
- [ ] visualize
  - [ ] tensorboard
- [ ] build model
  - [ ] VariationalAutoEncoder
- [x] model training batch file

## for demonstration

I will prepare VAE model for mnist dataset.

## my task

以下のように統一させたい

    input data shape [h,w,c]で固定
    input_shape = [-1, h, w, c]
    input_size = [h * w * c]
    data_shape = [h, w, c=3] or [h, w]

以下のように統一させたい

    input data shape [h * w * c]で固定
    input_size = [h * w * c]
    data_shape = [h, w, c=3] or [h, w]
