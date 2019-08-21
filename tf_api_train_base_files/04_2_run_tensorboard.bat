@rem 作成日2019/8/16 tensorboardで学習経過確認。http://localhost:6006 からtensorboard確認する

call activate tfgpu_v1-11
call tensorboard --logdir PATH_TO_BE_CONFIGURED/log_train
