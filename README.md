
这是我的命令，运行之后就报错了，也不知道咋改了
 nohup python run_models.py --config_path configs/your_dataset_SAITS.ini > results/AUST_gait_prediction/train_output.out &
这是报错
2024-08-05 13:05:58,967 - args: Namespace(config_path='configs/your_dataset_SAITS.ini', result_saving_base_dir='results', model_name='AUST_gait_prediction', test_mode=False, log_saving='results/AUST_gait_prediction/logs', dataset_base_dir='generated_datasets', seq_len=11, batch_size=128, num_workers=4, feature_num=13, dataset_name='saits_input_data.npz', dataset_path='generated_datasets/saits_input_data.npz', eval_every_n_steps=60, MIT=True, ORT=True, lr=0.000669204734411692, optimizer_type='adam', weight_decay=0.0, device='cuda', epochs=100, early_stop_patience=30, model_saving_strategy='best', max_norm=0.0, imputation_loss_weight=1.0, reconstruction_loss_weight=1.0, model_type='Transformer', model_saving='results/AUST_gait_prediction/models/2024-08-05_T13:05:58')
2024-08-05 13:05:58,967 - Config file path: configs/your_dataset_SAITS.ini
2024-08-05 13:05:58,967 - Model name: AUST_gait_prediction
2024-08-05 13:05:58,972 - Num of total trainable params is: 668429
2024-08-05 13:05:59,083 - Creating adam optimizer...
2024-08-05 13:05:59,347 - Entering training mode...
2024-08-05 13:05:59,363 - train set len is 6879, batch size is 128, so each epoch has 54 steps
train_dataset shape: (6879, 11, 13)
train_mask shape: (6879, 11, 13)
val_dataset shape: (6879, 11, 13)
val_mask shape: (6879, 11, 13)
test_dataset shape: (6879, 11, 13)
test_mask shape: (6879, 11, 13)
Stage: train
Data length: 2
Item 0 shape: torch.Size([128, 11, 13])
Item 1 shape: torch.Size([128, 11, 13])
X shape: torch.Size([128, 11, 13])
masks shape: torch.Size([128, 11, 13])
Traceback (most recent call last):
  File "/home/lijiabao/SAITS/run_models.py", line 594, in <module>
    train(model, optimizer, train_dataloader, val_dataloader, tb_summary_writer, training_controller, logger, args)
  File "/home/lijiabao/SAITS/run_models.py", line 328, in train
    early_stopping = model_processing(
                     ^^^^^^^^^^^^^^^^^
  File "/home/lijiabao/SAITS/run_models.py", line 253, in model_processing
    results = result_processing(model(inputs, stage))
                                ^^^^^^^^^^^^^^^^^^^^
  File "/home/lijiabao/miniconda3/envs/SAITS-env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl

 
