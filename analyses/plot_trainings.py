import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data_path = 'TrainResult22kHz_4subs1518/subj15-16-17-18/imagined_speech/logs/metrics.csv'
path_parts = os.path.normpath(data_path).split(os.sep)
output_prefix = path_parts[0] if len(path_parts) > 1 else os.path.splitext(os.path.basename(data_path))[0]

df = pd.read_csv(data_path)

print(df.head())
print(df.shape)
#print(f"Max epoch: {df['epoch'].max()}")
#print(f"Unique epochs: {df['epoch'].nunique()}")

epochs = df['epoch']
#train_loss = df['train_loss']
#val_loss = df['val_loss']

train_lossG = df['train_loss_g']
val_lossG = df['val_loss_g']

train_loss_rmseG = df['train_loss_g_recon']
val_loss_rmseG = df['val_loss_g_recon']

ctc_train_loss = df['train_loss_g_ctc']
ctc_val_loss = df['val_loss_g_ctc']

validLoss_trainG = df['train_loss_g_valid']
validLoss_valG = df['val_loss_g_valid']

train_cer_gt = df['train_cer_gt']
train_cer_recon = df['train_cer_recon']

train_loss_d,train_acc_d_real,train_acc_d_fake = df['train_loss_d'],df['train_acc_d_real'],df['train_acc_d_fake']
val_loss_d,val_acc_d_real,val_acc_d_fake = df['val_loss_d'],df['val_acc_d_real'],df['val_acc_d_fake']

epoch_times = df['epoch_time_sec']
sum_epoch_times = sum(epoch_times)
print(f"Total training time: {sum_epoch_times/3600:.2f} hours")

# Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_loss, label='Train Loss', color='blue')
# plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
# plt.xlabel('Epochs')
# plt.ylabel('Loss Transformer')
# plt.title('Training and Validation Losses over Epochs (Tranformer)')
# plt.legend()
# plt.savefig('training_losses_eeg2melTransfomer_riem_16.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs,train_loss_rmseG, label='Train', color='blue')
plt.plot(epochs, val_loss_rmseG, label='Validation', color='orange')
plt.xlabel('Epochs')
plt.ylabel('RMSE (Generator 22kHz)')
plt.title('RMSE Losses over Epochs (Generator 22kHz)')
plt.legend()
plt.savefig(f'{output_prefix}_rmse_loss_Generator.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs,validLoss_trainG, label='Train', color='blue')
plt.plot(epochs, validLoss_valG, label='Validation', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Adversarial loss (Generator 22kHz)')
plt.title('Train and Validation Adversarial losses (Generator 22kHz)')
plt.legend()
plt.savefig(f'{output_prefix}_valid_loss_Generator.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, ctc_train_loss, label='Train', color='blue')
plt.plot(epochs, ctc_val_loss, label='Validation', color='orange')
plt.xlabel('Epochs')
plt.ylabel('CTC Loss Generator')
plt.title('CTC Loss over Epochs (Generator 22kHz)')
plt.legend()
plt.savefig(f'{output_prefix}_ctc_loss_Generator.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_lossG, label='Train', color='blue')
plt.plot(epochs, val_lossG, label='Validation', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Losses Generator')
plt.title('Cumulative Loss over Epochs (Generator 22kHz)')
plt.legend()
plt.savefig(f'{output_prefix}_training_losses_Generator.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_d, label='Train', color='blue')
plt.plot(epochs, val_loss_d, label='Validation', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Losses Discriminator')
plt.title('Cumulative Loss over Epochs (Discriminator 22kHz)')
plt.legend()
plt.savefig(f'{output_prefix}_training_losses_Discriminator.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc_d_real, label='Train Real', color='blue')
plt.plot(epochs, train_acc_d_fake, label='Train Fake', color='green')
plt.plot(epochs, val_acc_d_real, label='Validation Real', color='orange')
plt.plot(epochs, val_acc_d_fake, label='Validation Fake', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracies Discriminator')
plt.title('Accuracies over Epochs (Discriminator 22kHz)')
plt.legend()
plt.savefig(f'{output_prefix}_accuracy_Discriminator.png')

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_cer_gt, label='CER ground truth', color='blue')
plt.plot(epochs, train_cer_recon, label='CER generated', color='red')
plt.xlabel('Epochs')
plt.ylabel('CER score [0,1]')
plt.title(' CER scores on ground truth and generated waveforms over Epochs (Generator 22kHz)')
plt.legend()
plt.savefig(f'{output_prefix}_cer_Generator.png')
