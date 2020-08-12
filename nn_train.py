#!/usr/bin/env python
# coding: utf-8

# Training code based on code for the book NLP with PyTorch by Rao & McMahan
# The code has been adapted to train on speech data
# Author: Badr M. Abdullah @  LSV, LST department Saarland University
# Follow me on Twitter @badr_nlp

import os
import yaml
import sys

# NOTE: import torch before pandas, otherwise segementation fault error occurs
# The couse of this problem is UNKNOWN, and not solved yet
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn.metrics import balanced_accuracy_score

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nn_speech_models import *

# Training Routine
# Helper functions
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args['training_hyperparams']['learning_rate'],
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args['model_state_file']}


def update_train_state(args, model, train_state):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """
    # save model
    torch.save(model.state_dict(),
        train_state['model_filename'] + \
        str(train_state['epoch_index'] + 1) + '.pth')

    # save model after first epoch
    if train_state['epoch_index'] == 0:
        train_state['stop_early'] = False
        train_state['best_val_accuracy'] = train_state['val_acc'][-1]

    # after first epoch check early stopping criteria
    elif train_state['epoch_index'] >= 1:
        acc_t = train_state['val_acc'][-1]

        # if acc decreased, add one to early stopping criteria
        if acc_t <= train_state['best_val_accuracy']:
            # Update step
            train_state['early_stopping_step'] += 1

        else: # if acc improved
            train_state['best_val_accuracy'] = train_state['val_acc'][-1]

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        early_stop = train_state['early_stopping_step'] >= args['training_hyperparams']['early_stopping_criteria']

        train_state['stop_early'] = early_stop

    return train_state


def compute_accuracy(y_pred, y_target):
    #y_target = y_target.cpu()
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def compute_binary_accuracy(y_pred, y_target):
    y_target = y_target.cpu().long()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long() #.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def get_predictions(y_pred, y_target):
    """Return indecies of predictions. """

    _, y_pred_indices = y_pred.max(dim=1)

    pred_labels = y_pred_indices.tolist()
    true_labels = y_target.tolist()

    return (true_labels, pred_labels)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# obtain user input
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_pah = sys.argv[1] #'/LANG-ID-X/config_1.yml'

config_args = yaml.safe_load(open(config_file_pah))


config_args['model_id'] = '_'.join(str(ip) for ip in
    [
        config_args['model_arch']['nn_model'],
        #config_args['model_arch']['bottleneck_size'],
        #config_args['model_arch']['output_dim'],
        #config_args['model_arch']['signal_dropout_prob'],
        config_args['input_signal_params']['feature_type'],
        config_args['input_signal_params']['experiment_type']
    ]
)

if config_args['expand_filepaths_to_save_dir']:
    config_args['model_state_file'] = os.path.join(
        config_args['model_save_dir'],
        '_'.join([config_args['model_state_file'],
        config_args['model_id']])
    )

    print("Expanded filepaths: ")
    print("\t{}".format(config_args['model_state_file']))


# Check CUDA
if not torch.cuda.is_available():
    config_args['cuda'] = False

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

print("Using CUDA: {}".format(config_args['cuda']))

# Set seed for reproducibility
set_seed_everywhere(config_args['seed'], config_args['cuda'])

# handle dirs
handle_dirs(config_args['model_save_dir'])

##### HERE IT ALL STARTS ...
# source vectorizer ...
source_speech_df = pd.read_csv(config_args['source_speech_metadata'],
    delimiter="\t", encoding='utf-8')

source_label_set=config_args['source_language_set'].split()

# make sure no utterances with 0 duration such as
source_speech_df = source_speech_df[(source_speech_df.duration!=0)]

source_speech_df = source_speech_df[(source_speech_df['language'].isin(source_label_set))]


len(source_speech_df), source_label_set

# source vectorizer ...
target_speech_df = pd.read_csv(config_args['target_speech_metadata'],
    delimiter="\t", encoding='utf-8')

target_label_set=config_args['target_language_set'].split()

# make sure no utterances with 0 duration such as
target_speech_df = target_speech_df[(target_speech_df.duration!=0)]

target_speech_df = target_speech_df[(target_speech_df['language'].isin(target_label_set))]

len(target_speech_df), target_label_set


source_speech_vectorizer = LID_Vectorizer(
    data_dir=config_args['source_data_dir'],
    speech_df=source_speech_df,
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['source_language_set'].split(),
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    num_frames=config_args['input_signal_params']['num_frames'],
    feature_dim=config_args['model_arch']['feature_dim'],
    start_idx=config_args['input_signal_params']['start_index'],
    end_idx=config_args['input_signal_params']['end_index']
)
print(source_speech_vectorizer.index2lang)



target_speech_vectorizer = LID_Vectorizer(
    data_dir=config_args['target_data_dir'],
    speech_df=target_speech_df,
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['target_language_set'].split(),
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    num_frames=config_args['input_signal_params']['num_frames'],
    feature_dim=config_args['model_arch']['feature_dim'],
    start_idx=config_args['input_signal_params']['start_index'],
    end_idx=config_args['input_signal_params']['end_index']
)
print(target_speech_vectorizer.index2lang)


# data loaders ....
source_speech_dataset = LID_Dataset(source_speech_df, source_speech_vectorizer)
target_speech_dataset = LID_Dataset(target_speech_df, target_speech_vectorizer)


if config_args['model_arch']['nn_model'] == 'ConvNet_DA':
    nn_LID_model_DA = ConvNet_LID_DA(
        feature_dim=config_args['model_arch']['feature_dim'],
        bottleneck=config_args['model_arch']['bottleneck'],
        bottleneck_size=config_args['model_arch']['bottleneck_size'],
        output_dim=config_args['model_arch']['output_dim'],
        dropout_frames=config_args['model_arch']['frame_dropout'],
        dropout_features=config_args['model_arch']['feature_dropout'],
        signal_dropout_prob=config_args['model_arch']['signal_dropout_prob'],
        num_channels=config_args['model_arch']['num_channels'],
        num_classes= len(source_label_set),   # or config_args['model_arch']['num_classes'],
        filter_sizes=config_args['model_arch']['filter_sizes'],
        stride_steps=config_args['model_arch']['stride_steps'],
        pooling_type=config_args['model_arch']['pooling_type']
    )


# test model
#x_in = torch.rand(1, 13, 384)
#nn_LID_model_DA.forward(x_in)


print(nn_LID_model_DA)



loss_func_cls = nn.CrossEntropyLoss()
loss_func_dmn  = nn.CrossEntropyLoss()

optimizer = optim.Adam(nn_LID_model_DA.parameters(),lr=config_args['training_hyperparams']['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                mode='min', factor=0.5,
                patience=1)

train_state = make_train_state(config_args)

# this line was added due to RunTimeError
# NOTE: uncomment this
nn_LID_model_DA.cuda()


src_val_balanced_acc_scores = []
tgt_val_balanced_acc_scores = []

num_epochs = config_args['training_hyperparams']['num_epochs']
batch_size = config_args['training_hyperparams']['batch_size']

try:
    print('Training started.')
    for epoch_index in range(num_epochs):


        ##### TRAINING SECTION
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset
        # setup: batch generator, set loss and acc to 0, set train mode on
        source_speech_dataset.set_mode('TRA')
        source_total_batches = source_speech_dataset.get_num_batches(batch_size)

        source_batch_generator = generate_batches(
            source_speech_dataset,
            batch_size=batch_size,
            device=config_args['device']
        )

        target_speech_dataset.set_mode('TRA')
        target_total_batches = target_speech_dataset.get_num_batches(batch_size)

        target_batch_generator = generate_batches(
            target_speech_dataset,
            batch_size=batch_size,
            device=config_args['device']
        )


        max_batches = min(source_total_batches, target_total_batches)
        #print(source_total_batches, target_total_batches)

        running_cls_loss = 0.0
        running_dmn_loss = 0.0

        running_cls_acc = 0.0
        running_src_dmn_acc = 0.0
        running_tgt_dmn_acc = 0.0

        nn_LID_model_DA.train()


        for batch_index, (src_batch_dict, tgt_batch_dict) in enumerate(zip(source_batch_generator, target_batch_generator)):
            # the training routine is these 5 steps:

            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. training progress and GRL lambda
            p = float(batch_index + epoch_index * max_batches) / (num_epochs * max_batches)
            _lambda = 2. / (1. + np.exp(-5 * p)) - 1


            # step 3. forward pass and compute loss on source domain
            src_dmn_trues = torch.zeros(batch_size, dtype=torch.long, device=config_args['device']) # generate source domain labels
            src_cls_trues = src_batch_dict['y_target']
            src_cls_preds, src_dmn_preds = nn_LID_model_DA(x_in=src_batch_dict['x_data'], grl_lambda=_lambda)

            loss_src_cls = loss_func_cls(src_cls_preds, src_cls_trues)

            #print(src_dmn_preds.shape, src_dmn_trues.shape)
            loss_src_dmn = loss_func_dmn(src_dmn_preds, src_dmn_trues)

            # step 4. forward pass and compute loss on target domain
            tgt_dmn_trues = torch.ones(batch_size, dtype=torch.long, device=config_args['device']) # generate source domain labels
            _, tgt_dmn_preds = nn_LID_model_DA(x_in=tgt_batch_dict['x_data'], grl_lambda=_lambda)

            loss_tgt_dmn = loss_func_dmn(tgt_dmn_preds, tgt_dmn_trues)

            # step 5. add different losses to one var
            loss = loss_src_cls + loss_src_dmn + loss_tgt_dmn

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # step 6. compute different running losses
            cls_loss = loss_src_cls.item()
            running_cls_loss += (cls_loss - running_cls_loss) / (batch_index + 1)

            dmn_loss = loss_src_dmn.item() + loss_tgt_dmn.item()
            running_dmn_loss += (dmn_loss - running_dmn_loss) / (batch_index + 1)

            # step 7. compute running source cls accuracy
            src_cls_acc = compute_accuracy(src_cls_preds, src_cls_trues)
            running_cls_acc += (src_cls_acc - running_cls_acc) / (batch_index + 1)

            # step 8. compute running source domain prediction accuracy
            src_dmn_acc = compute_accuracy(src_dmn_preds, src_dmn_trues)
            running_src_dmn_acc += (src_dmn_acc - running_src_dmn_acc) / (batch_index + 1)

            # step 9. compute running source domain prediction accuracy
            tgt_dmn_acc = compute_accuracy(tgt_dmn_preds, tgt_dmn_trues)
            running_tgt_dmn_acc += (tgt_dmn_acc - running_tgt_dmn_acc) / (batch_index + 1)

            # print summary
            print(f"{config_args['model_id']} " # {config_args['model_id']}
                f"Train Ep [{epoch_index + 1:>2}/{num_epochs}][{batch_index + 1:>3}/{max_batches}] "
                f"CLS L: {running_cls_loss:>1.5f} "
                f"DP L: {running_dmn_loss:>1.5f} "
                f"CLS ACC: {running_cls_acc:>3.2f} "
                f"S-DP ACC: {running_src_dmn_acc:>3.2f} "
                f"T-DP ACC: {running_tgt_dmn_acc:>3.2f} "
                f"l: {_lambda:.3f}"
                )


        train_state['train_loss'].append(running_cls_loss)
        train_state['train_acc'].append(running_cls_acc)

        ##### VALIDATION SECTION
        # Iterate over val dataset
        # setup: batch generator, set loss and acc to 0, set val  mode on
        source_speech_dataset.set_mode('DEV')
        source_total_batches = source_speech_dataset.get_num_batches(batch_size)

        source_batch_generator = generate_batches(
            source_speech_dataset,
            batch_size=batch_size,
            device=config_args['device']
        )

        target_speech_dataset.set_mode('DEV')
        target_total_batches = target_speech_dataset.get_num_batches(batch_size)

        target_batch_generator = generate_batches(
            target_speech_dataset,
            batch_size=batch_size,
            device=config_args['device']
        )


        max_batches = min(source_total_batches, target_total_batches)
        #print(source_total_batches, target_total_batches)

        running_cls_loss = 0.0
        running_dmn_loss = 0.0

        running_cls_acc = 0.0
        running_src_dmn_acc = 0.0
        running_tgt_dmn_acc = 0.0

        nn_LID_model_DA.eval()


        y_src_true, y_src_pred = [], []
        y_tgt_true, y_tgt_pred = [], []

        for batch_index, (src_batch_dict, tgt_batch_dict) in enumerate(zip(source_batch_generator, target_batch_generator)):

            # step 1. forward pass and compute loss on source domain
            src_dmn_trues = torch.zeros(batch_size, dtype=torch.long, device=config_args['device']) # generate source domain labels
            src_cls_trues = src_batch_dict['y_target']
            src_cls_preds, src_dmn_preds = nn_LID_model_DA(x_in=src_batch_dict['x_data'])

            loss_src_cls = loss_func_cls(src_cls_preds, src_cls_trues)
            loss_src_dmn = loss_func_dmn(src_dmn_preds, src_dmn_trues)

            # step 2. forward pass and compute loss on target domain
            tgt_dmn_trues = torch.ones(batch_size, dtype=torch.long, device=config_args['device']) # generate source domain labels
            tgt_cls_trues = tgt_batch_dict['y_target']
            tgt_cls_preds, tgt_dmn_preds = nn_LID_model_DA(x_in=tgt_batch_dict['x_data'])

            loss_tgt_dmn = loss_func_dmn(tgt_dmn_preds, tgt_dmn_trues)

            # step 3. compute overall loss
            loss = loss_src_cls + loss_src_dmn + loss_tgt_dmn

            # step 6. compute different running losses
            cls_loss = loss_src_cls.item()
            running_cls_loss += (cls_loss - running_cls_loss) / (batch_index + 1)

            dmn_loss = loss_src_dmn.item() + loss_tgt_dmn.item()
            running_dmn_loss += (dmn_loss - running_dmn_loss) / (batch_index + 1)

            # step 7. compute running source cls accuracy
            src_cls_acc = compute_accuracy(src_cls_preds, src_cls_trues)
            running_cls_acc += (src_cls_acc - running_cls_acc) / (batch_index + 1)

            # step 8. compute running source domain prediction accuracy
            src_dmn_acc = compute_accuracy(src_dmn_preds, src_dmn_trues)
            running_src_dmn_acc += (src_dmn_acc - running_src_dmn_acc) / (batch_index + 1)

            # step 9. compute running source domain prediction accuracy
            tgt_dmn_acc = compute_accuracy(tgt_dmn_preds, tgt_dmn_trues)
            running_tgt_dmn_acc += (tgt_dmn_acc - running_tgt_dmn_acc) / (batch_index + 1)

            # valid print summary
            print(f"{config_args['model_id']} " # {config_args['model_id']}
                f"Valid Ep [{epoch_index + 1:>2}/{num_epochs}][{batch_index + 1:>3}/{max_batches}] "
                f"CLS L: {running_cls_loss:>1.5f} "
                f"DP L: {running_dmn_loss:>1.5f} "
                f"CLS ACC: {running_cls_acc:>3.2f} "
                f"S-DP ACC: {running_src_dmn_acc:>3.2f} "
                f"T-DP ACC: {running_tgt_dmn_acc:>3.2f} "
                f"l: {_lambda:.3f}"
                )

            # compute balanced acc calc
            src_true_labels, src_pred_labels = get_predictions(src_cls_preds, src_cls_trues)
            tgt_true_labels, tgt_pred_labels = get_predictions(tgt_cls_preds, tgt_cls_trues)

            y_src_true.extend(src_true_labels)
            y_src_pred.extend(src_pred_labels)
            y_tgt_true.extend(tgt_true_labels)
            y_tgt_pred.extend(tgt_pred_labels)

        src_cls_acc_ep = balanced_accuracy_score(y_src_true, y_src_pred)*100
        tgt_cls_acc_ep = balanced_accuracy_score(y_tgt_true, y_tgt_pred)*100

        print(f"Summary Epoch num: [{epoch_index + 1:>2}/{num_epochs}]: "
              f"OVERALL CLS LOSS: {loss: 2.3f} "
              f"SRC CLS ACC: {src_cls_acc_ep:2.3f} "
              f"TGT CLS ACC: {tgt_cls_acc_ep:2.3f} \n"
            )

        src_val_balanced_acc_scores.append(src_cls_acc_ep)
        tgt_val_balanced_acc_scores.append(tgt_cls_acc_ep)

        train_state['val_loss'].append(running_cls_loss)
        train_state['val_acc'].append(running_cls_acc)

        train_state = update_train_state(args=config_args,
            model=nn_LID_model_DA,
            train_state=train_state
        )

        scheduler.step(train_state['val_loss'][-1])

        if train_state['stop_early']:
            break

except KeyboardInterrupt:
    print("Exiting loop")



for i, acc in enumerate(src_val_balanced_acc_scores):
    print("Validation Acc {} {:.3f}".format(i+1, acc))


print('Best epoch by balanced acc: {:.3f} epoch {}'.format(max(src_val_balanced_acc_scores),
    1 + np.argmax(src_val_balanced_acc_scores)))


print()
for i, acc in enumerate(tgt_val_balanced_acc_scores):
    print("Validation Acc {} {:.3f}".format(i+1, acc))


print('Best epoch by balanced acc: {:.3f} epoch {}'.format(max(tgt_val_balanced_acc_scores),
    1 + np.argmax(tgt_val_balanced_acc_scores)))
