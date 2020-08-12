# coding: utf-8

# Vectorizer and data loader code are based on the code provided with
# the book: NLP with PyTorch by Rao & McMahan
# Vectorizer and models have been adapted to work with speech data
# Author: Badr M. Abdullah @  LSV, LST department Saarland University
# Follow me on Twitter @badr_nlp

import numpy as np
import random
from sklearn  import preprocessing

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Function


##### CLASS LID_Vectorizer
class LID_Vectorizer(object):
    """ The Vectorizer which takes care of speech data transformation. """

    def __init__(self,
        data_dir,
        speech_df,
        feature_type,
        label_set,
        max_num_frames,
        num_frames,
        feature_dim,
        start_idx,
        end_idx
    ):
        """
        Args:
            data_dir (str): the path to the data on disk to read .npy files
            speech_df (pandas.df): a pandas DF with (label, split, file)
            features_type (str): the feature representation of the speech data,
                for exampel, MFCCs
            num_frames (int): the number of acoustic frames to sample from the
                speech signal, 200 frames is equivalent to 2 seconds
        """
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.max_num_frames = max_num_frames
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.start_idx = start_idx
        self.end_idx = end_idx

        # form the dataframe, obtain the set of the labels in the dataset
        self.label_set = label_set
        self.lang2index = {}

        for i, label in enumerate(self.label_set):
            self.lang2index[label] = i

        self.index2lang = {i:l for (l, i) in self.lang2index.items()}


    def transform_signal_X(self,
        uttr_id,
        max_num_frames=None,
        num_frames=None,
        feature_dim=None,
        start_idx=None,
        end_idx=None,
        segment_random=False
    ):
        """
        Given a uttr_id and some other utterance-related data,
        return a feature sequence representation of the utterance.
        Args:
            uttr_id (str): the name of the wav file
            max_max_frames (int): max length of the vector sequence
                defualt is 200 which corresponds to 2 sec
        Returns:
            the vetorized data point (torch.Tensor)
        """
        if max_num_frames is None: max_num_frames = self.max_num_frames
        if num_frames is None: num_frames = self.num_frames
        if feature_dim is None: feature_dim = self.feature_dim
        if start_idx is None: start_idx = self.start_idx
        if end_idx is None: end_idx = self.end_idx

        # path to feature vector sequence (normalized)
        feat_path = self.data_dir + uttr_id + '.' + \
            self.feature_type.lower() + '.norm.npy'

        # load normalized feature sequence from desk
        feature_repr  = np.load(feat_path)

        # sampling is used to get a random segment from the speech signal
        # by default random segmentation is disabled
        if segment_random:
            # sample N frames from the utterance
            uttr_len = feature_repr.shape[1]   # utterance length in frames

            # if the signal is shorter than num_frames, take it as it is
            # added this for short utterances in DEV, EVA set
            if uttr_len - num_frames <= 0:
                sample_beg = 0
                num_frames = uttr_len
            else:
                # beginning of the random speech sample
                sample_beg = random.randrange(uttr_len - num_frames)

            sample_end = sample_beg + num_frames
            feature_seq = feature_repr[start_idx:end_idx, sample_beg:sample_end]

        else: # if no random segmentation, i.e., during inference
            feature_seq = feature_repr[start_idx:end_idx, :num_frames]


        # convert to pytorch tensor
        feature_tensor = torch.from_numpy(feature_seq)

        # apply padding to the speech sample represenation
        padded_feature_tensor = torch.zeros(feature_dim, max_num_frames)

        # this step controls both x-axis (frames) and y-axis (mels)
        # for example, when only 13 coefficients are required, then
        # use padded_feature_tensor[:14,:num_frames] = feature_tensor[:14,:num_frames]
        # likewise, the speech signal can be sampled (frame-level) as
        # padded_feature_tensor[:feature_dim,:25] = feature_tensor[:feature_dim,:25]

        # sample a random start index
        frame_start_idx = random.randrange(1 + max_num_frames - num_frames)

        # to deal with short utterances in DEV and EVA splits
        num_frames = min(feature_repr.shape[1], num_frames)

        padded_feature_tensor[:feature_dim,frame_start_idx:frame_start_idx + num_frames] = \
            feature_tensor[:feature_dim,:num_frames]


        return padded_feature_tensor.float() # convert to float tensor


    def transform_label_y(self, label):
        """
        Given the label of the data point (language), return label index.
        Args:
            label (str): the target language in the dataset (e.g., 'ru')
        Returns:
            the index of the label in the experiment.
        """
        return self.lang2index[label]


##### CLASS LID_Dataset
class LID_Dataset(Dataset):
    def __init__(self, speech_df, vectorizer):
        """
        Args:
            speech_df (pandas.df): a pandas dataframe (label, split, file)
            vectorizer (LID_Vectorizer): the speech vectorizer
        """
        self.speech_df = speech_df
        self._vectorizer = vectorizer

        # read data and make splits
        self.train_df = self.speech_df[self.speech_df.split=='TRA']
        self.train_size = len(self.train_df)

        self.val_df = self.speech_df[self.speech_df.split=='DEV']
        self.val_size = len(self.val_df)

        self.test_df = self.speech_df[self.speech_df.split=='EVA']
        self.test_size = len(self.test_df)

        #print(self.train_size, self.val_size, self.test_size)

        self._lookup_dict = {
            'TRA': (self.train_df, self.train_size),
            'DEV': (self.val_df, self.val_size),
            'EVA': (self.test_df, self.test_size)
        }

        # by default set mode to train
        self.set_mode(split='TRA')

        # this was added to differentiate between training & inference
        self.debug_mode = None


    def set_mode(self, split='TRA'):
         """Set the mode using the split column in the dataframe. """
         self._target_split = split
         self._target_df, self._target_size = self._lookup_dict[split]


    def __len__(self):
        return self._target_size


    def __getitem__(self, index):
        """Data transformation logic for one data point.
        Args:
            index (int): the index to the data point in the dataframe
        Returns:
            a dictionary holding the point representation:
                signal (x_data), label (y_target), and uttr ID (uttr_id)
        """
        uttr = self._target_df.iloc[index]

        # enable random segmentation during training
        is_training = (self._target_split=='TRA')

        feature_sequence = self._vectorizer.transform_signal_X(uttr.uttr_id,
            segment_random = is_training,
            num_frames=None, # it is important to set this to None
            feature_dim=None
        )

        lang_idx = self._vectorizer.transform_label_y(uttr.language)

        return {
            'x_data': feature_sequence,
            'y_target': lang_idx,
            'uttr_id': uttr.uttr_id
        }


    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


##### A METHOD TO GENERATE BATCHES
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=False, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            if name != 'uttr_id':
                out_data_dict[name] = data_dict[name].to(device)
            else:
                out_data_dict[name] = data_dict[name]
        yield out_data_dict


##### A custome layer for frame dropout
class FrameDropout(nn.Module):
    def __init__(self, dropout_prob=0.2):
        """Applies dropout on the frame level so entire feature vector will be
            evaluated to zero vector with probability p.
        Args:
            p (float): dropout probability
        """
        super(FrameDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x_in):
        batch_size, feature_dim, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_frame_idx = [i for i in range(sequence_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, :, drop_frame_idx] = 0

        return x_in


##### A custome layer for feature dropout
class FeatureDropout(nn.Module):
    def __init__(self, dropout_prob=0.2, feature_idx=None):
        """Applies dropout on the feature level so feature accross vectors are
            are replaced with zero vector with probability p.
        Args:
            p (float): dropout probability
        """
        super(FeatureDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x_in):
        batch_size, feature_dim, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_feature_idx = [i for i in range(feature_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, drop_feature_idx, :] = 0

        return x_in


##### A custome layer for frame sequence reversal
class FrameReverse(nn.Module):
    def __init__(self):
        """Reverses the frame sequence in the input signal. """
        super(FrameReverse, self).__init__()

    def forward(self, x_in):
        batch_size, feature_dim, sequence_dim = x_in.shape
        # reverse indicies
        reversed_idx = [i for i in reversed(range(sequence_dim))]
        x_in[:, :, reversed_idx] = x_in

        return x_in


##### A custome layer for frame sequence shuflle
class FrameShuffle(nn.Module):
    def __init__(self):
        """Shuffle the frame sequence in the input signal, given a bag size. """
        super(FrameShuffle, self).__init__()

    def forward(self, x_in, bag_size):
        batch_size, feature_dim, seq_dim = x_in.shape

        # shuffle idicies according to bag of frames size
        # make the bags of frames
        seq_idx = list(range(seq_dim))

        # here, a list of bags (lists) will be made
        frame_bags = [seq_idx[i:i+bag_size] for i in range(0, seq_dim, bag_size)]

        # shuffle the bags
        random.shuffle(frame_bags)

        # flatten the bags into a sequential list
        shuffled_idx = [idx for bag in frame_bags for idx in bag]

        x_in[:, :, shuffled_idx] = x_in

        return x_in


##### A Convolutional model: Spoken Language ID model
class ConvNet_LID(nn.Module):
    def __init__(self,
        feature_dim=14,
        num_classes=6,
        bottleneck=False,
        bottleneck_size=64,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enables
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_features: # if frame dropout is enables
            self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()



        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=362, stride=1)
        else:
            raise NotImplementedError

        # Fully conntected layers block
        self.language_classifier = torch.nn.Sequential()

        if bottleneck:
            # Bottleneck layer
            self.language_classifier.add_module("fc_bn",
                nn.Linear(num_channels[2], self.bottleneck_size))
            self.language_classifier.add_module("relu_bn", nn.ReLU())

            # then project to higher dim
            self.language_classifier.add_module("fc2",
                nn.Linear(self.bottleneck_size, self.output_dim))
            self.language_classifier.add_module("relu_fc2", nn.ReLU())

        else:
            # then project to two identical fc layers
            #self.language_classifier.add_module("fc1", nn.Linear(512, 512))
            #self.language_classifier.add_module("relu_fc1", nn.ReLU())

            self.language_classifier.add_module("fc2",
                nn.Linear(num_channels[2], self.output_dim))
            self.language_classifier.add_module("relu_fc2", nn.ReLU())

        # Output fully connected --> softmax
        self.language_classifier.add_module("y_hat",
            nn.Linear(self.output_dim, num_classes))


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)


        # signal masking during inference
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in)
        if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        f = self.ConvLayer1(x_in)
        f = self.ConvLayer2(f)
        f = self.ConvLayer3(f)

        # max pooling
        f = self.PoolLayer(f).squeeze(dim=2)

        # if we need to analyze bottleneck feature, go into this code block
        if return_bn:
            feature_vector = f
            for _name, module in self.language_classifier._modules.items():
                feature_vector = module(feature_vector)

                if _name == 'relu_bn':
                    return feature_vector


        y_hat = self.language_classifier(f)

        # softmax
        if apply_softmax:
            y_hat = torch.softmax(y_hat, dim=1)

        return y_hat


# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, _lambda):
        # Store context for backprop
        ctx._lambda = _lambda

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -_lambda the gradient
        output = grad_output.neg() * ctx._lambda

        # Must return same number as inputs to forward()
        return output, None


##### DA-LID I: Spoken Language ID Model with Domain Adaptation [1]
class ConvNet_LID_DA(nn.Module):
    def __init__(self,
        feature_dim=14,
        num_classes=6,
        bottleneck=False,
        bottleneck_size=64,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID_DA, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enables
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_features: # if frame dropout is enables
            self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()

        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=362, stride=1)
        else:
            raise NotImplementedError

        # Fully conntected layers block - Language classifier
        self.language_classifier = torch.nn.Sequential()

        self.language_classifier.add_module("fc_bn",
            nn.Linear(num_channels[2], self.bottleneck_size))
        self.language_classifier.add_module("relu_bn", nn.ReLU())

        # then project to higher dim
        self.language_classifier.add_module("fc2",
            nn.Linear(self.bottleneck_size, self.output_dim))
        self.language_classifier.add_module("relu_fc2", nn.ReLU())

        # Output fully connected --> softmax
        self.language_classifier.add_module("y_hat",
            nn.Linear(self.output_dim, num_classes))

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_channels[2], 1024),
            nn.ReLU(),
            nn.Linear(num_channels[2], 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1,
        grl_lambda=1.0
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)


        # signal masking during inference
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in)
        if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        f = self.ConvLayer1(x_in)
        f = self.ConvLayer2(f)
        f = self.ConvLayer3(f)

        # max pooling
        f = self.PoolLayer(f).squeeze(dim=2)

        # if we need to analyze bottle neck feature, go into this code block
        # if return_bn:
        #     feature_vector = f
        #     for _name, module in self.language_classifier._modules.items():
        #         feature_vector = module(feature_vector)
        #
        #         if _name == 'relu_bn':
        #             return feature_vector


        reverse_f = GradientReversal.apply(f, grl_lambda)

        y_hat = self.language_classifier(f)
        d_hat = self.domain_classifier(reverse_f)

        # softmax
        if apply_softmax:
            y_hat = torch.softmax(y_hat, dim=1)

        return y_hat, d_hat


##### DA-LID II: Spoken Language ID Model with Domain Adaptation [2]
class ConvNet_LID_DA_2(nn.Module):
    def __init__(self,
        feature_dim=14,
        num_classes=6,
        bottleneck=False,
        bottleneck_size=64,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        #unit_dropout_prob=0.5,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID_DA_2, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        #self.unit_dropout_prob = unti_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enables
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_features: # if frame dropout is enables
            self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()

        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=362, stride=1)
        else:
            raise NotImplementedError

        # Fully conntected layers block - Language classifier
        self.fc_layer = torch.nn.Sequential()

        self.fc_layer.add_module("fc1",
            nn.Linear(num_channels[2], self.bottleneck_size))
        self.fc_layer.add_module("relu_fc1", nn.ReLU())



        self.language_classifier = torch.nn.Sequential()

        self.language_classifier.add_module("fc2",
            nn.Linear(self.bottleneck_size, self.output_dim))
        self.language_classifier.add_module("relu_fc2", nn.ReLU())
        #self.language_classifier.add_module("drop_fc2", nn.Dropout(self.unit_dropout_prob))

        # Output fully connected --> softmax
        self.language_classifier.add_module("y_out",
            nn.Linear(self.output_dim, num_classes))

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_channels[2], 1024), #nn.BatchNorm1d(100),
            nn.ReLU(),
            #nn.Dropout(self.unit_dropout_prob),
            nn.Linear(1024, 1024), #nn.BatchNorm1d(100),
            nn.ReLU(),
            #nn.Dropout(self.unit_dropout_prob),
            nn.Linear(1024, 2)
        )


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1,
        grl_lambda=1.0
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)


        # signal masking during inference
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in)
        if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        f = self.ConvLayer1(x_in)
        f = self.ConvLayer2(f)
        f = self.ConvLayer3(f)

        # max pooling
        f = self.PoolLayer(f).squeeze(dim=2)

        # fc features
        f = self.fc_layer(f)

        #
        # if we need to analyze bottle neck feature, go into this code block
        if return_bn:
            feature_vector = f

            for _name, module in self.language_classifier._modules.items():
                feature_vector = module(feature_vector)

                if _name == 'relu_fc2':
                    return feature_vector



        reverse_f = GradientReversal.apply(f, grl_lambda)

        y_hat = self.language_classifier(f)
        d_hat = self.domain_classifier(reverse_f)

        # softmax
        if apply_softmax:
            y_hat = torch.softmax(y_hat, dim=1)

        return y_hat, d_hat
