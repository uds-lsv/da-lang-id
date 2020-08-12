# -*- coding: utf-8 -*-

# core imports
import sys
import random
import collections
import csv
import random
import argparse
import glob

# numerical computation
import pandas as pd
import numpy as np
from sklearn  import preprocessing

# for audio transformations
import librosa


def extract_features(wavfile, feature, sampling_rate=16000):
    """
    Given a raw wav signal and sample rate,
    return feature representation numerical object and duration.
    """

    raw_signal, sr = librosa.core.load(wavfile,
        sampling_rate,
        mono=True,
        dtype='float'
    )


    if feature == 'MFCC':
        feat_seq = librosa.feature.mfcc(raw_signal,
            sampling_rate,
            n_fft=400,
            hop_length=160,
            n_mfcc=13,
            fmin=75,
            fmax=5999
        )
        # Numerical Stability
        #feat_seq = np.where(feat_seq == 0, np.finfo(float).eps, feat_seq)


    elif feature == 'FBANK':
        feat_seq = librosa.feature.melspectrogram(raw_signal,
            sampling_rate,
            n_fft=400,
            hop_length=160,
            n_mels=13,
            fmin=75,
            fmax=5999
        )

        # Numerical Stability
        feat_seq = np.where(feat_seq == 0, np.finfo(float).eps, feat_seq)

        # 20 * log | convert to Me-Scale
        feat_seq = 20*np.log10(feat_seq)

    # z-norm: feature normalization
    feat_norm = preprocessing.scale(feat_seq, axis=1)

    return feat_norm



def parse_user_input():
    """
    Read folder path from user input, and prepare dataset.
    """
    DISC = 'Generate dataset from input files to one csv frame.'
    parser = argparse.ArgumentParser(description=DISC)

    # USER ARGS
    parser.add_argument('-raw_dir',
        type=str,
        help='Path to the dir of raw data.',
        required=True
    )

    parser.add_argument('-csv_file',
        type=str,
        help='CSV file of the utterances to transform.',
        required=True
    )

    parser.add_argument('-feature_dir',
        type=str,
        help='Path to the dir of output feature representations.',
        required=True
    )

    parser.add_argument('-feature_type',
        type=str,
        help='Feature representation of the speech signal.',
        required=True
    )

    return parser.parse_args()


def transform_wav_audio_files():

    user_args = parse_user_input()

    # ARGs
    args = argparse.Namespace(
        path_to_raw_dir=user_args.raw_dir,
        path_to_feature_dir=user_args.feature_dir,
        csv_file=user_args.csv_file,
        feature_type=user_args.feature_type
    )

    # read all .csv file from user Path
    dataset_df = pd.read_csv(args.csv_file, sep='\t', encoding='utf-8')

    dataset_df = dataset_df.rename(columns={'Unnamed: 0': 'utterance_id'})

    print(dataset_df.head())

    # iterate over each wav files
    for idx, uttr in dataset_df.iterrows():

        try:
            feat_seq = extract_features(args.path_to_raw_dir + uttr.file,
                    args.feature_type
                )

            # write to disk
            op_file_str = args.path_to_feature_dir + uttr.utterance_id + '.' + \
                args.feature_type.lower() + '.norm.npy'

            np.save(op_file_str, feat_seq)

            print('Number of transformed samples:', idx + 1, op_file_str, \
                feat_seq.shape)
        except  Exception as e:
            print('Exception occured at', uttr.utterance_id, 'len' ,uttr.duration, str(e))



def main():
    transform_wav_audio_files()

if __name__ == '__main__':
    main()
