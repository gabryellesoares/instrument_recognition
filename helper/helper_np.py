import os
import sys
import math
import random
import pickle
import librosa
import warnings
import argparse
import traceback
import numpy as np
import pandas as pd
import vggish.vggish as vggish

warnings.filterwarnings('ignore')


def remove_silence(y):
    db = librosa.core.amplitude_to_db(y)
    mean_db = np.abs(db).mean()
    splitted_audio = librosa.effects.split(y=y, top_db=mean_db)

    silence_removed = []

    for inter in splitted_audio:
        silence_removed.extend(y[inter[0]:inter[1]])

    return np.array(silence_removed)


def split(y, sr, split_duration=10):
    duration = librosa.get_duration(y=y, sr=sr)
    num_segments = math.ceil(duration/split_duration)
    segments = []

    for segment in range(num_segments - 1):
        splitted_track = get_segment(y, sr, segment, split_duration, duration)
        segments.append(splitted_track)

    return segments


def get_segment(y, sr, segment, split_duration, duration):
    start_time = segment * split_duration
    end_time = min((segment + 1) * split_duration, duration)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    return y[start_sample:end_sample]


def process_demucs(audio_path, outfile):
    new_dict = {'track_id': [], 'instrument': [], 'predicted': []}
    inst_openmic = ['bass', 'drums', 'guitar', 'piano', 'voice']

    for track in os.listdir(audio_path):
        path = os.path.join(audio_path, track)

        if os.path.isfile(path):
            print(track)
            audio, sr = librosa.load(path)
            audio_splitted = remove_silence(audio)
            segments = split(audio_splitted, sr, split_duration=5)
            inst = track.split('_')[0]
            insts = []

            for _, segment in enumerate(segments):
                _, features = vggish.waveform_to_features(segment, sr)
                feature_mean = np.mean(features, axis=0, keepdims=True)
                probs = []

                for instrument in inst_openmic:
                    with open(f'openmic/models/{instrument}.pkl', 'rb') as f:
                        clf = pickle.load(f)
                    prob = clf.predict_proba(feature_mean)[0, 1]
                    probs.append((instrument, prob))

                max_prob = max(probs, key=lambda x: x[1])
                if max_prob[0] == 'voice':
                    insts.append('vocals')
                else:
                    insts.append(max_prob[0])

            most_common_instrument = max(set(insts), key=insts.count)
            new_dict['track_id'].append(track)
            new_dict['instrument'].append(inst)
            new_dict['predicted'].append(most_common_instrument)
            print('-' * 50)

    new_df = pd.DataFrame(new_dict)
    new_df.to_csv(outfile, index=False)
    print('Done.')


def process_jamendo(audio_path, outfile):
    df_jamendo = pd.read_csv('mtg-jamendo/selected-instruments.csv')
    new_dict = {'track_id': [], 'instrument': [], 'predicted': []}
    inst_openmic = ['bass', 'drums', 'guitar', 'piano', 'voice']

    for idx, row in df_jamendo.iterrows():
        print(f'track {idx}')
        full_path = os.path.join(
            audio_path, row.path.replace(".mp3", ".low.mp3").split('/')[1])
        try:
            audio, sr = librosa.load(full_path)
            audio_splitted = remove_silence(audio)
            segments = split(audio_splitted, sr, split_duration=5)
            indices = list(range(len(segments)))
            random.shuffle(indices)
            subsampled_indices = indices[:5]
            inst = row.instrument
            insts = []

            if inst == 'acousticguitar' or inst == 'electricguitar':
                inst = 'guitar'

            for i, j in enumerate(subsampled_indices):
                segment = segments[j]
                _, features = vggish.waveform_to_features(segment, sr)
                feature_mean = np.mean(features, axis=0, keepdims=True)
                probs = []

                for instrument in inst_openmic:
                    with open(f'openmic/models/{instrument}.pkl', 'rb') as f:
                        clf = pickle.load(f)
                    prob = clf.predict_proba(feature_mean)[0, 1]
                    probs.append((instrument, prob))

                max_prob = max(probs, key=lambda x: x[1])
                insts.append(max_prob[0])
        except:
            print(traceback.format_exc())
            sys.exit()

        most_common_instrument = max(set(insts), key=insts.count)
        new_dict['track_id'].append(row.track_id)
        new_dict['instrument'].append(inst)
        new_dict['predicted'].append(most_common_instrument)
        print('-' * 50)

    new_df = pd.DataFrame(new_dict)
    new_df.to_csv(outfile, index=False)
    print('Done.')

def process_nsynth(audio_path, outfile):
    df_nsynth = pd.read_csv('nsynth/nsynth_instruments.csv')
    new_dict = {'track_id': [], 'instrument': [], 'predicted': []}
    inst_openmic = ['bass', 'guitar', 'synthesizer', 'voice']

    for idx, row in df_nsynth.iterrows():
        print(f'track {idx}')
        full_path = os.path.join(audio_path, row.track_id + '.wav')
        try:
            audio, sr = librosa.load(full_path)
            #audio_splitted = remove_silence(audio)
            inst = row.instrument

            if inst == 'keyboard':
                inst = 'synthesizer'
            elif inst == 'vocal':
                inst = 'voice'
            
            _, features = vggish.waveform_to_features(audio, sr)
            feature_mean = np.mean(features, axis=0, keepdims=True)
            probs = []
            
            for instrument in inst_openmic:
                with open(f'openmic/models/{instrument}.pkl', 'rb') as f:
                    clf = pickle.load(f)
                prob = clf.predict_proba(feature_mean)[0, 1]
                probs.append((instrument, prob))
            
            max_prob = max(probs, key=lambda x: x[1])
            new_dict['track_id'].append(row.track_id)
            new_dict['instrument'].append(inst)
            new_dict['predicted'].append(max_prob[0])
            print('-' * 50) 
        except:
            print(traceback.format_exc())
            sys.exit()

    new_df = pd.DataFrame(new_dict)
    new_df.to_csv(outfile, index=False)
    print('Done.')

def process_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)
    parser.add_argument('--type', default='', type=str)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    if not args.audio_path or not args.output_file or not args.type:
        raise ValueError(
            "Both `--audio_path`, `--output_file` and `--type` must be given.")
    print('n' + args.type + 'n')
    if args.type == 'demucs':
        print('demucs')
        process_demucs(args.audio_path, args.output_file)
    elif args.type == 'jamendo':
        process_jamendo(args.audio_path, args.output_file)
    elif args.type == 'nsynth':
        process_nsynth(args.audio_path, args.output_file)
