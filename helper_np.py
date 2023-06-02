import os
import sys
import math
import random
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


def main(csv_file, audio_path, outfile):
    df_jamendo = pd.read_csv(csv_file)
    instruments = df_jamendo['instrument'].unique()
    tracks = df_jamendo['track_id'].unique()
    inst_num = len(instruments)
    songs_num = len(tracks)

    print('Extracting the vggish features...')
    X = np.empty([songs_num*5, 10, 128], dtype=int)
    count = 0

    new_dict = {'track_id': [], 'instrument': []}

    for _, row in df_jamendo.iterrows():
        full_path = os.path.join(
            audio_path, row.path.replace(".mp3", ".low.mp3"))
        try:
            y, sr = librosa.load(full_path)
            new_y = remove_silence(y)
            segments = split(new_y, sr)
            indices = list(range(len(segments)))
            random.shuffle(indices)
            subsampled_indices = indices[:5]

            for i, idx in enumerate(subsampled_indices):
                segment = segments[idx]
                _, features = vggish.waveform_to_features(segment, sr)
                X[count] = features
                count += 1

                new_dict['track_id'].append(f'{row.track_id}_{i}')
                new_dict['instrument'].append(row.instrument)

        except:
            print(traceback.format_exc())
            print(f"Not found: {full_path}")
            continue

    new_df = pd.DataFrame(new_dict)
    
    tracks2 = new_df['track_id'].unique()
    print(tracks == tracks2)
    songs_num2 = len(tracks2)
    print(songs_num == songs_num2)
    print(songs_num)
    print(songs_num2)
    print(inst_num)

    print('Extracting the labels information...')
    Y_mask = np.zeros([songs_num2, inst_num], dtype=bool)

    for _, row in new_df.iterrows():
        x_pos = int(np.arange(songs_num2)[tracks2 == row.track_id])
        y_pos = int(np.arange(inst_num)[instruments == row.instrument])
        Y_mask[x_pos, y_pos] = 1

    print(X.shape)
    print(Y_mask.shape)
    print(len(tracks2))
    
    print('Saving NPZ and CSV files...')
    np.savez(outfile, X=X, Y_mask=Y_mask, track_id=tracks2)
    new_df.to_csv('mtg-jamendo/selected-instruments-splitted.csv', index=False)   
    print('Done.')


def process_args(args):

    parser = argparse.ArgumentParser(
        description='VGGish to NumPy data generator')

    parser.add_argument('--csv_file', default='', type=str,
                        help='Path to the sparse labels CSV file.')
    parser.add_argument('--audio_path', default='',
                        type=str, help='Path to where the audio files are stored.')

    parser.add_argument('--output_file', default='',
                        type=str, help='Path and name to store the inputs files in a single NPZ file')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    if not args.csv_file or not args.audio_path or not args.output_file:
        raise ValueError(
            "Both `--csv_file`, `--audio_path` and `--output_file` must be given.")

    main(args.csv_file, args.audio_path, args.output_file)
