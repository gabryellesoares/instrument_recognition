import os
import sys
import argparse
import wget
import hashlib
import csv
import tarfile
import pandas as pd

ID_FILE_PATH = "mtg-jamendo/"


def compute_sha256(filename):
    with open(filename, 'rb') as f:
        contents = f.read()
        checksum = hashlib.sha256(contents).hexdigest()
        return checksum


def download(output_dir):
    if not os.path.exists(output_dir):
        print('Output directory {} does not exist'.format(
            output_dir), file=sys.stderr)
        return

    file_sha256_tars = os.path.join(
        ID_FILE_PATH, 'raw_30s_audio-low_sha256_tars.txt')
    file_sha256_tracks = os.path.join(
        ID_FILE_PATH, 'raw_30s_audio-low_sha256_tracks.txt')

    with open(file_sha256_tars) as f:
        sha256_tars = dict([(row[1], row[0])
                           for row in csv.reader(f, delimiter=' ')])

    with open(file_sha256_tracks) as f:
        sha256_tracks = dict([(row[1], row[0])
                             for row in csv.reader(f, delimiter=' ')])

    ids = sha256_tars.keys()

    tracks_path = pd.read_csv('mtg-jamendo/selected-instruments.csv')['path']

    tracks_path = [path.replace(".", ".low.") for path in tracks_path]

    removed = []
    for filename in ids:
        output = os.path.join(output_dir, filename)

        url = f'https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/{filename}'

        print('From:', url)
        print('To:', output)
        wget.download(url, out=output)

        if compute_sha256(output) != sha256_tars[filename]:
            print('%s does not match the checksum, removing the file' %
                  output, file=sys.stderr)
            removed.append(filename)
            os.remove(output)
        else:
            print('%s checksum OK' % filename)

        tar = tarfile.open(output)

        members = tar.getmembers()[1:]
        tracks = tar.getnames()[1:]

        selected_members = [
            member for member in members if member.name in tracks_path]
        selected_tracks = [track for track in tracks if track in tracks_path]

        tar.extractall(path=f"{output_dir}", members=selected_members)
        tar.close()

        for track in selected_tracks:
            trackname = os.path.join(output_dir, track)
            if compute_sha256(trackname) != sha256_tracks[track]:
                print('%s does not match the checksum' %
                      trackname, file=sys.stderr)
                raise Exception('Corrupt file in the dataset: %s' % trackname)

        os.remove(output)

    if removed:
        print('Missing files:', ' '.join(removed))
        print('Re-run the script again')
        return

    print('Download complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the MTG-Jamendo dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output', help='directory to store the dataset')

    args = parser.parse_args()
    download(args.output)
