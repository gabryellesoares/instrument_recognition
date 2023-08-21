import pandas as pd

instruments = ['acousticguitar', 'bass',
               'drums', 'electricguitar', 'piano', 'voice']

df = pd.read_csv('mtg-jamendo/autotagging_instrument.tsv',
                 sep='\t', on_bad_lines='skip')
df.rename(columns={'TRACK_ID': 'track_id',
          'PATH': 'path', 'TAGS': 'instrument'}, inplace=True)
df.drop(columns=['ALBUM_ID', 'ARTIST_ID', 'DURATION'], inplace=True)
df['instrument'] = df['instrument'].apply(lambda x: x.split('-')[-1])
df = df[df['instrument'].isin(instruments)]

piano_samples = df[df['instrument'] == 'piano'].sample(n=150, random_state=42)
other_samples = df[df['instrument'] != 'piano']

balanced_df = pd.concat([piano_samples, other_samples])
balanced_df = balanced_df.reset_index(drop=True)

balanced_df.to_csv('mtg-jamendo/selected-instruments.csv', index=False)
