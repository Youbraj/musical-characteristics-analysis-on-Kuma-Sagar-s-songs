import numpy as np        # Import the numpy library for numerical operations, especially with arrays and matrices
import matplotlib.pyplot as plt         # Import the matplotlib library for plotting and visualizing data
import IPython.display as ipd           # Import IPython display module to play audio and display media in Jupyter Notebooks
import librosa     # Import the librosa library for audio analysis
import librosa.display          # Import the display submodule from librosa to help with visualizing audio features (like waveforms, spectrograms)

# source of Furfuri (song by Kuma Sagar):
# https://www.youtube.com/watch?v=FLbE0IkHLkg

# source of A Mai Re - 'Hawa Ko Lahar' (song by Kuma Sagar):
# https://www.youtube.com/watch?v=gebozQyu-pY

""" A class built with the librosa library to identify the musical key and emotion of an MP3 file,
offering configurable parameters for enhanced analysis.
The key arguments include the waveform (an MP3 file loaded through librosa,
preferably with percussive elements removed),
the sr (sampling rate obtained during file loading),
and tstart and tend (optional parameters defining the time range in seconds for analysis,
defaulting to the file's full duration if unspecified).
"""
class Tonal_Fragment(object):
    def __init__(self, waveform, sr, tstart=None, tend=None):
        # Initialize the instance with the provided parameters
        self.waveform = waveform    # Store the waveform (audio signal)
        self.sr = sr            # Store the sample rate (samples per second)
        self.tstart = tstart        # Store the start time for the fragment (in seconds)
        self.tend = tend            # Store the end time for the fragment (in seconds)

        # If an end time is provided, convert it to sample index
        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)

        # If an end time is provided, convert it to sample index
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)

        # Extract the segment of the waveform between tstart and tend (in sample indices)
        self.y_segment = self.waveform[self.tstart:self.tend]

        # Compute the chroma feature (chromagram) from the waveform segment using Constant-Q transform
        # Chroma features represent the energy in different pitch classes (e.g., C, D, E)
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)

        # chroma_vals represents the quantity of each pitch class present
        # within the specified time interval.
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        # List of musical pitch classes
        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        # A dictionary mapping pitch names to their corresponding intensity values within the song.
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)}
        # Generate a list of key names (both major and minor keys)
        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]
        """
        Using the Krumhansl-Schmuckler key-finding algorithm, the chroma data is analyzed and
        compared against established key profiles for major and minor scales,
        helping to identify the most likely key of the music.
        """
        # Major and minor key profiles (based on the Krumhansl-Schmuckler algorithm)
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        """
        The algorithm computes the correlations between the distribution of each pitch class
        in the time interval and the key profiles for all 12 pitches, then generates
        a dictionary mapping each musical key (major/minor) to its corresponding correlation value.
        """
        # List to hold correlation coefficients for each key
        self.min_key_corrs = []
        self.maj_key_corrs = []
        # Calculate the correlation between the chroma data and major/minor key profiles
        for i in range(12):
            # Construct the chroma vector for the current pitch class
            key_test = [self.keyfreqs.get(pitches[(i + m)%12]) for m in range(12)]
            # Correlation coefficients represent the strength of the relationship between the pitch class
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1,0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1,0], 3))

        # Create a dictionary mapping each key (major/minor) to its correlation value
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)},
                         **{keys[i+12]: self.min_key_corrs[i] for i in range(12)}}

        # Identify the key with the highest correlation
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())

        # Identify the second most likely key, if its correlation is close to the primary key
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr*0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr

    # Displays the relative prominence of each pitch class in the analysis
    def print_chroma_e(self):
        # Normalize chroma values to show their prominence
        self.chroma_max = max(self.chroma_vals)
        for key, chrom in self.keyfreqs.items():
            m=""
            if key==max(self.key_dict, key=self.key_dict.get)[:-6] : m="(Max)"
            print(key, '\t', f'{chrom/self.chroma_max:5.3f}', m)

    # Generates the correlation coefficients linked to each major and minor key.
    def corr_table_e(self, kr):
        for key, corr in self.key_dict.items():
            e=""
            m=""
            if key==max(self.key_dict, key=self.key_dict.get) :
              e=kr
              m="(Max)"
            x=max(self.key_dict, key=self.key_dict.get)
            print(key, '\t', f'{corr:6.3f}', m, '\t', e)

    # printout of the key determined by the algorithm; if another key is close, that key is mentioned
    def print_key(self):
        print("likely key: ", max(self.key_dict, key=self.key_dict.get), ", correlation: ", self.bestcorr, sep='')
        if self.altkey is not None:
                print("also possible: ", self.altkey, ", correlation: ", self.altbestcorr, sep='')


    # Display the key identified by the algorithm, and if another key is nearby, include that key as well.
    def chromagram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=sr, bins_per_octave=24)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        if title is None:
            plt.title('Chromagram')
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Returns an emotional description based on the key detected
    def key_return(self):
        k= max(self.key_dict, key=self.key_dict.get)        # Get the primary key
        c=k
        if self.altkey is not None:
            lk=self.altkey                # Include the alternate key if close enough
            c=k+"(also likely "+lk+")"

        # Dictionary mapping keys to emotional descriptions
        thiss ={"C major": "Innocently Happy", "C minor": "Innocently Sad, Love-Sick", "C# minor": "Despair, Wailing, Weeping", "C# major": "Grief, Depressive", "D major": "Triumphant, Victorious War-Cries", "D minor": "Serious, Pious, Ruminating", "D# minor": "Deep Distress, Existential Angst", "D# major": "Cruel, Hard, Yet Full of Devotion", "E major": "Quarrelsome, Boisterous, Incomplete Pleasure", "E minor": "Effeminate, Amorous, Restless", "F major": "Furious, Quick-Tempered, Passing Regret", "F minor": "Obscure, Plaintive, Funereal", "F# major": "Conquering Difficulties, Sighs of Relief", "F# minor": "Gloomy, Passionate Resentment", "G major": "Serious, Magnificent, Fantasy", "G minor": "Discontent, Uneasiness", "G# major": "Death, Eternity, Judgement", "G# minor": "Grumbling, Moaning, Wailing", "A major": "Joyful, Pastoral, Declaration of Love", "A minor": "Tender, Plaintive, Pious", "A# major": "Joyful, Quaint, Cheerful", "A# minor": "Terrible, the Night, Mocking", "B major": "Harsh, Strong, Wild, Rage", "B minor": "Solitary, Melancholic, Patience"}
        # Return the corresponding emotional description based on the detected key
        x="Emotion: " +thiss[k]
        return x


