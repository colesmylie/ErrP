import os
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
import config
from scipy.stats import zscore

from mne.time_frequency import tfr_morlet
from mne.time_frequency import AverageTFR

from Utils.preprocessing import (
    apply_notch_filter,
    extract_segments,
    separate_classes,
    compute_grand_average,
    concatenate_streams
)
from Utils.stream_utils import get_channel_names_from_xdf, load_xdf

subject = "PILOT_ERP_"
session = "S001OFFLINE_NOFES"

xdf_dir =  "/home/millanslab/cole_ErrP/ErrP/"

if not os.path.exists(xdf_dir):
    raise FileNotFoundError(f"❌ EEG directory not found: {xdf_dir}")

xdf_files = [
    os.path.join(xdf_dir, f)
    for f in os.listdir(xdf_dir)
    if f.endswith(".xdf")
]

if not xdf_files:
    raise FileNotFoundError(f"❌ No XDF files found in: {xdf_dir}")

print(f"📂 Found {len(xdf_files)} XDF files in: {xdf_dir}")
for idx, file in enumerate(xdf_files, start=1):
    print(f" [{idx}] {os.path.basename(file)}")

print("\nPress ENTER to merge **all** files, or enter the number(s) of the file(s) to load (comma-separated, e.g., 1,3): ")
user_input = input("➡️  Selection: ").strip()

selected_files = []
if user_input:
    try:
        selected_indices = [int(i) - 1 for i in user_input.split(",")]
        selected_files = [xdf_files[i] for i in selected_indices if 0 <= i < len(xdf_files)]
    except ValueError:
        print("❌ Invalid input. Loading all files instead.")
        selected_files = xdf_files
else:
    selected_files = xdf_files

all_streams = []
all_headers = []

eeg_streams, marker_streams = [], []
for xdf_file in xdf_files:
    eeg_s, marker_s = load_xdf(xdf_file)
    eeg_streams.append(eeg_s)
    marker_streams.append(marker_s)

print(f"✅ Successfully loaded and merged {len(all_streams)} streams from {len(selected_files)} XDF file(s).")

# If multiple files are chosen, concatenate them; if only one, just take the single streams
eeg_stream, marker_stream = (
    (eeg_streams[0], marker_streams[0]) if len(eeg_streams) == 1
    else concatenate_streams(eeg_streams, marker_streams)
)

eeg_timestamps = np.array(eeg_stream["time_stamps"])  
eeg_data = np.array(eeg_stream["time_series"]).T  
channel_names = get_channel_names_from_xdf(eeg_stream)
marker_data = np.array([int(value[0]) for value in marker_stream['time_series']])
marker_timestamps = np.array(marker_stream['time_stamps'])
print("\n EEG Channels from XDF:", channel_names)

montage = mne.channels.make_standard_montage("standard_1020")

rename_dict = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
    "POZ": "POz", "OZ": "Oz"
}

non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]
valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]  
eeg_data = eeg_data[valid_indices, :]  

sfreq = config.FS
info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(eeg_data, info)

first_channel_unit = raw.info["chs"][0]["unit"]
print(f"First Channel Unit (FIFF Code): {first_channel_unit}")

# Convert from volts to microvolts (data now in SI units)
raw._data /= 1e3  
for ch in raw.info['chs']:
    ch['unit'] = 201  # FIFF_UNIT_V

print(f"Updated Units for EEG Channels: {[ch['unit'] for ch in raw.info['chs']]}")

if "M1" in raw.ch_names and "M2" in raw.ch_names:
    raw.drop_channels(["M1", "M2"])
    print("Removed Mastoid Channels: M1, M2")
else:
    print("No Mastoid Channels Found in Data")

raw.rename_channels(rename_dict)

missing_in_montage = set(raw.ch_names) - set(montage.ch_names)
print(f"⚠️ Channels in Raw but Missing in Montage: {missing_in_montage}")

# Set the montage
raw.set_montage(montage, match_case=True, on_missing="warn")

# Filtering and applying Current Source Density (CSD)
highband = 10
lowband = 1
raw.notch_filter(60)  
raw.filter(l_freq=lowband, h_freq=highband, fir_design="firwin")  
raw = mne.preprocessing.compute_current_source_density(raw)

print("\n Final EEG Channels After Processing:", raw.ch_names)

# Create MNE events array
unique_markers = np.unique(marker_data)
event_dict = {str(marker): marker for marker in unique_markers}
events = np.column_stack((
    np.searchsorted(eeg_timestamps, marker_timestamps),  
    np.zeros(len(marker_data), dtype=int),
    marker_data
))

marker_labels = {
    "100": "Rest",
    "200": "Right Arm MI",
    "300": "Robot Move",
    "340": "Robot Early Stop"
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-1,
    tmax=5,
    baseline=None,
    detrend=1,
    preload=True
)

for marker in ["100", "200", "300", "340"]:
    if marker in epochs.event_id:
        print(f"Marker {marker}: {len(epochs[marker])} epochs")

##############################################################################
# Select epochs for the three conditions: Rest (100), MI (200), Error (340)
##############################################################################
conditions = {
    "Rest": "100",
    "MI": "200",
    "Error": "340"
}

# Dictionary to hold the cropped & baseline-corrected epochs for each condition.
conditions_epochs = {}

# Extract continuous events from the 'events' array:
mi_events = events[events[:, 2] == 200]
errp_events = events[events[:, 2] == 340]

# For MI events that are paired with an error trial, build a set of MI sample indices to remove.
paired_mi_set = set()
for err in errp_events:
    candidate_indices = np.where(mi_events[:, 0] < err[0])[0]
    if candidate_indices.size > 0:
        candidate = mi_events[candidate_indices[-1], 0]
        paired_mi_set.add(candidate)

##############################################################################
# Process each condition
##############################################################################
for cond_name, marker_str in conditions.items():
    if marker_str in epochs.event_id:
        ep_cond = epochs[marker_str].copy()
        
        # For MI, exclude trials that are paired with error trials.
        if marker_str == "200":
            pure_indices = [i for i, ev in enumerate(ep_cond.events) if ev[0] not in paired_mi_set]
            ep_cond = ep_cond[pure_indices].copy()
            print(f"Selected {len(ep_cond)} pure MI trials (marker 200) after excluding error trials.")
        else:
            print(f"Found {len(ep_cond)} trials for condition {cond_name} (marker {marker_str}).")
        
        # For Rest and MI, ignore early activity by using the segment starting 3.0 s in.
        # For Error, use the segment as is (cropped from -0.2 to 1.0 s).
        if cond_name in ['Rest', 'MI']:
            # Crop from 3.0 to 4.2 sec.
            ep_crop = ep_cond.crop(tmin=3.0, tmax=4.2)
            # Use the first 200 ms of the cropped segment (3.0 to 3.2 s) as baseline.
            baseline_indices = ep_crop.time_as_index([ep_crop.times[0], ep_crop.times[0] + 0.2])
        else:
            ep_crop = ep_cond.crop(tmin=-0.2, tmax=1.0)
            baseline_indices = ep_crop.time_as_index([-0.2, 0])
        
        idx_start, idx_end = baseline_indices
        baseline_mean = np.mean(ep_crop._data[:, :, idx_start:idx_end], axis=2, keepdims=True)
        ep_crop._data -= baseline_mean
        
        conditions_epochs[cond_name] = ep_crop
    else:
        print(f"⚠️ Condition {cond_name} (marker {marker_str}) not found.")
        conditions_epochs[cond_name] = None

##############################################################################
# Plot time-domain Grand Averages for Fz and Cz, with all conditions overlaid
##############################################################################
physio_channels = ['Fz', 'Cz']
colors = ['r', 'b', 'g']  # One color per condition

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)
fig.suptitle("Grand Average EEG Signals by Trial Class for Fz and Cz")

for ax_idx, ch_name in enumerate(physio_channels):
    ax = axes[ax_idx]
    # Draw vertical line at time 0 (error trigger remains at its actual time)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # Plot each condition on the same time axis.
    for i, (cond_name, ep_obj) in enumerate(conditions_epochs.items()):
        if ep_obj is None:
            continue
        
        # For Error, do not shift the time axis; for Rest and MI, subtract 3.0 s.
        if cond_name == 'Error':
            times_plot = ep_obj.times * 1000  # remains as -200 to 1000 ms
        else:
            times_plot = (ep_obj.times - 3.2) * 1000  # shifted to overlay (0 to 1200 ms)
        
        # Compute mean and SEM (convert from volts to microvolts)
        data = ep_obj._data[:, ep_obj.ch_names.index(ch_name), :]
        mean_trace = np.mean(data, axis=0) 
        sem_trace = (np.std(data, axis=0) / np.sqrt(data.shape[0])) 
        
        # Plot shaded SEM and mean trace.
        ax.fill_between(times_plot,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=colors[i % len(colors)],
                        alpha=0.3)
        ax.plot(times_plot, mean_trace,
                label=cond_name,
                color=colors[i % len(colors)],
                linewidth=2)
    
    ax.set_title(f"{ch_name} Channel")
    ax.set_xlabel("Time (ms)")
    if ax_idx == 0:
        ax.set_ylabel("Amplitude (µV)")
    ax.legend(loc='best')

plt.tight_layout()
plt.show()


##############################################################################
# Optional: Time-Frequency for Error epochs (or any other condition you like)
##############################################################################
# Example shown only for Error (340). You can replicate for Rest / MI if desired.
errp_epochs_crop_nobase = epochs["340"].copy().crop(tmin=-0.2, tmax=1.0)
# (No baseline correction here, or you could do it similarly as above.)
physio_channels = ['Fz', 'Cz']  # TFR picks

freqs = np.arange(1, 31, 1)
n_cycles = freqs / 2.0

tfr = errp_epochs_crop_nobase.compute_tfr(
    method='morlet',
    freqs=freqs,
    n_cycles=n_cycles,
    picks=physio_channels,
    average=True,   # Returns an AverageTFR
    return_itc=False
)

# Plot TFR for Fz
if "Fz" in tfr.ch_names:
    pick_fz = [tfr.ch_names.index("Fz")]
    tfr.plot(picks=pick_fz,
             title="Time-Frequency Decomposition for Fz (Error)",
             cmap='viridis')
else:
    print("Channel Fz was not found in the TFR data.")

# Plot TFR for Cz
if "Cz" in tfr.ch_names:
    pick_cz = [tfr.ch_names.index("Cz")]
    tfr.plot(picks=pick_cz,
             title="Time-Frequency Decomposition for Cz (Error)",
             cmap='viridis')
else:
    print("Channel Cz was not found in the TFR data.")
"""
##############################################################################
# Compute Difference in Time-Frequency Representations between Error and MI trials
##############################################################################
# Define frequency range and number of cycles.
freqs = np.arange(1, 10, .1)
n_cycles = 6
physio_channels = ['Fz', 'Cz']

# ---------------------
# Process Error epochs for TFR:
# ---------------------
# Use the Error epochs (marker "340") and crop them to [-0.2, 1.0] s.
errp_epochs = epochs["340"].copy().crop(tmin=-0.2, tmax=1.0)

# ---------------------
# Process MI epochs for TFR:
# ---------------------
# Select the MI epochs (marker "200") and remove those paired with error events.
mi_epochs = epochs["200"].copy()
pure_indices = [i for i, ev in enumerate(mi_epochs.events) if ev[0] not in paired_mi_set]
mi_epochs = mi_epochs[pure_indices].copy()
# Crop MI epochs (as originally processed from 3.0 to 4.2 s)...
mi_epochs = mi_epochs.crop(tmin=3.0, tmax=4.2)
# ...and shift the time axis by -3.2 s to align with Error epochs
mi_epochs.shift_time(-3.2, relative=True)

# ---------------------
# Compute TFR for each condition using Morlet wavelets:
# ---------------------
from mne.time_frequency import tfr_morlet
tfr_err = tfr_morlet(errp_epochs, freqs=freqs, n_cycles=n_cycles,
                     picks=physio_channels, average=True, return_itc=False)
tfr_mi = tfr_morlet(mi_epochs, freqs=freqs, n_cycles=n_cycles,
                    picks=physio_channels, average=True, return_itc=False)

# ---------------------
# Apply baseline correction:
# ---------------------
# Use the baseline interval -0.2 to 0 s and normalize the power (percent change from baseline).
baseline = (-0.2, 0)
tfr_err.apply_baseline(baseline=baseline, mode='percent')
tfr_mi.apply_baseline(baseline=baseline, mode='percent')

# ---------------------
# Compute difference in TFR power (Error - MI):
# ---------------------
# Create a new TFR object to store the difference.
difference_tfr = tfr_err.copy()
difference_tfr.data = tfr_err.data - tfr_mi.data
difference_tfr.comment = "Difference (Error - MI) power (percent change from baseline)"

# ---------------------
# Plot the difference for each channel:
# ---------------------
for channel in physio_channels:
    # Pick the channel of interest for individual plotting.
    tfr_diff_channel = difference_tfr.copy().pick_channels([channel])
    tfr_diff_channel.plot(title=f"Difference TFR (Error - MI) for {channel}",
                          cmap='RdBu_r')  # Use a diverging colormap for difference plots
"""