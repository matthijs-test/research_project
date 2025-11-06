"""
PHM2010 Data Preprocessing Module

This module contains functions for:
- Downloading and preprocessing PHM2010 dataset
- Applying transforms (CWT, STFT)
- Creating sliding windows
- Exporting datasets as images for ViT training
"""

import os
import re
import numpy as np
import pandas as pd
import pywt
import scipy.signal as scipy_signal
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import kagglehub
import shutil
from pathlib import Path
import numpy as np
import pandas as pd





def preprocess_phm2010(
    c_files=["c1", "c4"],
    columns=[2],
    out_dir="./phm2010_preprocessed",
    reset=False
):
    """
    Preprocess PHM2010 dataset.
    
    Args:
        c_files: List of C files to process, e.g. ["c1", "c2"]
        columns: List of column indices to extract (0-6):
                 0=Fx, 1=Fy, 2=Fz, 3=Vx, 4=Vy, 5=Vz, 6=AE
        out_dir: Output directory path
    
    Returns:
        manifest: DataFrame with metadata for all processed files
    """
    ds_path = Path(kagglehub.dataset_download("rabahba/phm-data-challenge-2010"))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if reset and out_path.exists():
        shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"Reset output directory: {out_path}")
    
    col_names = ["Fx_N", "Fy_N", "Fz_N", "Vx_g", "Vy_g", "Vz_g", "AE_V"]
    all_rows = []
    
    for c_file in c_files:
        c_dir = ds_path / c_file / c_file
        wear_csv = ds_path / c_file / f"{c_file}_wear.csv"
        
        if not wear_csv.exists():
            continue
        
        wear = pd.read_csv(wear_csv)
        csvs = sorted([p for p in c_dir.iterdir() if p.suffix == ".csv"])
        
        for csv_path in csvs:
            cut_num = int(re.search(r"_(\d{3})\.csv$", csv_path.name).group(1))
            data = pd.read_csv(csv_path, header=None, usecols=columns)
            
            npz_path = out_path / f"{c_file}_cut{cut_num:03d}.npz"
            arrays = {col_names[col]: data.iloc[:, i].values.astype("float32")
                     for i, col in enumerate(columns)}
            np.savez_compressed(npz_path, **arrays)
            
            row = {"c_file": c_file, "cut": cut_num, "npz_path": str(npz_path)}
            wear_row = wear[wear["cut"] == cut_num]
            if not wear_row.empty:
                for col in wear.columns:
                    if col != "cut":
                        row[col] = wear_row.iloc[0][col]
            all_rows.append(row)
    
    manifest = pd.DataFrame(all_rows).sort_values(["c_file", "cut"])
    manifest_path = out_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    
    return manifest


def add_wear_stage(manifest, labels=None):
    """
    Add wear stage column to manifest.
    
    Args:
        manifest: DataFrame from preprocess_phm2010
        labels: List of tuples (stage_name, min_cut, max_cut)
                Default: [("initial", 1, 67), ("stable", 68, 239), ("rapid", 240, 315)]
    
    Returns:
        manifest with added 'stage' column
    """
    if labels is None:
        labels = [
            ("initial", 1, 67),
            ("stable", 68, 239),
            ("rapid", 240, 315),
        ]
    
    def get_stage(cut_num):
        for name, lo, hi in labels:
            if lo <= cut_num <= hi:
                return name
        return "unlabeled"
    
    manifest["stage"] = manifest["cut"].apply(get_stage)
    return manifest


# ==============================================================================
# TRANSFORM FUNCTIONS
# ==============================================================================

def cwt_transform(signal, scales, wavelet='morl'):
    """Apply Continuous Wavelet Transform to a signal."""
    coeffs, _ = pywt.cwt(signal, scales, wavelet)
    return np.abs(coeffs).astype('float32')


def stft_transform(signal, fs=1.0, window='hann', nperseg=256, noverlap=None, 
                   nfft=None, log_scale=False):
    """
    Apply Short-Time Fourier Transform to a signal.
    
    Args:
        signal: 1D numpy array
        fs: Sampling frequency
        window: Window function
        nperseg: Length of each segment
        noverlap: Number of points to overlap (default: nperseg // 2)
        nfft: Length of FFT (default: nperseg)
        log_scale: If True, return 20*log10(magnitude) in dB
    
    Returns:
        2D numpy array (frequency x time) of magnitudes (or log magnitudes if log_scale=True)
    """
    _, _, Zxx = scipy_signal.stft(signal, fs=fs, window=window, nperseg=nperseg,
                                   noverlap=noverlap, nfft=nfft)
    magnitude = np.abs(Zxx)
    
    if log_scale:
        magnitude = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
    
    return magnitude.astype('float32')


# ==============================================================================
# CORE PROCESSING PIPELINE
# ==============================================================================

def process_windows(manifest,
                   window_length=1000,
                   overlap=0,
                   slice_start=0,
                   slice_end=20000,
                   transform=None,
                   transform_params=None,
                   c_files=None,
                   cut_numbers=None):
    """
    Core pipeline: Filter → Load → Window → Transform (optional)
    This is the SINGLE source of truth for processing PHM2010 data.
    
    Args:
        manifest: DataFrame from preprocess_phm2010
        window_length: Length of each window
        overlap: Overlap fraction (0-1)
        slice_start: Start index for slicing signal
        slice_end: End index for slicing signal
        transform: None (raw), 'cwt', or 'stft'
        transform_params: Dict of transform parameters
            For CWT: {'scales': array, 'wavelet': str}
            For STFT: {'fs': float, 'window': str, 'nperseg': int, 'noverlap': int, 'nfft': int}
        c_files: List of C files to process (None = all)
        cut_numbers: List of cut numbers to process (None = all)
    
    Yields:
        Dict with keys: c_file, cut, stage, window_idx, data, column, transform_type
        where 'data' is either raw signal or transformed result
    """
    # Set default transform params
    if transform == 'cwt' and transform_params is None:
        transform_params = {'scales': np.arange(1, 128), 'wavelet': 'morl'}
    elif transform == 'stft' and transform_params is None:
        transform_params = {'fs': 50000, 'nperseg': 256}
    
    # Calculate step size for windowing
    step = int(window_length * (1 - overlap))
    
    # Filter manifest
    filtered = manifest.copy()
    if c_files:
        filtered = filtered[filtered["c_file"].isin(c_files)]
    if cut_numbers:
        filtered = filtered[filtered["cut"].isin(cut_numbers)]
    
    # Process each cut
    for row in filtered.itertuples():
        data = np.load(row.npz_path)
        
        # Process each column (Fx, Fy, Fz, Vx, Vy, Vz, AE)
        for col_name in data.files:
            signal = data[col_name][slice_start:slice_end]
            
            # Extract windows
            for idx, i in enumerate(range(0, len(signal) - window_length + 1, step)):
                window = signal[i:i + window_length]
                
                # Apply transform if specified
                if transform is None:
                    result = window
                elif transform == 'cwt':
                    result = cwt_transform(
                        window,
                        transform_params['scales'],
                        transform_params.get('wavelet', 'morl')
                    )
                elif transform == 'stft':
                    result = stft_transform(
                        window,
                        fs=transform_params.get('fs', 1.0),
                        window=transform_params.get('window', 'hann'),
                        nperseg=transform_params.get('nperseg', 256),
                        noverlap=transform_params.get('noverlap'),
                        nfft=transform_params.get('nfft'),
                        log_scale=transform_params.get('log_scale', False)
                    )
                else:
                    raise ValueError(f"Unknown transform: {transform}")
                
                yield {
                    "c_file": row.c_file,
                    "cut": row.cut,
                    "stage": row.stage,
                    "window_idx": idx,
                    "data": result,
                    "column": col_name,
                    "transform_type": transform
                }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_transforms(manifest,
                   window_length=1000,
                   overlap=0,
                   slice_start=0,
                   slice_end=20000,
                   transform=None,
                   transform_params=None,
                   c_files=None,
                   cut_numbers=None,
                   max_windows=5,
                   max_freq=None,
                   log_freq_axis=False,
                   figsize_per_row=15):
    """
    Unified plotting function for raw signals and transforms.
    
    Args:
        (same as process_windows, plus:)
        max_windows: Maximum windows to plot per cut
        max_freq: Maximum frequency to display (STFT only)
        log_freq_axis: If True, use log scale for frequency axis (STFT only, for visualization)
        figsize_per_row: Figure width scale
    """
    # Collect data
    data_list = list(process_windows(
        manifest, window_length, overlap, slice_start, slice_end,
        transform, transform_params, c_files, cut_numbers
    ))
    
    if not data_list:
        print("No data to plot")
        return
    
    # Group by cut
    cuts = sorted(set((d["c_file"], d["cut"]) for d in data_list))
    
    # Create subplots
    fig, axes = plt.subplots(len(cuts), max_windows,
                        figsize=(figsize_per_row, 3*len(cuts)))
    
    # fig, axes = plt.subplots(len(cuts), max_windows,
    #                     figsize=(10, 2*len(cuts)))
    if len(cuts) == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each cut
    for i, (c_file, cut) in enumerate(cuts):
        windows = [d for d in data_list
                  if d["c_file"] == c_file and d["cut"] == cut][:max_windows]
        
        for j, window_data in enumerate(windows):
            ax = axes[i, j]
            data = window_data["data"]
            
            # Plot based on transform type
            if transform is None:
                # Raw signal
                ax.plot(data, linewidth=0.5)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
            
            elif transform == 'cwt':
                # CWT
                ax.imshow(data, aspect='auto', cmap='viridis', origin='lower')
                ax.set_xlabel("Time")
                ax.set_ylabel("Scale")
            
            elif transform == 'stft':
                # STFT - calculate extents
                fs = transform_params.get('fs', 50000)
                nperseg = transform_params.get('nperseg', 256)
                noverlap = transform_params.get('noverlap', nperseg // 2)
                
                n_samples = data.shape[1] * (nperseg - noverlap) + noverlap
                time_extent = n_samples / fs
                freq_extent = fs / 2
                
                # Set frequency limits
                min_freq = 100 if log_freq_axis else 0
                
                ax.imshow(data, aspect='auto', cmap='viridis', origin='lower',
                         extent=[0, time_extent, min_freq or 0, freq_extent])
                
                # Apply log scale to frequency axis if requested (visualization only)
                if log_freq_axis:
                    ax.set_yscale('log')
                
                # Set frequency limits
                if max_freq is not None:
                    ax.set_ylim(min_freq or 0, min(max_freq, freq_extent))
                elif log_freq_axis:
                    ax.set_ylim(min_freq, freq_extent)
                
                ylabel = "Frequency (Hz"
                if log_freq_axis:
                    ylabel += ", log axis"
                ylabel += ")"
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(ylabel)
            
            # Set title
            stage = window_data.get('stage', 'unknown')
            ax.set_title(f"{c_file.upper()} Cut{cut} ({stage}) W{window_data['window_idx']}")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nTotal windows: {len(data_list)}")
    if transform:
        print(f"Transform shape: {data_list[0]['data'].shape}")
    print(f"Columns: {sorted(set(d['column'] for d in data_list))}")
    print(f"C files: {sorted(set(d['c_file'] for d in data_list))}")
    print(f"Cuts: {sorted(set(d['cut'] for d in data_list))}")
    print(f"Stages: {sorted(set(d['stage'] for d in data_list))}")


# ==============================================================================
# DATASET EXPORT
# ==============================================================================

def export_dataset(manifest,
                  output_folder,
                  window_length=15000,
                  overlap=0,
                  slice_start=2000,
                  slice_end=200000,
                  transform='stft',
                  transform_params=None,
                  c_files=None,
                  cut_numbers=None,
                  max_freq=None,
                  dpi=200):
    """
    Export transformed data as images for ViT training.
    Images are saved as: {output_folder}/{stage}/{c_file}_cut{cut}_w{idx}_{column}.png
    
    Args:
        manifest: DataFrame from preprocess_phm2010
        output_folder: Root folder for saving images
        (rest same as process_windows)
        max_freq: Maximum frequency to display (STFT only)
        dpi: Image resolution
    """
    if transform_params is None:
        if transform == 'stft':
            transform_params = {'fs': 50000, 'nperseg': 256}
        elif transform == 'cwt':
            transform_params = {'scales': np.arange(1, 128), 'wavelet': 'morl'}
    
    # Get fs and nperseg for extent calculation (STFT only)
    fs = transform_params.get('fs', 50000)
    nperseg = transform_params.get('nperseg', 256)
    noverlap = transform_params.get('noverlap', nperseg // 2)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Count total for progress bar
    temp_list = list(process_windows(
        manifest, window_length, overlap, slice_start, slice_end,
        transform, transform_params, c_files, cut_numbers
    ))
    total_count = len(temp_list)
    
    # Process and save
    saved_count = 0
    for window_data in tqdm(temp_list, desc="Exporting images", total=total_count):
        # Create stage folder
        stage = window_data['stage']
        stage_folder = os.path.join(output_folder, stage)
        os.makedirs(stage_folder, exist_ok=True)
        
        # Generate filename
        filename = (f"{window_data['c_file']}_cut{window_data['cut']:03d}_"
                   f"w{window_data['window_idx']}_{window_data['column']}.png")
        filepath = os.path.join(stage_folder, filename)
        
        # Get data
        data = window_data['data']
        
        # Create figure
        plt.figure(figsize=(5, 5))
        if transform == 'stft':
            # Calculate extents for STFT
            h, w = data.shape
            freq_extent = fs / 2
            n_samples = w * (nperseg - noverlap) + noverlap
            time_extent = n_samples / fs
            
            plt.imshow(data, aspect='auto', cmap='viridis', origin='lower',
                      extent=[0, time_extent, 0, freq_extent])
            if max_freq is not None:
                plt.ylim(0, min(max_freq, freq_extent))
        else:
            # CWT or raw
            plt.imshow(data, aspect='auto', cmap='viridis', origin='lower')
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
        
        saved_count += 1
    
    print(f"\n✓ Saved {saved_count} images to {output_folder}")
    print(f"  Stages: {sorted(set(d['stage'] for d in temp_list))}")


##############################TEST##############################


def process_windows_2(manifest,
                      window_length=1000,
                      overlap=0,
                      slice_start=0,
                      slice_end=20000,
                      transform=None,
                      transform_params=None,
                      c_files=None,
                      cut_numbers=None,
                      tf_input_col=None,
                      tf_output_col=None):
    """
    Process windows with transfer function computation.
    Computes TF magnitude between input/output signal pairs, then applies transform.
    
    Args:
        (standard args same as process_windows)
        tf_input_col: Input column name (e.g., 'Fz_N')
        tf_output_col: Output column name (e.g., 'Vz_g')
    
    Yields:
        Same format as process_windows, but data is TF magnitude (or transformed TF)
    """
    #  default transform params
    if transform == 'cwt' and transform_params is None:
        transform_params = {'scales': np.arange(1, 128), 'wavelet': 'morl'}
    elif transform == 'stft' and transform_params is None:
        transform_params = {'fs': 50000, 'nperseg': 256}
    
    #  step size for windowing
    step = int(window_length * (1 - overlap))
    
    # Filter manifest
    filtered = manifest.copy()
    if c_files:
        filtered = filtered[filtered["c_file"].isin(c_files)]
    if cut_numbers:
        filtered = filtered[filtered["cut"].isin(cut_numbers)]
    
    # Process each cut
    for row in filtered.itertuples():
        data = np.load(row.npz_path)
        
        # Check both columns exist
        if tf_input_col not in data.files or tf_output_col not in data.files:
            continue
        
        # Load both signals
        input_signal = data[tf_input_col][slice_start:slice_end]
        output_signal = data[tf_output_col][slice_start:slice_end]
        
        # Extract windows from both signals
        for idx, i in enumerate(range(0, len(input_signal) - window_length + 1, step)):
            input_window = input_signal[i:i + window_length]
            output_window = output_signal[i:i + window_length]
            
            # Compute transfer function: H(f) = FFT(output) / FFT(input)
            fft_input = np.fft.rfft(input_window)
            fft_output = np.fft.rfft(output_window)
            tf = fft_output / (fft_input + 1e-10)  #division by zero
            tf_magnitude = np.abs(tf)  #magnitude
            
            # Apply transform if specified
            if transform is None:
                result = tf_magnitude
            elif transform == 'cwt':
                result = cwt_transform(
                    tf_magnitude,
                    scales=transform_params.get('scales', np.arange(1, 128)),
                    wavelet=transform_params.get('wavelet', 'morl')
                )
            elif transform == 'stft':
                result = stft_transform(
                    tf_magnitude,
                    fs=transform_params.get('fs', 1.0),
                    window=transform_params.get('window', 'hann'),
                    nperseg=transform_params.get('nperseg', 256),
                    noverlap=transform_params.get('noverlap', None),
                    nfft=transform_params.get('nfft', None),
                    log_scale=transform_params.get('log_scale', False)
                )
            else:
                raise ValueError(f"Unknown transform: {transform}")
            
            yield {
                "c_file": row.c_file,
                "cut": row.cut,
                "stage": row.stage,
                "window_idx": idx,
                "data": result,
                "column": f"{tf_input_col}_to_{tf_output_col}",
                "transform_type": transform
            }



def plot_transforms_2(manifest,
                   window_length=1000,
                   overlap=0,
                   slice_start=0,
                   slice_end=20000,
                   transform=None,
                   transform_params=None,
                   c_files=None,
                   cut_numbers=None,
                   max_windows=5,
                   max_freq=None,
                   log_freq_axis=False,
                   figsize_per_row=15,
                   process_fn=None,
                   **kwargs):
    """
    Unified plotting function for raw signals and transforms.
    
    Args:
        (same as process_windows, plus:)
        max_windows: Maximum windows to plot per cut
        max_freq: Maximum frequency to display (STFT only)
        log_freq_axis: If True, use log scale for frequency axis (STFT only, for visualization)
        figsize_per_row: Figure width scale
        process_fn: Processing function to use (default: process_windows)
        **kwargs: Additional arguments passed to process_fn (e.g., tf_input_col, tf_output_col)
    """
    # Default to process_windows if not specified
    if process_fn is None:
        process_fn = process_windows
    
    # Collect data
    data_list = list(process_fn(
        manifest, window_length, overlap, slice_start, slice_end,
        transform, transform_params, c_files, cut_numbers, **kwargs
    ))
    
    if not data_list:
        print("No data to plot")
        return
    
    # Group by cut
    cuts = sorted(set((d["c_file"], d["cut"]) for d in data_list))
    
    # Create subplots
    fig, axes = plt.subplots(len(cuts), max_windows,
                        figsize=(figsize_per_row, 3*len(cuts)))
    
    if len(cuts) == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each cut
    for i, (c_file, cut) in enumerate(cuts):
        windows = [d for d in data_list
                  if d["c_file"] == c_file and d["cut"] == cut][:max_windows]
        
        for j, window_data in enumerate(windows):
            ax = axes[i, j]
            data = window_data["data"]
            
            # Plot based on transform type
            if transform is None:
                # Raw signal
                ax.plot(data, linewidth=0.5)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
            
            elif transform == 'cwt':
                # CWT
                ax.imshow(data, aspect='auto', cmap='viridis', origin='lower')
                ax.set_xlabel("Time")
                ax.set_ylabel("Scale")
            
            elif transform == 'stft':
                # STFT - calculate extents
                fs = transform_params.get('fs', 50000)
                nperseg = transform_params.get('nperseg', 256)
                noverlap = transform_params.get('noverlap', nperseg // 2)
                
                n_samples = data.shape[1] * (nperseg - noverlap) + noverlap
                time_extent = n_samples / fs
                freq_extent = fs / 2
                
                # Set frequency limits
                min_freq = 100 if log_freq_axis else 0
                
                ax.imshow(data, aspect='auto', cmap='viridis', origin='lower',
                         extent=[0, time_extent, min_freq or 0, freq_extent])
                
                # Apply log scale to frequency axis if requested (visualization only)
                if log_freq_axis:
                    ax.set_yscale('log')
                
                # Set frequency limits
                if max_freq is not None:
                    ax.set_ylim(min_freq or 0, min(max_freq, freq_extent))
                elif log_freq_axis:
                    ax.set_ylim(min_freq, freq_extent)
                
                ylabel = "Frequency (Hz"
                if log_freq_axis:
                    ylabel += ", log axis"
                ylabel += ")"
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(ylabel)
            
            # Set title
            stage = window_data.get('stage', 'unknown')
            ax.set_title(f"{c_file.upper()} Cut{cut} ({stage}) W{window_data['window_idx']}")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTotal windows: {len(data_list)}")
    if transform:
        print(f"Transform shape: {data_list[0]['data'].shape}")
    print(f"Columns: {sorted(set(d['column'] for d in data_list))}")
    print(f"C files: {sorted(set(d['c_file'] for d in data_list))}")
    print(f"Cuts: {sorted(set(d['cut'] for d in data_list))}")
    print(f"Stages: {sorted(set(d['stage'] for d in data_list))}")