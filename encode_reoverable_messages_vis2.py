"""
Improved 3D visualization with variable envelope frequency support
Handles both fixed and variable frequency encoding
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import glob

def create_improved_3d_visualization():
    """Create multiple visualization styles for clearer sideband analysis"""
    
    json_files = sorted(glob.glob('nanogpt_*_data.json'))
    
    if len(json_files) < 2:
        print("Need at least 2 datasets. Generate them first.")
        return
    
    print(f"Loading {len(json_files)} datasets...")
    
    topics = []
    all_spectra = []
    correlations = []
    envelope_freqs = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        topic = result.get('topic', json_file.replace('nanogpt_', '').replace('_data.json', ''))
        topics.append(topic)
        correlations.append(result.get('correlation', 0))
        envelope_freqs.append(result.get('envelope_frequency', 0.05))  # Get envelope freq
        
        # Compute FFT with proper normalization
        signal = np.array(result['actual_diversities'])
        n = len(signal)
        
        # Remove DC component and normalize
        signal_centered = signal - np.mean(signal)
        signal_normalized = signal_centered / np.std(signal_centered)
        
        # Apply window to reduce spectral leakage
        window = np.hanning(n)
        signal_windowed = signal_normalized * window
        
        # Compute FFT
        fft_result = np.fft.fft(signal_windowed)
        freqs = np.fft.fftfreq(n, d=1.0)
        
        # Get positive frequencies only
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        all_spectra.append((pos_freqs, pos_magnitudes))
        print(f"  {topic}: r={correlations[-1]:.3f}, env_freq={envelope_freqs[-1]:.3f}, max_magnitude={np.max(pos_magnitudes):.1f}")
    
    # Carrier frequency (constant)
    expected_carrier = 1.0 / 3.0
    
    # ========================================================================
    # Visualization 1: Individual 2D plots (clearest for sideband inspection)
    # ========================================================================
    
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, ((freqs, mags), topic, corr, env_freq) in enumerate(zip(all_spectra, topics, correlations, envelope_freqs)):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # Calculate expected sidebands for THIS dataset
        expected_lower = expected_carrier - env_freq
        expected_upper = expected_carrier + env_freq
        
        # Plot full spectrum
        ax.plot(freqs, mags, 'b-', linewidth=1.5, alpha=0.7, label='Spectrum')
        
        # Mark expected peaks with vertical lines (specific to this envelope freq)
        ax.axvline(expected_lower, color='purple', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Lower SB ({expected_lower:.3f})')
        ax.axvline(expected_carrier, color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Carrier ({expected_carrier:.3f})')
        ax.axvline(expected_upper, color='orange', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Upper SB ({expected_upper:.3f})')
        
        # Find and mark actual peaks near expected locations
        for expected_freq, color, name in [
            (expected_lower, 'purple', 'Lower'),
            (expected_carrier, 'red', 'Carrier'),
            (expected_upper, 'orange', 'Upper')
        ]:
            # Find peak within ±0.02 of expected
            mask = np.abs(freqs - expected_freq) < 0.02
            if np.any(mask):
                local_freqs = freqs[mask]
                local_mags = mags[mask]
                peak_idx = np.argmax(local_mags)
                peak_freq = local_freqs[peak_idx]
                peak_mag = local_mags[peak_idx]
                ax.plot(peak_freq, peak_mag, 'o', color=color, 
                       markersize=10, markeredgecolor='black', markeredgewidth=1.5)
                ax.text(peak_freq, peak_mag * 1.1, f'{peak_mag:.1f}', 
                       ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Frequency (cycles/step)', fontsize=11, fontweight='bold')
        ax.set_ylabel('FFT Magnitude', fontsize=11, fontweight='bold')
        ax.set_title(f'{topic}\n(correlation = {corr:.3f}, env_freq = {env_freq:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim(0, 0.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sidebands_2d_grid.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: sidebands_2d_grid.png (clearest view)")
    plt.close()
    
    # ========================================================================
    # Visualization 2: Zoomed 3D waterfall (carrier region only)
    # ========================================================================
    
    # Create meshgrid for carrier region only
    freq_min, freq_max = 0.2, 0.5
    
    # Extract carrier region
    carrier_spectra = []
    carrier_freqs = None
    
    for freqs, mags in all_spectra:
        mask = (freqs >= freq_min) & (freqs <= freq_max)
        if carrier_freqs is None:
            carrier_freqs = freqs[mask]
        carrier_spectra.append(mags[mask])
    
    # Ensure all same length
    min_carrier_len = min(len(s) for s in carrier_spectra)
    X = carrier_freqs[:min_carrier_len]
    Y = np.arange(len(topics))
    Z = np.array([s[:min_carrier_len] for s in carrier_spectra])
    
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Create 3D figure
    fig2 = plt.figure(figsize=(16, 12))
    ax = fig2.add_subplot(111, projection='3d')
    
    # Surface plot with better colormap
    surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='plasma', 
                          alpha=0.8, edgecolor='k', linewidth=0.2)
    
    # Mark peaks for each dataset with their specific envelope frequency
    for y_idx, env_freq in enumerate(envelope_freqs):
        expected_lower = expected_carrier - env_freq
        expected_upper = expected_carrier + env_freq
        
        for peak_freq, color in [
            (expected_lower, 'purple'),
            (expected_carrier, 'red'),
            (expected_upper, 'orange')
        ]:
            if freq_min <= peak_freq <= freq_max:
                peak_idx = np.argmin(np.abs(X - peak_freq))
                # Single point marker
                ax.scatter([peak_freq], [y_idx], [Z[y_idx, peak_idx]], 
                          color=color, s=100, alpha=0.9, edgecolors='black', linewidth=2)
    
    # Individual traces
    for idx, s in enumerate(carrier_spectra):
        s_truncated = s[:len(X)]
        ax.plot(X, [idx]*len(X), s_truncated, 'k-', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frequency (cycles/step)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Message/Dataset', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel('FFT Magnitude', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('3D Frequency Spectrum: Variable Envelope Frequencies\n(Each message has unique sideband spacing)', 
                fontsize=15, fontweight='bold', pad=20)
    
    ax.set_yticks(Y)
    ax.set_yticklabels([f"{t[:20]}\n(r={c:.2f}, f={ef:.2f})" 
                       for t, c, ef in zip(topics, correlations, envelope_freqs)], fontsize=8)
    ax.set_xlim(freq_min, freq_max)
    ax.grid(True, alpha=0.2)
    
    # Better viewing angle
    ax.view_init(elev=20, azim=135)
    
    # Add colorbar
    fig2.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Magnitude')
    
    plt.tight_layout()
    plt.savefig('sidebands_3d_zoomed_static.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sidebands_3d_zoomed_static.png (3D carrier region)")
    plt.close()
    
    # ========================================================================
    # Visualization 3: Animated rotating 3D (carrier region)
    # ========================================================================
    
    fig3 = plt.figure(figsize=(16, 12))
    ax = fig3.add_subplot(111, projection='3d')
    
    def animate(frame):
        ax.clear()
        
        # Surface
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='plasma', 
                              alpha=0.8, edgecolor='k', linewidth=0.2)
        
        # Peak markers for each dataset
        for y_idx, env_freq in enumerate(envelope_freqs):
            expected_lower = expected_carrier - env_freq
            expected_upper = expected_carrier + env_freq
            
            for peak_freq, color in [
                (expected_lower, 'purple'),
                (expected_carrier, 'red'),
                (expected_upper, 'orange')
            ]:
                if freq_min <= peak_freq <= freq_max:
                    peak_idx = np.argmin(np.abs(X - peak_freq))
                    ax.scatter([peak_freq], [y_idx], [Z[y_idx, peak_idx]], 
                             color=color, s=80, alpha=0.9, edgecolors='black', linewidth=1.5)
        
        # Traces
        for idx, s in enumerate(carrier_spectra):
            s_truncated = s[:len(X)]
            ax.plot(X, [idx]*len(X), s_truncated, 'k-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Frequency (cycles/step)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Dataset', fontsize=13, fontweight='bold')
        ax.set_zlabel('FFT Magnitude', fontsize=13, fontweight='bold')
        ax.set_title('3D Spectrum: Variable Envelope Frequencies (Rotating)', 
                    fontsize=15, fontweight='bold')
        
        ax.set_yticks(Y)
        ax.set_yticklabels([f"{t[:15]}" for t in topics], fontsize=8)
        ax.set_xlim(freq_min, freq_max)
        ax.grid(True, alpha=0.2)
        
        # Rotate
        ax.view_init(elev=20, azim=frame)
        
        return surf,
    
    print("\nCreating animated GIF...")
    anim = FuncAnimation(fig3, animate, frames=np.arange(0, 360, 2), 
                        interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save('sidebands_3d_rotating.gif', writer=writer, dpi=100)
    print("✓ Saved: sidebands_3d_rotating.gif")
    plt.close()
    
    # ========================================================================
    # Visualization 4: Peak magnitude comparison
    # ========================================================================
    
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract peak magnitudes (using variable envelope frequencies)
    peak_data = {
        'Lower SB': [],
        'Carrier': [],
        'Upper SB': []
    }
    
    for (freqs, mags), env_freq in zip(all_spectra, envelope_freqs):
        expected_lower = expected_carrier - env_freq
        expected_upper = expected_carrier + env_freq
        
        for expected_freq, name in [
            (expected_lower, 'Lower SB'),
            (expected_carrier, 'Carrier'),
            (expected_upper, 'Upper SB')
        ]:
            mask = np.abs(freqs - expected_freq) < 0.02
            if np.any(mask):
                peak_mag = np.max(mags[mask])
                peak_data[name].append(peak_mag)
            else:
                peak_data[name].append(0)
    
    # Bar chart
    x = np.arange(len(topics))
    width = 0.25
    
    ax1.bar(x - width, peak_data['Lower SB'], width, label='Lower SB', color='purple', alpha=0.8)
    ax1.bar(x, peak_data['Carrier'], width, label='Carrier', color='red', alpha=0.8)
    ax1.bar(x + width, peak_data['Upper SB'], width, label='Upper SB', color='orange', alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peak Magnitude', fontsize=12, fontweight='bold')
    ax1.set_title('Peak Magnitudes Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t[:15] for t in topics], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Sideband ratios
    ratios_lower = [l/c if c > 0 else 0 for l, c in zip(peak_data['Lower SB'], peak_data['Carrier'])]
    ratios_upper = [u/c if c > 0 else 0 for u, c in zip(peak_data['Upper SB'], peak_data['Carrier'])]
    
    ax2.bar(x - width/2, ratios_lower, width, label='Lower SB / Carrier', color='purple', alpha=0.8)
    ax2.bar(x + width/2, ratios_upper, width, label='Upper SB / Carrier', color='orange', alpha=0.8)
    ax2.axhline(y=0.32, color='green', linestyle='--', linewidth=2, label='Expected (~0.32)')
    
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sideband / Carrier Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Sideband Symmetry Analysis', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t[:15] for t in topics], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('sideband_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sideband_analysis.png (peak analysis)")
    plt.close()
    
    # ========================================================================
    # Visualization 5: Envelope frequency comparison
    # ========================================================================
    
    fig5, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot envelope frequencies
    x = np.arange(len(topics))
    bars = ax.bar(x, envelope_freqs, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Color code by frequency
    for bar, freq in zip(bars, envelope_freqs):
        if freq < 0.06:
            bar.set_color('lightcoral')
        elif freq < 0.10:
            bar.set_color('lightgreen')
        else:
            bar.set_color('lightblue')
    
    ax.set_xlabel('Message/Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Envelope Frequency (cycles/step)', fontsize=12, fontweight='bold')
    ax.set_title('Frequency-Division Encoding: Each Message Has Unique Envelope Frequency', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t[:20] for t in topics], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (freq, corr) in enumerate(zip(envelope_freqs, correlations)):
        ax.text(i, freq + 0.005, f'{freq:.3f}\nr={corr:.2f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('envelope_frequency_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: envelope_frequency_comparison.png (frequency encoding)")
    plt.close()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. sidebands_2d_grid.png - Individual spectra with correct sideband markers")
    print("  2. sidebands_3d_zoomed_static.png - 3D view showing variable sideband spacing")
    print("  3. sidebands_3d_rotating.gif - Animated rotation")
    print("  4. sideband_analysis.png - Peak magnitude comparison")
    print("  5. envelope_frequency_comparison.png - Shows frequency-division encoding")
    print("\nEach message has unique sideband spacing based on its envelope frequency")
    print("This demonstrates frequency-division multiplexing for message encoding")


if __name__ == "__main__":
    create_improved_3d_visualization()
