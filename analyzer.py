import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

class LexicalDiversityModulator:
    """
    Modulates lexical diversity (TTR) in text generation to create FFT-detectable patterns.
    Based on the paper's continuous feature modulation approach.
    """
    
    def __init__(self, n_sentences=100, carrier_freq=1/3, envelope_freq=0.05, 
                 modulation_depth=0.6, ttr_min=0.4, ttr_max=0.9):
        """
        Args:
            n_sentences: Number of sentences to generate
            carrier_freq: Carrier frequency in cycles/step (default: 0.333)
            envelope_freq: Envelope frequency in cycles/step (default: 0.05)
            modulation_depth: Modulation depth (default: 0.6)
            ttr_min: Minimum target TTR (default: 0.4)
            ttr_max: Maximum target TTR (default: 0.9)
        """
        self.n_sentences = n_sentences
        self.carrier_freq = carrier_freq
        self.envelope_freq = envelope_freq
        self.modulation_depth = modulation_depth
        self.ttr_min = ttr_min
        self.ttr_max = ttr_max
        
        # Generate target TTR sequence
        self.target_ttr = self._generate_target_ttr()
        
    def _generate_target_ttr(self):
        """Generate the target TTR values following AM pattern."""
        n = np.arange(self.n_sentences)
        
        # Carrier signal
        carrier = np.cos(2 * np.pi * self.carrier_freq * n)
        
        # Envelope signal
        envelope = 1 + self.modulation_depth * np.cos(2 * np.pi * self.envelope_freq * n)
        
        # Modulated signal
        modulated = carrier * envelope
        
        # Map to TTR range [ttr_min, ttr_max]
        # Normalize modulated signal from [-1-d, 1+d] to [0, 1]
        normalized = (modulated + (1 + self.modulation_depth)) / (2 * (1 + self.modulation_depth))
        
        # Scale to TTR range
        target_ttr = self.ttr_min + normalized * (self.ttr_max - self.ttr_min)
        
        return target_ttr
    
    def calculate_ttr(self, text):
        """Calculate type-token ratio for a piece of text."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        unique_words = len(set(words))
        return unique_words / len(words)
    
    def get_target_ttr(self, step):
        """Get the target TTR for a given step."""
        return self.target_ttr[step]
    
    def generate_prompt_for_step(self, step, topic="general discussion"):
        """
        Generate a prompt that guides the model toward the target TTR.
        
        Args:
            step: Current sentence number (0 to n_sentences-1)
            topic: Topic context for generation
        """
        target = self.target_ttr[step]
        
        if target < 0.6:
            # Low diversity: simple, repetitive
            style = "using simple words and repeating key terms frequently"
            example = "Use basic vocabulary. Repeat important words. Keep it simple."
        else:
            # High diversity: complex, varied
            style = "using sophisticated vocabulary with diverse, unique words"
            example = "Employ elaborate terminology. Utilize distinctive expressions. Demonstrate lexical variety."
        
        prompt = f"Write one sentence about {topic}, {style}. Sentence {step+1}:"
        
        return prompt, target
    
    def validate_generation(self, generated_texts):
        """
        Validate that generated texts follow the target TTR pattern.
        
        Args:
            generated_texts: List of generated sentences
            
        Returns:
            actual_ttr: Array of actual TTR values
            correlation: Correlation between target and actual
        """
        actual_ttr = np.array([self.calculate_ttr(text) for text in generated_texts])
        correlation = np.corrcoef(self.target_ttr[:len(actual_ttr)], actual_ttr)[0, 1]
        
        return actual_ttr, correlation
    
    def analyze_spectrum(self, actual_ttr):
        """
        Perform FFT analysis on the actual TTR values.
        
        Args:
            actual_ttr: Array of actual TTR values
            
        Returns:
            frequencies: Frequency array
            magnitude: FFT magnitude spectrum
            detected_peaks: Dictionary of detected peaks
        """
        # Normalize
        normalized = (actual_ttr - np.mean(actual_ttr)) / np.std(actual_ttr)
        
        # Compute FFT
        fft_result = np.fft.fft(normalized)
        n = len(normalized)
        
        # Get positive frequencies only
        frequencies = np.fft.fftfreq(n)[:n//2]
        magnitude = np.abs(fft_result)[:n//2]
        
        # Detect peaks
        peaks = self._detect_peaks(frequencies, magnitude)
        
        return frequencies, magnitude, peaks
    
    def _detect_peaks(self, frequencies, magnitude, threshold=0.1):
        """Detect spectral peaks above threshold."""
        # Find carrier peak
        max_idx = np.argmax(magnitude)
        carrier_freq = frequencies[max_idx]
        carrier_mag = magnitude[max_idx]
        
        peaks = {
            'carrier': {'frequency': carrier_freq, 'magnitude': carrier_mag}
        }
        
        # Look for sidebands
        expected_lower = self.carrier_freq - self.envelope_freq
        expected_upper = self.carrier_freq + self.envelope_freq
        
        # Find peaks near expected sideband locations
        for freq, mag in zip(frequencies, magnitude):
            if mag > threshold * carrier_mag:
                if abs(freq - expected_lower) < 0.02:
                    peaks['lower_sideband'] = {'frequency': freq, 'magnitude': mag}
                elif abs(freq - expected_upper) < 0.02:
                    peaks['upper_sideband'] = {'frequency': freq, 'magnitude': mag}
        
        return peaks
    
    def plot_results(self, actual_ttr, save_path=None):
        """
        Create comprehensive visualization of results.
        
        Args:
            actual_ttr: Array of actual TTR values
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Target vs Actual TTR
        ax = axes[0, 0]
        steps = np.arange(len(actual_ttr))
        ax.plot(steps, self.target_ttr[:len(actual_ttr)], 'b-', 
                label='Target TTR', linewidth=2, alpha=0.7)
        ax.plot(steps, actual_ttr, 'r--', 
                label='Actual TTR', linewidth=2, alpha=0.7)
        ax.set_xlabel('Sentence Number')
        ax.set_ylabel('Type-Token Ratio')
        ax.set_title('TTR Modulation Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: FFT Spectrum
        ax = axes[0, 1]
        frequencies, magnitude, peaks = self.analyze_spectrum(actual_ttr)
        ax.plot(frequencies, magnitude, 'b-', linewidth=2)
        
        # Mark detected peaks
        for peak_name, peak_data in peaks.items():
            ax.axvline(peak_data['frequency'], color='r', 
                      linestyle='--', alpha=0.5, 
                      label=f"{peak_name}: {peak_data['frequency']:.3f}")
        
        ax.set_xlabel('Frequency (cycles/step)')
        ax.set_ylabel('Magnitude')
        ax.set_title('FFT Spectrum of TTR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)
        
        # Plot 3: Correlation scatter
        ax = axes[1, 0]
        ax.scatter(self.target_ttr[:len(actual_ttr)], actual_ttr, alpha=0.5)
        
        # Add diagonal line
        min_val = min(self.target_ttr.min(), actual_ttr.min())
        max_val = max(self.target_ttr.max(), actual_ttr.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        correlation = np.corrcoef(self.target_ttr[:len(actual_ttr)], actual_ttr)[0, 1]
        ax.set_xlabel('Target TTR')
        ax.set_ylabel('Actual TTR')
        ax.set_title(f'Target vs Actual Correlation: {correlation:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Peak detection detail
        ax = axes[1, 1]
        # Zoom in on carrier region
        carrier_region = (frequencies > 0.2) & (frequencies < 0.45)
        ax.plot(frequencies[carrier_region], magnitude[carrier_region], 'b-', linewidth=2)
        
        # Expected peaks
        expected_peaks = {
            'Lower SB': self.carrier_freq - self.envelope_freq,
            'Carrier': self.carrier_freq,
            'Upper SB': self.carrier_freq + self.envelope_freq
        }
        
        for name, freq in expected_peaks.items():
            ax.axvline(freq, color='g', linestyle=':', alpha=0.5, label=f'{name} (exp)')
        
        ax.set_xlabel('Frequency (cycles/step)')
        ax.set_ylabel('Magnitude')
        ax.set_title('Sideband Detail View')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def print_detection_report(self, actual_ttr):
        """Print a detailed detection report."""
        frequencies, magnitude, peaks = self.analyze_spectrum(actual_ttr)
        correlation = np.corrcoef(self.target_ttr[:len(actual_ttr)], actual_ttr)[0, 1]
        
        print("=" * 60)
        print("LEXICAL DIVERSITY MODULATION DETECTION REPORT")
        print("=" * 60)
        print(f"\nSequence Length: {len(actual_ttr)} sentences")
        print(f"Target-Actual Correlation: {correlation:.4f}")
        print(f"\nExpected Carrier: {self.carrier_freq:.3f} cycles/step")
        print(f"Expected Lower Sideband: {self.carrier_freq - self.envelope_freq:.3f} cycles/step")
        print(f"Expected Upper Sideband: {self.carrier_freq + self.envelope_freq:.3f} cycles/step")
        
        print("\n" + "-" * 60)
        print("DETECTED PEAKS:")
        print("-" * 60)
        
        for peak_name, peak_data in peaks.items():
            freq = peak_data['frequency']
            mag = peak_data['magnitude']
            
            if peak_name == 'carrier':
                error = abs(freq - self.carrier_freq)
                print(f"\n{peak_name.upper()}:")
                print(f"  Frequency: {freq:.4f} cycles/step")
                print(f"  Magnitude: {mag:.2f}")
                print(f"  Detection Error: {error:.4f} cycles/step")
            else:
                ratio = mag / peaks['carrier']['magnitude']
                print(f"\n{peak_name.upper()}:")
                print(f"  Frequency: {freq:.4f} cycles/step")
                print(f"  Magnitude: {mag:.2f}")
                print(f"  Ratio to Carrier: {ratio:.3f}")
        
        print("\n" + "=" * 60)


# Example usage function
def example_usage():
    """
    Demonstrates how to use the modulator with nanoGPT.
    
    In practice, you would:
    1. Initialize the modulator
    2. For each step, get the target TTR and generate a prompt
    3. Use nanoGPT to generate text with that prompt
    4. Collect all generated sentences
    5. Analyze the results
    """
    
    # Initialize modulator
    modulator = LexicalDiversityModulator(
        n_sentences=100,
        carrier_freq=1/3,
        envelope_freq=0.05,
        modulation_depth=0.6,
        ttr_min=0.4,
        ttr_max=0.9
    )
    
    print("Initialized modulator with:")
    print(f"  Carrier frequency: {modulator.carrier_freq:.3f} cycles/step")
    print(f"  Envelope frequency: {modulator.envelope_freq:.3f} cycles/step")
    print(f"  Expected sidebands: {modulator.carrier_freq - modulator.envelope_freq:.3f}, "
          f"{modulator.carrier_freq + modulator.envelope_freq:.3f}")
    
    # Example: Generate prompts for first 5 steps
    print("\n" + "="*60)
    print("EXAMPLE PROMPTS FOR NANOGPT:")
    print("="*60)
    
    for step in range(5):
        prompt, target = modulator.generate_prompt_for_step(step, topic="artificial intelligence")
        print(f"\nStep {step}:")
        print(f"  Target TTR: {target:.3f}")
        print(f"  Prompt: {prompt}")
    
    print("\n" + "="*60)
    print("INTEGRATION WORKFLOW:")
    print("="*60)
    print("""
1. For each step from 0 to n_sentences-1:
   - Get prompt: prompt, target = modulator.generate_prompt_for_step(step, topic)
   - Generate with nanoGPT: sentence = model.generate(prompt)
   - Store sentence in list
   
2. After generation:
   - actual_ttr, correlation = modulator.validate_generation(sentences)
   - modulator.print_detection_report(actual_ttr)
   - fig = modulator.plot_results(actual_ttr, save_path='results.png')
   - plt.show()
    """)

if __name__ == "__main__":
    example_usage()
