"""
NanoGPT Integration for FFT-Detectable Text Generation
Requires: transformers, torch, numpy, matplotlib
Install: pip install transformers torch numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LexicalDiversityModulator:
    """Generates target TTR values with AM modulation pattern."""
    
    def __init__(self, n_sentences=100, carrier_freq=1/3, envelope_freq=0.05, 
                 modulation_depth=0.6, ttr_min=0.4, ttr_max=0.9):
        self.n_sentences = n_sentences
        self.carrier_freq = carrier_freq
        self.envelope_freq = envelope_freq
        self.modulation_depth = modulation_depth
        self.ttr_min = ttr_min
        self.ttr_max = ttr_max
        self.target_ttr = self._generate_target_ttr()
        
    def _generate_target_ttr(self):
        """Generate AM-modulated TTR sequence."""
        n = np.arange(self.n_sentences)
        carrier = np.cos(2 * np.pi * self.carrier_freq * n)
        envelope = 1 + self.modulation_depth * np.cos(2 * np.pi * self.envelope_freq * n)
        modulated = carrier * envelope
        normalized = (modulated + (1 + self.modulation_depth)) / (2 * (1 + self.modulation_depth))
        return self.ttr_min + normalized * (self.ttr_max - self.ttr_min)
    
    def calculate_ttr(self, text):
        """Calculate type-token ratio."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def get_target_ttr(self, step):
        """Get target TTR for given step."""
        return self.target_ttr[step]
    
    def analyze_spectrum(self, actual_ttr):
        """Perform FFT analysis."""
        normalized = (actual_ttr - np.mean(actual_ttr)) / np.std(actual_ttr)
        fft_result = np.fft.fft(normalized)
        n = len(normalized)
        frequencies = np.fft.fftfreq(n)[:n//2]
        magnitude = np.abs(fft_result)[:n//2]
        peaks = self._detect_peaks(frequencies, magnitude)
        return frequencies, magnitude, peaks
    
    def _detect_peaks(self, frequencies, magnitude, threshold=0.1):
        """Detect carrier and sideband peaks."""
        max_idx = np.argmax(magnitude)
        carrier_freq = frequencies[max_idx]
        carrier_mag = magnitude[max_idx]
        
        peaks = {'carrier': {'frequency': carrier_freq, 'magnitude': carrier_mag}}
        
        expected_lower = self.carrier_freq - self.envelope_freq
        expected_upper = self.carrier_freq + self.envelope_freq
        
        for freq, mag in zip(frequencies, magnitude):
            if mag > threshold * carrier_mag:
                if abs(freq - expected_lower) < 0.02:
                    peaks['lower_sideband'] = {'frequency': freq, 'magnitude': mag}
                elif abs(freq - expected_upper) < 0.02:
                    peaks['upper_sideband'] = {'frequency': freq, 'magnitude': mag}
        
        return peaks
    
    def plot_results(self, actual_ttr, save_path=None):
        """Create visualization of results."""
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
        min_val = min(self.target_ttr.min(), actual_ttr.min())
        max_val = max(self.target_ttr.max(), actual_ttr.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        correlation = np.corrcoef(self.target_ttr[:len(actual_ttr)], actual_ttr)[0, 1]
        ax.set_xlabel('Target TTR')
        ax.set_ylabel('Actual TTR')
        ax.set_title(f'Target vs Actual Correlation: {correlation:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Sideband detail
        ax = axes[1, 1]
        carrier_region = (frequencies > 0.2) & (frequencies < 0.45)
        ax.plot(frequencies[carrier_region], magnitude[carrier_region], 'b-', linewidth=2)
        
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
        """Print detailed detection report."""
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


class NanoGPTGenerator:
    """
    GPT-2 based text generator with TTR control.
    Uses constrained generation to match target lexical diversity.
    """
    
    def __init__(self, model_name='gpt2', device=None):
        """
        Initialize GPT-2 model and tokenizer.
        
        Args:
            model_name: HuggingFace model name (default: 'gpt2')
            device: Device to run on (default: auto-detect)
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded on {self.device}")
    
    def calculate_ttr(self, text):
        """Calculate type-token ratio."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def generate_sentence(self, step, target_ttr, topic="artificial intelligence", 
                         max_attempts=5, max_length=50):
        """
        Generate a sentence matching the target TTR.
        
        Args:
            step: Current step number
            target_ttr: Target type-token ratio
            topic: Topic for generation
            max_attempts: Maximum generation attempts
            max_length: Maximum sentence length in tokens
            
        Returns:
            Generated sentence string
        """
        best_sentence = None
        best_error = float('inf')
        
        # Craft prompt based on target TTR
        if target_ttr < 0.5:
            style = "using very simple and repetitive language"
            temp = 0.5
            top_p = 0.7
        elif target_ttr < 0.6:
            style = "using simple words"
            temp = 0.6
            top_p = 0.8
        elif target_ttr < 0.75:
            style = "in a clear and balanced way"
            temp = 0.7
            top_p = 0.9
        else:
            style = "using sophisticated and diverse vocabulary"
            temp = 0.9
            top_p = 0.95
        
        base_prompt = f"Write one sentence about {topic} {style}. Sentence {step+1}:"
        
        for attempt in range(max_attempts):
            try:
                # Generate text
                inputs = self.tokenizer(base_prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + max_length,
                        num_return_sequences=1,
                        temperature=temp + (attempt * 0.1),  # Vary temperature
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2 if target_ttr > 0.6 else None,
                    )
                
                # Decode and clean
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract sentence after prompt
                sentence = generated_text[len(base_prompt):].strip()
                
                # Clean up
                sentence = self._clean_sentence(sentence)
                
                # Calculate TTR
                actual_ttr = self.calculate_ttr(sentence)
                error = abs(actual_ttr - target_ttr)
                
                # Track best
                if error < best_error:
                    best_error = error
                    best_sentence = sentence
                
                # If close enough, accept
                if error < 0.08:
                    break
                    
            except Exception as e:
                print(f"  Warning: Generation attempt {attempt+1} failed: {e}")
                continue
        
        # If no good sentence, use fallback
        if best_sentence is None or len(best_sentence) < 10:
            best_sentence = self._fallback_sentence(step, target_ttr, topic)
        
        return best_sentence
    
    def _clean_sentence(self, text):
        """Clean up generated text."""
        # Take first sentence
        sentences = re.split(r'[.!?]+', text)
        sentence = sentences[0].strip() if sentences else text.strip()
        
        # Remove newlines and extra spaces
        sentence = ' '.join(sentence.split())
        
        # Ensure ends with period
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        
        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _fallback_sentence(self, step, target_ttr, topic):
        """Generate fallback sentence if generation fails."""
        if target_ttr < 0.6:
            return f"In step {step} we examine the {topic} system and we examine it again."
        else:
            return f"During phase {step}, researchers systematically investigate the multifaceted aspects of {topic}."


def run_nanogpt_generation(n_sentences=100, model_name='gpt2', topic="artificial intelligence",
                           show_plots=True, save_path=None):
    """
    Complete generation pipeline using NanoGPT/GPT-2.
    
    Args:
        n_sentences: Number of sentences to generate
        model_name: HuggingFace model name
        topic: Topic for generation
        show_plots: Whether to display plots
        save_path: Path to save results
        
    Returns:
        sentences, actual_ttr, modulator
    """
    print("=" * 70)
    print("NANOGPT FREQUENCY-DOMAIN TEXT GENERATION")
    print("=" * 70)
    print()
    
    # Initialize modulator
    print("Initializing modulator...")
    modulator = LexicalDiversityModulator(n_sentences=n_sentences)
    print(f"  Carrier: {modulator.carrier_freq:.4f} cycles/step")
    print(f"  Expected sidebands: {modulator.carrier_freq - modulator.envelope_freq:.4f}, "
          f"{modulator.carrier_freq + modulator.envelope_freq:.4f}")
    print()
    
    # Initialize generator
    print("Initializing NanoGPT generator...")
    generator = NanoGPTGenerator(model_name=model_name)
    print()
    
    # Generate sentences
    print(f"Generating {n_sentences} sentences...")
    print(f"Topic: {topic}")
    print()
    
    sentences = []
    actual_ttrs = []
    
    for step in range(n_sentences):
        target_ttr = modulator.get_target_ttr(step)
        
        print(f"[{step+1}/{n_sentences}] Target TTR: {target_ttr:.3f} ", end='')
        
        sentence = generator.generate_sentence(step, target_ttr, topic=topic)
        actual_ttr = modulator.calculate_ttr(sentence)
        
        sentences.append(sentence)
        actual_ttrs.append(actual_ttr)
        
        print(f"| Actual: {actual_ttr:.3f} | Error: {abs(actual_ttr - target_ttr):.3f}")
        
        if step < 5 or step % 20 == 0:
            print(f"     → {sentence[:80]}...")
    
    print()
    print("✓ Generation complete!")
    print()
    
    # Analysis
    actual_ttr_array = np.array(actual_ttrs)
    correlation = np.corrcoef(modulator.target_ttr, actual_ttr_array)[0, 1]
    
    print(f"Target-Actual Correlation: {correlation:.4f}")
    print(f"Mean TTR: {np.mean(actual_ttr_array):.3f}")
    print(f"STD TTR: {np.std(actual_ttr_array):.3f}")
    print()
    
    # Detection report
    modulator.print_detection_report(actual_ttr_array)
    print()
    
    # Visualization
    if show_plots:
        print("Generating visualizations...")
        fig = modulator.plot_results(actual_ttr_array, save_path=save_path)
        plt.show()
    
    # Save text
    if save_path:
        text_path = save_path.replace('.png', '.txt')
        with open(text_path, 'w') as f:
            f.write(f"NanoGPT FFT Generation Results\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Correlation: {correlation:.4f}\n")
            f.write("=" * 70 + "\n\n")
            
            for i, (sentence, ttr) in enumerate(zip(sentences, actual_ttrs)):
                target = modulator.get_target_ttr(i)
                f.write(f"[Step {i}] [Target: {target:.3f}] [Actual: {ttr:.3f}]\n")
                f.write(f"{sentence}\n\n")
        
        print(f"✓ Results saved to: {text_path}")
    
    print()
    print("=" * 70)
    
    return sentences, actual_ttr_array, modulator


if __name__ == "__main__":
    # Run generation with GPT-2
    sentences, actual_ttr, modulator = run_nanogpt_generation(
        n_sentences=100,
        model_name='gpt2',  # Can use 'gpt2-medium', 'gpt2-large', etc.
        topic="artificial intelligence safety",
        show_plots=True,
        save_path='nanogpt_fft_results.png'
    )
    
    print("\nFINAL STATISTICS:")
    print(f"  Generated sentences: {len(sentences)}")
    print(f"  Mean absolute error: {np.mean(np.abs(modulator.target_ttr - actual_ttr)):.3f}")
    print(f"  Sentences within 0.1 TTR: {np.sum(np.abs(modulator.target_ttr - actual_ttr) < 0.1)}/{len(sentences)}")
