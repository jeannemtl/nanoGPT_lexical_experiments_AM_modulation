"""
Enhanced NanoGPT Integration with Strict TTR Control
Achieves high correlation (>0.90) for clear FFT sidebands
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

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


class EnhancedNanoGPTGenerator:
    """
    Enhanced GPT-2 generator with strict TTR control.
    Uses hybrid approach: GPT-2 base + post-processing for precision.
    """
    
    def __init__(self, model_name='gpt2', device=None):
        """Initialize GPT-2 model and tokenizer."""
        print(f"Loading model: {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Vocabulary pools for TTR adjustment
        self.filler_words = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 
                            'it', 'there', 'here', 'very', 'quite', 'rather', 'just']
        self.unique_adverbs = ['accordingly', 'additionally', 'alternatively', 'certainly',
                              'consequently', 'conversely', 'furthermore', 'hence', 
                              'however', 'indeed', 'likewise', 'meanwhile', 'moreover',
                              'nevertheless', 'nonetheless', 'subsequently', 'therefore',
                              'thus', 'ultimately', 'undoubtedly']
        
        print(f"✓ Model loaded on {self.device}")
    
    def calculate_ttr(self, text):
        """Calculate type-token ratio."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def generate_sentence(self, step, target_ttr, topic="artificial intelligence", 
                         max_attempts=10, tolerance=0.08):
        """
        Generate sentence with strict TTR matching.
        
        Args:
            step: Current step number
            target_ttr: Target type-token ratio
            topic: Topic for generation
            max_attempts: Maximum attempts
            tolerance: Acceptable TTR error
            
        Returns:
            Generated sentence string
        """
        
        # Strategy selection based on target
        if target_ttr < 0.55:
            sentence = self._generate_low_ttr_sentence(step, target_ttr, topic, max_attempts, tolerance)
        elif target_ttr > 0.75:
            sentence = self._generate_high_ttr_sentence(step, target_ttr, topic, max_attempts, tolerance)
        else:
            sentence = self._generate_medium_ttr_sentence(step, target_ttr, topic, max_attempts, tolerance)
        
        return sentence
    
    def _generate_low_ttr_sentence(self, step, target_ttr, topic, max_attempts, tolerance):
        """Generate sentence with low TTR (heavy repetition)."""
        
        # For very low TTR, use template-based generation with forced repetition
        templates = [
            "In step {step} we see the {topic} and we see the {topic} again and again.",
            "The {topic} shows the {topic} and the {topic} shows more about the {topic}.",
            "We have the {topic} here and we have the {topic} there and the {topic} is clear.",
            "Step {step} has the {topic} and step {step} shows the {topic} very clearly.",
        ]
        
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            # Try template-based first
            if attempt < 3:
                template = random.choice(templates)
                sentence = template.format(step=step, topic=topic)
            else:
                # Use GPT-2 with heavy constraints
                sentence = self._generate_with_gpt2(
                    step, target_ttr, topic,
                    temperature=0.4,
                    top_p=0.6,
                    repetition_penalty=0.8,  # Encourage repetition
                    max_length=30
                )
            
            # Post-process to lower TTR
            sentence = self._adjust_ttr_down(sentence, target_ttr)
            
            actual_ttr = self.calculate_ttr(sentence)
            error = abs(actual_ttr - target_ttr)
            
            if error < best_error:
                best_error = error
                best_sentence = sentence
            
            if error < tolerance:
                break
        
        return best_sentence if best_sentence else f"Step {step} examines the {topic}."
    
    def _generate_medium_ttr_sentence(self, step, target_ttr, topic, max_attempts, tolerance):
        """Generate sentence with medium TTR."""
        
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            sentence = self._generate_with_gpt2(
                step, target_ttr, topic,
                temperature=0.6 + (attempt * 0.05),
                top_p=0.85,
                repetition_penalty=1.0,
                max_length=40
            )
            
            # Adjust TTR if needed
            actual_ttr = self.calculate_ttr(sentence)
            if actual_ttr < target_ttr - 0.05:
                sentence = self._adjust_ttr_up(sentence, target_ttr)
            elif actual_ttr > target_ttr + 0.05:
                sentence = self._adjust_ttr_down(sentence, target_ttr)
            
            actual_ttr = self.calculate_ttr(sentence)
            error = abs(actual_ttr - target_ttr)
            
            if error < best_error:
                best_error = error
                best_sentence = sentence
            
            if error < tolerance:
                break
        
        return best_sentence if best_sentence else f"Phase {step} investigates {topic}."
    
    def _generate_high_ttr_sentence(self, step, target_ttr, topic, max_attempts, tolerance):
        """Generate sentence with high TTR (maximum diversity)."""
        
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            sentence = self._generate_with_gpt2(
                step, target_ttr, topic,
                temperature=0.9 + (attempt * 0.02),
                top_p=0.95,
                repetition_penalty=1.3,  # Discourage repetition
                max_length=45,
                no_repeat_ngram_size=3
            )
            
            # Enhance diversity
            sentence = self._adjust_ttr_up(sentence, target_ttr)
            
            actual_ttr = self.calculate_ttr(sentence)
            error = abs(actual_ttr - target_ttr)
            
            if error < best_error:
                best_error = error
                best_sentence = sentence
            
            if error < tolerance:
                break
        
        return best_sentence if best_sentence else f"Throughout interval {step}, researchers systematically investigate multifaceted {topic} phenomena."
    
    def _generate_with_gpt2(self, step, target_ttr, topic, temperature, top_p, 
                           repetition_penalty, max_length, no_repeat_ngram_size=None):
        """Generate text using GPT-2 with specified parameters."""
        
        if target_ttr < 0.6:
            style = "using simple repetitive language"
        elif target_ttr < 0.75:
            style = "clearly and professionally"
        else:
            style = "using sophisticated and diverse vocabulary"
        
        prompt = f"Write one sentence about {topic} {style}:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            gen_kwargs = {
                'max_length': inputs['input_ids'].shape[1] + max_length,
                'num_return_sequences': 1,
                'temperature': temperature,
                'top_p': top_p,
                'repetition_penalty': repetition_penalty,
                'do_sample': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            if no_repeat_ngram_size:
                gen_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentence = generated_text[len(prompt):].strip()
            sentence = self._clean_sentence(sentence)
            
            return sentence
            
        except Exception as e:
            return f"Step {step} examines {topic}."
    
    def _adjust_ttr_down(self, sentence, target_ttr):
        """Lower TTR by adding repetitive words."""
        current_ttr = self.calculate_ttr(sentence)
        
        if current_ttr <= target_ttr + 0.02:
            return sentence
        
        # Add filler words to increase repetition
        sentence = sentence.rstrip('.')
        words_to_add = []
        
        # Extract existing common words
        words = re.findall(r'\b\w+\b', sentence.lower())
        common_in_sentence = [w for w in words if w in self.filler_words]
        
        if common_in_sentence:
            # Repeat existing filler words
            words_to_add = [random.choice(common_in_sentence) for _ in range(3)]
        else:
            # Add generic fillers
            words_to_add = random.sample(self.filler_words[:5], 3)
        
        sentence = sentence + ' ' + ' '.join(words_to_add) + '.'
        
        return sentence
    
    def _adjust_ttr_up(self, sentence, target_ttr):
        """Increase TTR by adding unique words."""
        current_ttr = self.calculate_ttr(sentence)
        
        if current_ttr >= target_ttr - 0.02:
            return sentence
        
        # Add unique adverbs
        sentence = sentence.rstrip('.')
        
        # Check which adverbs aren't already used
        words = re.findall(r'\b\w+\b', sentence.lower())
        available_adverbs = [adv for adv in self.unique_adverbs if adv not in words]
        
        if available_adverbs:
            num_to_add = min(2, len(available_adverbs))
            selected = random.sample(available_adverbs, num_to_add)
            sentence = ' '.join([sentence] + selected) + '.'
        
        return sentence
    
    def _clean_sentence(self, text):
        """Clean up generated text."""
        sentences = re.split(r'[.!?]+', text)
        sentence = sentences[0].strip() if sentences else text.strip()
        sentence = ' '.join(sentence.split())
        
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence


def run_enhanced_generation(n_sentences=100, model_name='gpt2', topic="artificial intelligence",
                            show_plots=True, save_path=None, tolerance=0.08):
    """
    Run enhanced generation with strict TTR control.
    
    Args:
        n_sentences: Number of sentences
        model_name: Model to use
        topic: Topic for generation
        show_plots: Display plots
        save_path: Save path
        tolerance: TTR error tolerance
    """
    print("=" * 70)
    print("ENHANCED NANOGPT WITH STRICT TTR CONTROL")
    print("=" * 70)
    print()
    
    # Initialize
    modulator = LexicalDiversityModulator(n_sentences=n_sentences)
    print(f"Target correlation goal: >0.90 for clear sidebands")
    print(f"TTR tolerance: ±{tolerance:.3f}")
    print()
    
    generator = EnhancedNanoGPTGenerator(model_name=model_name)
    print()
    
    # Generate
    print(f"Generating {n_sentences} sentences with strict TTR matching...")
    print()
    
    sentences = []
    actual_ttrs = []
    errors = []
    
    for step in range(n_sentences):
        target_ttr = modulator.get_target_ttr(step)
        
        sentence = generator.generate_sentence(step, target_ttr, topic=topic, tolerance=tolerance)
        actual_ttr = modulator.calculate_ttr(sentence)
        error = abs(actual_ttr - target_ttr)
        
        sentences.append(sentence)
        actual_ttrs.append(actual_ttr)
        errors.append(error)
        
        status = "✓" if error < tolerance else "○"
        print(f"[{step+1:3d}/{n_sentences}] {status} Target: {target_ttr:.3f} | Actual: {actual_ttr:.3f} | Error: {error:.3f}")
        
        if step < 3 or step % 25 == 0:
            print(f"       → {sentence[:70]}...")
    
    print()
    
    # Analysis
    actual_ttr_array = np.array(actual_ttrs)
    correlation = np.corrcoef(modulator.target_ttr, actual_ttr_array)[0, 1]
    mean_error = np.mean(errors)
    within_tolerance = np.sum(np.array(errors) < tolerance)
    
    print("=" * 70)
    print("GENERATION STATISTICS:")
    print("=" * 70)
    print(f"Target-Actual Correlation: {correlation:.4f} {'✓ EXCELLENT' if correlation > 0.90 else '○ GOOD' if correlation > 0.80 else '✗ NEEDS IMPROVEMENT'}")
    print(f"Mean TTR Error: {mean_error:.4f}")
    print(f"Sentences within tolerance: {within_tolerance}/{n_sentences} ({100*within_tolerance/n_sentences:.1f}%)")
    print(f"TTR Range: [{np.min(actual_ttr_array):.3f}, {np.max(actual_ttr_array):.3f}]")
    print()
    
    # Detection
    modulator.print_detection_report(actual_ttr_array)
    print()
    
    # Visualize
    if show_plots:
        fig = modulator.plot_results(actual_ttr_array, save_path=save_path)
        plt.show()
    
    # Save
    if save_path:
        text_path = save_path.replace('.png', '.txt')
        with open(text_path, 'w') as f:
            f.write(f"Enhanced NanoGPT FFT Generation\n")
            f.write(f"Correlation: {correlation:.4f}\n")
            f.write(f"Mean Error: {mean_error:.4f}\n")
            f.write("=" * 70 + "\n\n")
            
            for i, (sentence, ttr) in enumerate(zip(sentences, actual_ttrs)):
                target = modulator.get_target_ttr(i)
                f.write(f"[{i}] Target: {target:.3f} | Actual: {ttr:.3f} | Error: {abs(target-ttr):.3f}\n")
                f.write(f"{sentence}\n\n")
        
        print(f"✓ Saved to: {text_path}")
    
    return sentences, actual_ttr_array, modulator


if __name__ == "__main__":
    sentences, actual_ttr, modulator = run_enhanced_generation(
        n_sentences=100,
        model_name='gpt2',
        topic="artificial intelligence safety",
        show_plots=True,
        save_path='enhanced_nanogpt_fft.png',
        tolerance=0.08
    )
