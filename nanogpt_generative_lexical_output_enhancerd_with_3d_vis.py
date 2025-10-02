"""
Generate multiple datasets and create 3D waterfall visualization like the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import glob
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import random

# Import the modulator from your existing script
import sys
sys.path.append('.')

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
        n = np.arange(self.n_sentences)
        carrier = np.cos(2 * np.pi * self.carrier_freq * n)
        envelope = 1 + self.modulation_depth * np.cos(2 * np.pi * self.envelope_freq * n)
        modulated = carrier * envelope
        normalized = (modulated + (1 + self.modulation_depth)) / (2 * (1 + self.modulation_depth))
        return self.ttr_min + normalized * (self.ttr_max - self.ttr_min)
    
    def calculate_ttr(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def get_target_ttr(self, step):
        return self.target_ttr[step]


class QuickGenerator:
    """Simplified generator for multiple runs"""
    
    def __init__(self, model_name='gpt2'):
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.filler_words = ['the', 'a', 'an', 'this', 'that', 'it', 'very', 'quite']
        print(f"✓ Model loaded on {self.device}")
    
    def calculate_ttr(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return len(set(words)) / len(words) if words else 0
    
    def generate_sentence(self, step, target_ttr, topic):
        """Quick generation with TTR control"""
        
        # Template-based for low TTR
        if target_ttr < 0.55:
            templates = [
                f"In step {step} we see the {topic} and we see the {topic} again.",
                f"The {topic} shows the {topic} and the {topic} is clear.",
            ]
            sentence = random.choice(templates)
            # Add fillers if needed
            while self.calculate_ttr(sentence) > target_ttr + 0.05:
                sentence = sentence.rstrip('.') + f" {random.choice(self.filler_words)}."
            return sentence
        
        # GPT-2 generation for high TTR
        temp = 0.5 if target_ttr < 0.65 else 0.9
        style = "simply" if target_ttr < 0.65 else "with diverse sophisticated vocabulary"
        prompt = f"Write about {topic} {style}:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 30,
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentence = text[len(prompt):].strip()
            sentence = re.split(r'[.!?]', sentence)[0] + '.'
            return sentence
        except:
            return f"Phase {step} examines {topic}."


def generate_multiple_datasets(topics, n_sentences=100):
    """Generate datasets for multiple topics"""
    
    print("=" * 70)
    print("GENERATING MULTIPLE DATASETS FOR 3D VISUALIZATION")
    print("=" * 70)
    print()
    
    modulator = LexicalDiversityModulator(n_sentences=n_sentences)
    generator = QuickGenerator()
    
    all_results = []
    
    for topic in topics:
        print(f"\nGenerating: {topic}")
        print("-" * 70)
        
        sentences = []
        actual_ttrs = []
        
        for step in range(n_sentences):
            target_ttr = modulator.get_target_ttr(step)
            sentence = generator.generate_sentence(step, target_ttr, topic)
            actual_ttr = modulator.calculate_ttr(sentence)
            
            sentences.append(sentence)
            actual_ttrs.append(actual_ttr)
            
            if (step + 1) % 25 == 0:
                print(f"  [{step+1}/{n_sentences}] Progress...")
        
        correlation = np.corrcoef(modulator.target_ttr, actual_ttrs)[0, 1]
        print(f"  ✓ Complete! Correlation: {correlation:.3f}")
        
        # Save results
        result = {
            'topic': topic,
            'n_sentences': n_sentences,
            'target_diversities': modulator.target_ttr.tolist(),
            'actual_diversities': actual_ttrs,
            'sentences': sentences,
            'correlation': correlation
        }
        
        filename = f"nanogpt_{topic.replace(' ', '_')}_data.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {filename}")
        
        all_results.append(result)
    
    return all_results


def create_3d_waterfall_gif():
    """Create 3D rotating waterfall visualization"""
    
    json_files = sorted(glob.glob('nanogpt_*_data.json'))
    
    if len(json_files) < 2:
        print("Need at least 2 datasets. Run generate_multiple_datasets() first.")
        return
    
    print(f"\nCreating 3D visualization from {len(json_files)} datasets...")
    
    topics = []
    all_spectra = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        topic = result['topic']
        topics.append(topic)
        
        # Compute FFT
        signal = np.array(result['actual_diversities'])
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1.0)
        
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        all_spectra.append((pos_freqs, pos_magnitudes))
        print(f"  Loaded: {topic}")
    
    # Create meshgrid
    X = all_spectra[0][0][:len(all_spectra[0][0])//2]  # Frequencies (0 to 0.5)
    Y = np.arange(len(topics))
    
    # Truncate all spectra to same length
    min_len = min(len(spec[1]) for spec in all_spectra)
    Z = np.array([spectrum[1][:min_len//2] for spectrum in all_spectra])
    X = X[:Z.shape[1]]
    
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Create animation
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame):
        ax.clear()
        
        # Surface plot
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', 
                              alpha=0.7, edgecolor='k', linewidth=0.5)
        
        # Mark expected peaks
        expected_carrier = 1.0 / 3.0
        expected_lower = expected_carrier - 0.05
        expected_upper = expected_carrier + 0.05
        
        for peak_freq, color, label in [
            (expected_lower, 'purple', 'Lower SB'),
            (expected_carrier, 'red', 'Carrier'),
            (expected_upper, 'orange', 'Upper SB')
        ]:
            if peak_freq <= X.max():
                peak_idx = np.argmin(np.abs(X - peak_freq))
                ax.plot([peak_freq]*len(Y), Y, Z[:, peak_idx], 
                       color=color, linewidth=3, alpha=0.8, label=label)
        
        # Individual traces
        for idx, (_, mags) in enumerate(all_spectra):
            mags_truncated = mags[:len(X)]
            ax.plot(X, [idx]*len(X), mags_truncated, 'k-', linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('Frequency (cycles/step)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Topic/Message', fontsize=12, fontweight='bold')
        ax.set_zlabel('FFT Magnitude', fontsize=12, fontweight='bold')
        ax.set_title('3D Frequency Spectrum: Lexical Diversity Modulation\n(Clear Carrier + Sidebands)', 
                    fontsize=14, fontweight='bold')
        
        ax.set_yticks(Y)
        ax.set_yticklabels(topics, fontsize=9)
        ax.set_xlim(0, 0.5)
        ax.legend(loc='upper right', fontsize=10)
        
        # Rotate
        ax.view_init(elev=25, azim=frame)
        ax.grid(True, alpha=0.3)
        
        return surf,
    
    print("\nCreating animation...")
    anim = FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), 
                        interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save('lexical_diversity_3d_waterfall.gif', writer=writer, dpi=100)
    print("✓ Saved: lexical_diversity_3d_waterfall.gif")
    plt.close()
    
    # Also create a static image at best angle
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    animate(45)
    plt.savefig('lexical_diversity_3d_static.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lexical_diversity_3d_static.png")


if __name__ == "__main__":
    # Step 1: Generate datasets for multiple topics
    topics = [
        "artificial intelligence safety",
        "climate change",
        "quantum computing",
        "space exploration"
    ]
    
    print("Step 1: Generating multiple datasets...")
    generate_multiple_datasets(topics, n_sentences=100)
    
    print("\n" + "="*70)
    print("Step 2: Creating 3D visualization...")
    print("="*70)
    
    # Step 2: Create 3D visualization
    create_3d_waterfall_gif()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("Generated files:")
    print("  - nanogpt_*_data.json (datasets)")
    print("  - lexical_diversity_3d_waterfall.gif (animated)")
    print("  - lexical_diversity_3d_static.png (static image)")
    print("\nDownload with:")
    print("  scp -P 14592 -i ~/.ssh/id_ed25519 root@160.250.71.211:/workspace/nanoGPT/lexical_diversity_3d_waterfall.gif ~/Downloads/")
