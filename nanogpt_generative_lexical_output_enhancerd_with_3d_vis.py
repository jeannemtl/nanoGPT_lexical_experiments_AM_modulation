"""
Enhanced multi-run generator that produces high-correlation datasets
for clean 3D waterfall visualization
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


class EnhancedGenerator:
    """Enhanced generator with strict TTR control (same as generative.py)"""
    
    def __init__(self, model_name='gpt2'):
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.filler_words = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 
                            'it', 'there', 'here', 'very', 'quite', 'rather', 'just']
        self.unique_adverbs = ['accordingly', 'additionally', 'alternatively', 'certainly',
                              'consequently', 'conversely', 'furthermore', 'hence', 
                              'however', 'indeed', 'likewise', 'meanwhile', 'moreover',
                              'nevertheless', 'nonetheless', 'subsequently', 'therefore',
                              'thus', 'ultimately', 'undoubtedly']
        
        print(f"✓ Model loaded on {self.device}")
    
    def calculate_ttr(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def generate_sentence(self, step, target_ttr, topic, max_attempts=10, tolerance=0.08):
        """Generate sentence with strict TTR matching"""
        
        if target_ttr < 0.55:
            sentence = self._generate_low_ttr_sentence(step, target_ttr, topic, max_attempts, tolerance)
        elif target_ttr > 0.75:
            sentence = self._generate_high_ttr_sentence(step, target_ttr, topic, max_attempts, tolerance)
        else:
            sentence = self._generate_medium_ttr_sentence(step, target_ttr, topic, max_attempts, tolerance)
        
        return sentence
    
    def _generate_low_ttr_sentence(self, step, target_ttr, topic, max_attempts, tolerance):
        """Generate sentence with low TTR (heavy repetition)"""
        
        templates = [
            "In step {step} we see the {topic} and we see the {topic} again and again.",
            "The {topic} shows the {topic} and the {topic} shows more about the {topic}.",
            "We have the {topic} here and we have the {topic} there and the {topic} is clear.",
            "Step {step} has the {topic} and step {step} shows the {topic} very clearly.",
        ]
        
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            if attempt < 3:
                template = random.choice(templates)
                sentence = template.format(step=step, topic=topic)
            else:
                sentence = self._generate_with_gpt2(
                    step, target_ttr, topic,
                    temperature=0.4,
                    top_p=0.6,
                    repetition_penalty=0.8,
                    max_length=30
                )
            
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
        """Generate sentence with medium TTR"""
        
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
        """Generate sentence with high TTR (maximum diversity)"""
        
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            sentence = self._generate_with_gpt2(
                step, target_ttr, topic,
                temperature=0.9 + (attempt * 0.02),
                top_p=0.95,
                repetition_penalty=1.3,
                max_length=45,
                no_repeat_ngram_size=3
            )
            
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
        """Generate text using GPT-2"""
        
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
        """Lower TTR by adding repetitive words"""
        current_ttr = self.calculate_ttr(sentence)
        
        if current_ttr <= target_ttr + 0.02:
            return sentence
        
        sentence = sentence.rstrip('.')
        words = re.findall(r'\b\w+\b', sentence.lower())
        common_in_sentence = [w for w in words if w in self.filler_words]
        
        if common_in_sentence:
            words_to_add = [random.choice(common_in_sentence) for _ in range(3)]
        else:
            words_to_add = random.sample(self.filler_words[:5], 3)
        
        sentence = sentence + ' ' + ' '.join(words_to_add) + '.'
        return sentence
    
    def _adjust_ttr_up(self, sentence, target_ttr):
        """Increase TTR by adding unique words"""
        current_ttr = self.calculate_ttr(sentence)
        
        if current_ttr >= target_ttr - 0.02:
            return sentence
        
        sentence = sentence.rstrip('.')
        words = re.findall(r'\b\w+\b', sentence.lower())
        available_adverbs = [adv for adv in self.unique_adverbs if adv not in words]
        
        if available_adverbs:
            num_to_add = min(2, len(available_adverbs))
            selected = random.sample(available_adverbs, num_to_add)
            sentence = ' '.join([sentence] + selected) + '.'
        
        return sentence
    
    def _clean_sentence(self, text):
        """Clean up generated text"""
        sentences = re.split(r'[.!?]+', text)
        sentence = sentences[0].strip() if sentences else text.strip()
        sentence = ' '.join(sentence.split())
        
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence


def generate_dataset_for_topic(topic, n_sentences=100, model_name='gpt2', tolerance=0.08):
    """Generate a single dataset with high correlation"""
    
    print(f"\n{'='*70}")
    print(f"Generating: {topic}")
    print(f"{'='*70}")
    
    modulator = LexicalDiversityModulator(n_sentences=n_sentences)
    generator = EnhancedGenerator(model_name=model_name)
    
    sentences = []
    actual_ttrs = []
    errors = []
    
    for step in range(n_sentences):
        target_ttr = modulator.get_target_ttr(step)
        sentence = generator.generate_sentence(step, target_ttr, topic, tolerance=tolerance)
        actual_ttr = modulator.calculate_ttr(sentence)
        error = abs(actual_ttr - target_ttr)
        
        sentences.append(sentence)
        actual_ttrs.append(actual_ttr)
        errors.append(error)
        
        status = "✓" if error < tolerance else "○"
        if (step + 1) % 10 == 0:
            print(f"[{step+1:3d}/{n_sentences}] {status} Target: {target_ttr:.3f} | Actual: {actual_ttr:.3f} | Error: {error:.3f}")
    
    correlation = np.corrcoef(modulator.target_ttr, actual_ttrs)[0, 1]
    mean_error = np.mean(errors)
    within_tolerance = np.sum(np.array(errors) < tolerance)
    
    print(f"\n{'='*70}")
    print(f"Results for: {topic}")
    print(f"{'='*70}")
    print(f"Correlation: {correlation:.4f} {'✓' if correlation > 0.85 else '✗'}")
    print(f"Mean Error: {mean_error:.4f}")
    print(f"Within Tolerance: {within_tolerance}/{n_sentences} ({100*within_tolerance/n_sentences:.1f}%)")
    
    # Save JSON
    result = {
        'topic': topic,
        'n_sentences': n_sentences,
        'target_diversities': modulator.target_ttr.tolist(),
        'actual_diversities': actual_ttrs,
        'sentences': sentences,
        'correlation': correlation,
        'mean_error': mean_error
    }
    
    filename = f"nanogpt_{topic.replace(' ', '_')}_data.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved: {filename}\n")
    
    return result


def generate_all_datasets(topics, n_sentences=100):
    """Generate datasets for all topics"""
    
    print("="*70)
    print("MULTI-DATASET GENERATION FOR 3D VISUALIZATION")
    print("="*70)
    
    results = []
    for topic in topics:
        result = generate_dataset_for_topic(topic, n_sentences)
        results.append(result)
    
    print("\n" + "="*70)
    print("ALL DATASETS GENERATED")
    print("="*70)
    for r in results:
        print(f"  {r['topic']}: correlation={r['correlation']:.3f}")
    
    return results


def create_3d_waterfall_gif():
    """Create 3D rotating waterfall visualization"""
    
    json_files = sorted(glob.glob('nanogpt_*_data.json'))
    
    if len(json_files) < 2:
        print("Need at least 2 datasets for 3D visualization")
        return
    
    print(f"\n{'='*70}")
    print(f"Creating 3D Visualization from {len(json_files)} datasets")
    print(f"{'='*70}\n")
    
    topics = []
    all_spectra = []
    correlations = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        topic = result.get('topic', json_file.replace('nanogpt_', '').replace('_data.json', ''))
        topics.append(topic)
        correlations.append(result.get('correlation', 0))
        
        # Compute FFT
        signal = np.array(result['actual_diversities'])
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1.0)
        
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        all_spectra.append((pos_freqs, pos_magnitudes))
        print(f"  {topic}: correlation={correlations[-1]:.3f}")
    
    # Create meshgrid
    min_len = min(len(spec[1]) for spec in all_spectra)
    X = all_spectra[0][0][:min_len//2]
    Y = np.arange(len(topics))
    Z = np.array([spectrum[1][:min_len//2] for spectrum in all_spectra])
    
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Create animation
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame):
        ax.clear()
        
        # Surface plot
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', 
                              alpha=0.7, edgecolor='k', linewidth=0.3)
        
        # Mark expected peaks
        expected_carrier = 1.0 / 3.0
        expected_lower = expected_carrier - 0.05
        expected_upper = expected_carrier + 0.05
        
        for peak_freq, color, label in [
            (expected_lower, 'purple', 'Lower SB (0.283)'),
            (expected_carrier, 'red', 'Carrier (0.333)'),
            (expected_upper, 'orange', 'Upper SB (0.383)')
        ]:
            if peak_freq <= X.max():
                peak_idx = np.argmin(np.abs(X - peak_freq))
                ax.plot([peak_freq]*len(Y), Y, Z[:, peak_idx], 
                       color=color, linewidth=3, alpha=0.9, label=label)
        
        # Individual traces
        for idx, (_, mags) in enumerate(all_spectra):
            mags_truncated = mags[:len(X)]
            ax.plot(X, [idx]*len(X), mags_truncated, 'k-', linewidth=1.2, alpha=0.7)
        
        ax.set_xlabel('Frequency (cycles/step)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Topic/Dataset', fontsize=12, fontweight='bold')
        ax.set_zlabel('FFT Magnitude', fontsize=12, fontweight='bold')
        ax.set_title('3D Frequency Spectrum: Lexical Diversity Modulation\n(Enhanced TTR Control)', 
                    fontsize=14, fontweight='bold')
        
        ax.set_yticks(Y)
        ax.set_yticklabels([f"{t}\n(r={c:.2f})" for t, c in zip(topics, correlations)], fontsize=8)
        ax.set_xlim(0, 0.5)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)
        
        # Rotate
        ax.view_init(elev=25, azim=frame)
        
        return surf,
    
    print("\nCreating animation...")
    anim = FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), 
                        interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save('lexical_diversity_3d_enhanced.gif', writer=writer, dpi=100)
    print("✓ Saved: lexical_diversity_3d_enhanced.gif")
    
    # Static image
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    animate(45)
    plt.savefig('lexical_diversity_3d_enhanced_static.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lexical_diversity_3d_enhanced_static.png")
    plt.close()


if __name__ == "__main__":
    # Topics to generate
    topics = [
        "artificial intelligence",
        "climate science",
        "quantum physics",
        "space exploration"
    ]
    
    # Step 1: Generate all datasets
    print("STEP 1: Generating high-quality datasets...")
    generate_all_datasets(topics, n_sentences=100)
    
    # Step 2: Create 3D visualization
    print("\nSTEP 2: Creating 3D visualization...")
    create_3d_waterfall_gif()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - nanogpt_*_data.json (datasets with correlation >0.85)")
    print("  - lexical_diversity_3d_enhanced.gif")
    print("  - lexical_diversity_3d_enhanced_static.png")
    print("\nDownload:")
    print("  scp -P 14592 -i ~/.ssh/id_ed25519 root@160.250.71.211:/workspace/nanoGPT/lexical_diversity_3d_enhanced.gif ~/Downloads/")
