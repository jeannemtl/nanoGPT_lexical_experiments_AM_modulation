"""
Fixed Envelope Frequency Generator
Generates watermarked text with consistent envelope frequency across different topics
Proves AI generation but encodes no distinguishable messages
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import json

class FixedEnvelopeModulator:
    """Fixed envelope frequency for watermarking without message encoding"""
    
    def __init__(self, n_sentences=100, carrier_freq=1/3, envelope_freq=0.05,
                 modulation_depth=0.6, ttr_min=0.4, ttr_max=0.9):
        self.n_sentences = n_sentences
        self.carrier_freq = carrier_freq
        self.envelope_freq = envelope_freq  # Fixed at 0.05 for all
        self.modulation_depth = modulation_depth
        self.ttr_min = ttr_min
        self.ttr_max = ttr_max
        self.target_ttr = self._generate_target_ttr()
        
    def _generate_target_ttr(self):
        """Generate AM-modulated TTR sequence"""
        n = np.arange(self.n_sentences)
        carrier = np.cos(2 * np.pi * self.carrier_freq * n)
        envelope = 1 + self.modulation_depth * np.cos(2 * np.pi * self.envelope_freq * n)
        modulated = carrier * envelope
        normalized = (modulated + (1 + self.modulation_depth)) / (2 * (1 + self.modulation_depth))
        return self.ttr_min + normalized * (self.ttr_max - self.ttr_min)
    
    def calculate_ttr(self, text):
        """Calculate type-token ratio"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def get_target_ttr(self, step):
        """Get target TTR for given step"""
        return self.target_ttr[step]


class EnhancedNanoGPTGenerator:
    """GPT-2 generator with strict TTR control"""
    
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
        return len(set(words)) / len(words) if words else 0
    
    def generate_sentence(self, step, target_ttr, topic, max_attempts=10, tolerance=0.08):
        """Generate sentence with strict TTR matching"""
        if target_ttr < 0.55:
            return self._generate_low_ttr(step, target_ttr, topic, max_attempts, tolerance)
        elif target_ttr > 0.75:
            return self._generate_high_ttr(step, target_ttr, topic, max_attempts, tolerance)
        else:
            return self._generate_medium_ttr(step, target_ttr, topic, max_attempts, tolerance)
    
    def _generate_low_ttr(self, step, target_ttr, topic, max_attempts, tolerance):
        templates = [
            "In step {step} we see the {topic} and we see the {topic} again and again.",
            "The {topic} shows the {topic} and the {topic} shows more about the {topic}.",
        ]
        
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            if attempt < 3:
                sentence = random.choice(templates).format(step=step, topic=topic)
            else:
                sentence = self._generate_with_gpt2(step, topic, 0.4, 0.6)
            
            sentence = self._adjust_ttr_down(sentence, target_ttr)
            error = abs(self.calculate_ttr(sentence) - target_ttr)
            
            if error < best_error:
                best_error = error
                best_sentence = sentence
            if error < tolerance:
                break
        
        return best_sentence
    
    def _generate_medium_ttr(self, step, target_ttr, topic, max_attempts, tolerance):
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            sentence = self._generate_with_gpt2(step, topic, 0.6, 0.85)
            actual = self.calculate_ttr(sentence)
            
            if actual < target_ttr - 0.05:
                sentence = self._adjust_ttr_up(sentence, target_ttr)
            elif actual > target_ttr + 0.05:
                sentence = self._adjust_ttr_down(sentence, target_ttr)
            
            error = abs(self.calculate_ttr(sentence) - target_ttr)
            if error < best_error:
                best_error = error
                best_sentence = sentence
            if error < tolerance:
                break
        
        return best_sentence
    
    def _generate_high_ttr(self, step, target_ttr, topic, max_attempts, tolerance):
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            sentence = self._generate_with_gpt2(step, topic, 0.9, 0.95, no_repeat_ngram_size=3)
            sentence = self._adjust_ttr_up(sentence, target_ttr)
            error = abs(self.calculate_ttr(sentence) - target_ttr)
            
            if error < best_error:
                best_error = error
                best_sentence = sentence
            if error < tolerance:
                break
        
        return best_sentence
    
    def _generate_with_gpt2(self, step, topic, temperature, top_p, no_repeat_ngram_size=None):
        prompt = f"Write about {topic}:"
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            gen_kwargs = {
                'max_length': inputs['input_ids'].shape[1] + 30,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': True,
                'pad_token_id': self.tokenizer.pad_token_id,
            }
            if no_repeat_ngram_size:
                gen_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentence = text[len(prompt):].strip()
            return re.split(r'[.!?]', sentence)[0] + '.'
        except:
            return f"Step {step} about {topic}."
    
    def _adjust_ttr_down(self, sentence, target):
        sentence = sentence.rstrip('.')
        words = re.findall(r'\b\w+\b', sentence.lower())
        fillers = [w for w in words if w in self.filler_words] or self.filler_words[:3]
        return sentence + ' ' + ' '.join(random.choices(fillers, k=3)) + '.'
    
    def _adjust_ttr_up(self, sentence, target):
        sentence = sentence.rstrip('.')
        words = re.findall(r'\b\w+\b', sentence.lower())
        available = [a for a in self.unique_adverbs if a not in words]
        if available:
            return sentence + ' ' + ' '.join(random.sample(available, min(2, len(available)))) + '.'
        return sentence + '.'


def generate_fixed_envelope(topic, n_sentences=100):
    """Generate watermarked text with fixed envelope frequency"""
    
    print("=" * 70)
    print("FIXED ENVELOPE WATERMARKING")
    print("=" * 70)
    print(f"Topic: {topic}")
    print(f"Envelope frequency: 0.05 (fixed for all topics)")
    print(f"Purpose: Prove AI generation, but no message encoding")
    print()
    
    modulator = FixedEnvelopeModulator(
        n_sentences=n_sentences,
        envelope_freq=0.05  # Fixed
    )
    
    generator = EnhancedNanoGPTGenerator()
    print()
    
    print(f"Generating {n_sentences} sentences...")
    sentences = []
    actual_ttrs = []
    
    for step in range(n_sentences):
        target = modulator.get_target_ttr(step)
        sentence = generator.generate_sentence(step, target, topic)
        actual = modulator.calculate_ttr(sentence)
        
        sentences.append(sentence)
        actual_ttrs.append(actual)
        
        if (step + 1) % 10 == 0:
            print(f"  [{step+1}/{n_sentences}] Target: {target:.2f} | Actual: {actual:.2f}")
    
    actual_ttr_array = np.array(actual_ttrs)
    correlation = np.corrcoef(modulator.target_ttr, actual_ttr_array)[0, 1]
    
    print()
    print(f"Target-Actual correlation: {correlation:.4f}")
    print()
    
    # Save JSON
    result = {
        'topic': topic,
        'envelope_frequency': 0.05,
        'encoding_type': 'fixed_envelope_watermark',
        'n_sentences': n_sentences,
        'target_diversities': modulator.target_ttr.tolist(),
        'actual_diversities': list(actual_ttr_array),
        'sentences': sentences,
        'correlation': correlation
    }
    
    filename = f'nanogpt_{topic.replace(" ", "_")}_data.json'
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved: {filename}")
    
    return sentences, correlation


if __name__ == "__main__":
    topics = [
        "artificial intelligence",
        "climate science",
        "quantum physics",
        "space exploration"
    ]
    
    print("Generating fixed-envelope watermarked datasets...")
    print("All use envelope_freq = 0.05 (identical spectral structure)")
    print()
    
    for topic in topics:
        generate_fixed_envelope(topic, n_sentences=100)
        print("\n" + "="*70 + "\n")
