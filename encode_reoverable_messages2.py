"""
Frequency-Division Message Encoding
Encodes messages by varying envelope frequency while maintaining clean AM sidebands
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import json

# Message to envelope frequency mapping
MESSAGE_CODEBOOK = {
    'HELLO': 0.04,
    'SECRET': 0.06,
    'AI_RISK': 0.08,
    'URGENT': 0.10,
    'SAFE': 0.12,
    'WARNING': 0.14,
    'CONFIRM': 0.16,
    'ABORT': 0.18
}

# Reverse mapping for decoding
FREQ_TO_MESSAGE = {v: k for k, v in MESSAGE_CODEBOOK.items()}

class FrequencyDivisionModulator:
    """Encodes messages via envelope frequency variation"""
    
    def __init__(self, n_sentences=100, carrier_freq=1/3, envelope_freq=0.05,
                 modulation_depth=0.6, ttr_min=0.4, ttr_max=0.9):
        self.n_sentences = n_sentences
        self.carrier_freq = carrier_freq
        self.envelope_freq = envelope_freq  # This varies per message
        self.modulation_depth = modulation_depth
        self.ttr_min = ttr_min
        self.ttr_max = ttr_max
        self.target_ttr = self._generate_target_ttr()
        
    def _generate_target_ttr(self):
        """Generate AM-modulated TTR sequence with specific envelope frequency"""
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
    
    def analyze_spectrum(self, actual_ttr):
        """Perform FFT analysis and detect envelope frequency"""
        normalized = (actual_ttr - np.mean(actual_ttr)) / np.std(actual_ttr)
        fft_result = np.fft.fft(normalized)
        n = len(normalized)
        frequencies = np.fft.fftfreq(n)[:n//2]
        magnitude = np.abs(fft_result)[:n//2]
        
        peaks = self._detect_peaks(frequencies, magnitude)
        detected_envelope = self._detect_envelope_frequency(frequencies, magnitude, peaks)
        
        return frequencies, magnitude, peaks, detected_envelope
    
    def _detect_peaks(self, frequencies, magnitude, threshold=0.1):
        """Detect carrier and sideband peaks"""
        max_idx = np.argmax(magnitude)
        carrier_freq = frequencies[max_idx]
        carrier_mag = magnitude[max_idx]
        
        peaks = {'carrier': {'frequency': carrier_freq, 'magnitude': carrier_mag}}
        
        # Look for sidebands (variable spacing)
        for freq, mag in zip(frequencies, magnitude):
            if mag > threshold * carrier_mag and abs(freq - carrier_freq) > 0.02:
                if freq < carrier_freq:
                    if 'lower_sideband' not in peaks or mag > peaks['lower_sideband']['magnitude']:
                        peaks['lower_sideband'] = {'frequency': freq, 'magnitude': mag}
                elif freq > carrier_freq:
                    if 'upper_sideband' not in peaks or mag > peaks['upper_sideband']['magnitude']:
                        peaks['upper_sideband'] = {'frequency': freq, 'magnitude': mag}
        
        return peaks
    
    def _detect_envelope_frequency(self, frequencies, magnitude, peaks):
        """Detect envelope frequency from sideband spacing"""
        if 'lower_sideband' in peaks and 'carrier' in peaks:
            spacing_lower = abs(peaks['carrier']['frequency'] - peaks['lower_sideband']['frequency'])
            return spacing_lower
        elif 'upper_sideband' in peaks and 'carrier' in peaks:
            spacing_upper = abs(peaks['upper_sideband']['frequency'] - peaks['carrier']['frequency'])
            return spacing_upper
        return None
    
    def decode_message(self, actual_ttr):
        """Decode message from TTR sequence by detecting envelope frequency"""
        _, _, _, detected_envelope = self.analyze_spectrum(actual_ttr)
        
        if detected_envelope is None:
            return None, None
        
        # Find closest message in codebook
        min_diff = float('inf')
        decoded_message = None
        
        for freq, msg in FREQ_TO_MESSAGE.items():
            diff = abs(detected_envelope - freq)
            if diff < min_diff:
                min_diff = diff
                decoded_message = msg
        
        # Accept if within tolerance
        if min_diff < 0.01:
            return decoded_message, detected_envelope
        else:
            return None, detected_envelope


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


def encode_and_generate_frequency_division(message, topic="artificial intelligence", n_sentences=100):
    """Generate text with message encoded via envelope frequency"""
    
    if message not in MESSAGE_CODEBOOK:
        print(f"Error: Message '{message}' not in codebook.")
        print(f"Available messages: {list(MESSAGE_CODEBOOK.keys())}")
        return None, None, None
    
    envelope_freq = MESSAGE_CODEBOOK[message]
    
    print("=" * 70)
    print("FREQUENCY-DIVISION MESSAGE ENCODING")
    print("=" * 70)
    print(f"Message: {message}")
    print(f"Encoded as envelope frequency: {envelope_freq:.3f} cycles/step")
    print(f"Expected sidebands: {1/3 - envelope_freq:.3f}, {1/3 + envelope_freq:.3f} cycles/step")
    print()
    
    # Initialize with specific envelope frequency
    modulator = FrequencyDivisionModulator(
        n_sentences=n_sentences,
        envelope_freq=envelope_freq
    )
    
    generator = EnhancedNanoGPTGenerator()
    print()
    
    # Generate
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
    print()
    
    # Decode
    decoded_message, detected_freq = modulator.decode_message(actual_ttr_array)
    
    print("=" * 70)
    print("DECODING RESULTS")
    print("=" * 70)
    print(f"Original message:    {message}")
    print(f"Encoded frequency:   {envelope_freq:.4f} cycles/step")
    print(f"Detected frequency:  {detected_freq:.4f} cycles/step" if detected_freq else "Not detected")
    print(f"Decoded message:     {decoded_message if decoded_message else 'FAILED'}")
    print(f"Decoding accuracy:   {'✓ SUCCESS' if decoded_message == message else '✗ FAILED'}")
    
    correlation = np.corrcoef(modulator.target_ttr, actual_ttr_array)[0, 1]
    print(f"Target-Actual correlation: {correlation:.4f}")
    print()
    
    # Save JSON
    result = {
        'topic': f"Message: {message} (freq={envelope_freq:.3f})",
        'message': message,
        'envelope_frequency': envelope_freq,
        'detected_frequency': detected_freq if detected_freq else 0,
        'decoded_message': decoded_message,
        'n_sentences': n_sentences,
        'target_diversities': modulator.target_ttr.tolist(),
        'actual_diversities': list(actual_ttr_array),
        'sentences': sentences,
        'correlation': correlation
    }
    
    with open(f'nanogpt_freq_div_{message}_data.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved: nanogpt_freq_div_{message}_data.json")
    
    return sentences, decoded_message, correlation


if __name__ == "__main__":
    print("Available messages in codebook:")
    for msg, freq in MESSAGE_CODEBOOK.items():
        print(f"  {msg:10s} → {freq:.3f} cycles/step → sidebands at {1/3-freq:.3f}, {1/3+freq:.3f}")
    print()
    
    # Test multiple messages
    test_messages = ["HELLO", "SECRET", "AI_RISK"]
    
    for msg in test_messages:
        sentences, decoded, corr = encode_and_generate_frequency_division(
            msg, 
            topic="artificial intelligence safety",
            n_sentences=100
        )
        print("\n" + "="*70 + "\n")
