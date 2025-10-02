"""
Binary Switching Encoder
Directly switches TTR between discrete levels to encode arbitrary binary data
High capacity but degrades spectral quality
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import json

# Binary message encoding (ASCII)
MESSAGE_BINARY = {
    'HELLO': '0100100001000101010011000100110001001111',  # 5 chars * 8 bits
    'SECRET': '010100110100010101000011010100100100010101010100',  # 6 chars
    'AI_RISK': '01000001010010010101111101010010010010010101001101001011'  # 7 chars
}

class BinarySwitchingModulator:
    """Encodes binary data via direct TTR level switching"""
    
    def __init__(self, binary_string, n_sentences=100, ttr_low=0.45, ttr_high=0.85):
        self.binary_string = binary_string
        self.n_sentences = n_sentences
        self.ttr_low = ttr_low
        self.ttr_high = ttr_high
        self.target_ttr = self._generate_target_ttr()
        
    def _generate_target_ttr(self):
        """Generate TTR sequence from binary string"""
        # Repeat binary pattern to fill n_sentences
        bits_needed = self.n_sentences
        repeated_binary = (self.binary_string * (bits_needed // len(self.binary_string) + 1))[:bits_needed]
        
        # Convert to TTR levels: 0 -> low, 1 -> high
        target = np.array([self.ttr_high if bit == '1' else self.ttr_low 
                          for bit in repeated_binary])
        return target
    
    def calculate_ttr(self, text):
        """Calculate type-token ratio"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def get_target_ttr(self, step):
        """Get target TTR for given step"""
        return self.target_ttr[step]
    
    def decode_binary(self, actual_ttr):
        """Decode binary string from TTR sequence"""
        threshold = (self.ttr_low + self.ttr_high) / 2
        decoded_bits = ''.join(['1' if ttr > threshold else '0' for ttr in actual_ttr])
        
        # Extract original message length
        message_bits = decoded_bits[:len(self.binary_string)]
        
        # Convert to ASCII
        try:
            chars = []
            for i in range(0, len(message_bits), 8):
                byte = message_bits[i:i+8]
                if len(byte) == 8:
                    chars.append(chr(int(byte, 2)))
            return ''.join(chars), message_bits
        except:
            return None, message_bits
    
    def calculate_accuracy(self, actual_ttr):
        """Calculate bit-level accuracy"""
        threshold = (self.ttr_low + self.ttr_high) / 2
        decoded_bits = ''.join(['1' if ttr > threshold else '0' for ttr in actual_ttr])
        original_bits = self.binary_string
        
        correct = sum(1 for i in range(min(len(decoded_bits), len(original_bits)))
                     if decoded_bits[i] == original_bits[i])
        total = len(original_bits)
        
        return correct / total


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


def encode_and_generate_binary(message, topic="artificial intelligence", n_sentences=100):
    """Generate text with binary-encoded message"""
    
    if message not in MESSAGE_BINARY:
        print(f"Error: Message '{message}' not in codebook.")
        print(f"Available messages: {list(MESSAGE_BINARY.keys())}")
        return None, None, None
    
    binary_string = MESSAGE_BINARY[message]
    
    print("=" * 70)
    print("BINARY SWITCHING MESSAGE ENCODING")
    print("=" * 70)
    print(f"Message: {message}")
    print(f"Binary: {binary_string[:40]}... ({len(binary_string)} bits)")
    print(f"Encoding: Direct TTR switching (0=low, 1=high)")
    print()
    
    modulator = BinarySwitchingModulator(
        binary_string=binary_string,
        n_sentences=n_sentences
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
    print()
    
    # Decode
    decoded_message, decoded_bits = modulator.decode_binary(actual_ttr_array)
    accuracy = modulator.calculate_accuracy(actual_ttr_array)
    
    print("=" * 70)
    print("DECODING RESULTS")
    print("=" * 70)
    print(f"Original message:  {message}")
    print(f"Original binary:   {binary_string[:40]}...")
    print(f"Decoded binary:    {decoded_bits[:40]}...")
    print(f"Decoded message:   {decoded_message if decoded_message else 'FAILED'}")
    print(f"Bit accuracy:      {accuracy*100:.1f}%")
    print(f"Decoding accuracy: {'✓ SUCCESS' if decoded_message == message else '✗ FAILED'}")
    print()
    
    # Save JSON
    result = {
        'topic': f"Binary: {message}",
        'message': message,
        'binary_string': binary_string,
        'envelope_frequency': 0.05,  # Not used, but keep for compatibility
        'encoding_type': 'binary_switching',
        'decoded_message': decoded_message,
        'bit_accuracy': accuracy,
        'n_sentences': n_sentences,
        'target_diversities': modulator.target_ttr.tolist(),
        'actual_diversities': list(actual_ttr_array),
        'sentences': sentences,
        'correlation': 0.0  # Not meaningful for binary switching
    }
    
    with open(f'nanogpt_binary_{message}_data.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved: nanogpt_binary_{message}_data.json")
    
    return sentences, decoded_message, accuracy


if __name__ == "__main__":
    print("Binary switching encoding:")
    print("Encodes arbitrary binary data via direct TTR level switching")
    print("High capacity but produces noisy spectrum")
    print()
    
    test_messages = ["HELLO", "SECRET", "AI_RISK"]
    
    for msg in test_messages:
        sentences, decoded, acc = encode_and_generate_binary(
            msg, 
            topic="artificial intelligence safety",
            n_sentences=100
        )
        print("\n" + "="*70 + "\n")
