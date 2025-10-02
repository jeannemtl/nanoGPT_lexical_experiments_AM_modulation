"""
Binary Message Encoding via TTR Modulation
Encodes actual recoverable messages in frequency patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

class BinaryMessageEncoder:
    """Encodes binary messages into TTR patterns"""
    
    def __init__(self, n_sentences=100, carrier_freq=1/3, 
                 ttr_min=0.4, ttr_max=0.9, bits_per_cycle=1):
        self.n_sentences = n_sentences
        self.carrier_freq = carrier_freq
        self.ttr_min = ttr_min
        self.ttr_max = ttr_max
        self.bits_per_cycle = bits_per_cycle
        self.sentences_per_bit = int(1 / carrier_freq)  # ~3 sentences per bit
        
    def message_to_binary(self, message):
        """Convert string message to binary"""
        return ''.join(format(ord(c), '08b') for c in message)
    
    def binary_to_message(self, binary):
        """Convert binary to string message"""
        chars = []
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        return ''.join(chars)
    
    def encode_message(self, message):
        """Generate TTR target pattern encoding the message"""
        binary = self.message_to_binary(message)
        max_bits = self.n_sentences // self.sentences_per_bit
        
        if len(binary) > max_bits:
            print(f"Warning: Message too long. Truncating to {max_bits} bits ({max_bits//8} chars)")
            binary = binary[:max_bits]
        
        print(f"Encoding message: '{message}'")
        print(f"Binary: {binary} ({len(binary)} bits)")
        print(f"Capacity: {max_bits} bits ({max_bits//8} chars)")
        
        target_ttr = np.zeros(self.n_sentences)
        
        n = np.arange(self.n_sentences)
        carrier = np.cos(2 * np.pi * self.carrier_freq * n)
        
        # Modulate carrier based on bits
        for i, bit in enumerate(binary):
            start_idx = i * self.sentences_per_bit
            end_idx = min(start_idx + self.sentences_per_bit, self.n_sentences)
            
            if bit == '1':
                # High TTR for bit 1
                target_ttr[start_idx:end_idx] = self.ttr_max
            else:
                # Low TTR for bit 0
                target_ttr[start_idx:end_idx] = self.ttr_min
        
        # Smooth transitions with carrier
        for i in range(self.n_sentences):
            # Apply carrier modulation on top of bit encoding
            modulation = 0.1 * carrier[i]  # Small carrier modulation
            target_ttr[i] = np.clip(target_ttr[i] + modulation, self.ttr_min, self.ttr_max)
        
        return target_ttr, binary
    
    def decode_message(self, actual_ttr):
        """Extract binary message from TTR sequence"""
        binary_decoded = []
        
        for i in range(0, len(actual_ttr), self.sentences_per_bit):
            chunk = actual_ttr[i:i+self.sentences_per_bit]
            avg_ttr = np.mean(chunk)
            
            # Threshold decision
            threshold = (self.ttr_min + self.ttr_max) / 2
            bit = '1' if avg_ttr > threshold else '0'
            binary_decoded.append(bit)
        
        binary_str = ''.join(binary_decoded)
        message = self.binary_to_message(binary_str)
        
        return message, binary_str
    
    def calculate_ttr(self, text):
        """Calculate type-token ratio"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)


class MessageGenerator:
    """Generate text with encoded binary messages"""
    
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
        self.unique_adverbs = ['accordingly', 'additionally', 'alternatively', 'certainly',
                              'consequently', 'conversely', 'furthermore', 'hence', 
                              'however', 'indeed', 'likewise', 'meanwhile', 'moreover',
                              'nevertheless', 'nonetheless', 'subsequently', 'therefore']
        print(f"✓ Model loaded on {self.device}")
    
    def calculate_ttr(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return len(set(words)) / len(words) if words else 0
    
    def generate_sentence(self, step, target_ttr, topic, max_attempts=8, tolerance=0.10):
        """Generate sentence matching target TTR"""
        
        best_sentence = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            if target_ttr < 0.6:
                sentence = self._generate_low_ttr(step, target_ttr, topic, attempt)
            else:
                sentence = self._generate_high_ttr(step, target_ttr, topic, attempt)
            
            actual_ttr = self.calculate_ttr(sentence)
            
            # Adjust if needed
            if actual_ttr < target_ttr - 0.08:
                sentence = self._adjust_ttr_up(sentence, target_ttr)
            elif actual_ttr > target_ttr + 0.08:
                sentence = self._adjust_ttr_down(sentence, target_ttr)
            
            actual_ttr = self.calculate_ttr(sentence)
            error = abs(actual_ttr - target_ttr)
            
            if error < best_error:
                best_error = error
                best_sentence = sentence
            
            if error < tolerance:
                break
        
        return best_sentence
    
    def _generate_low_ttr(self, step, target_ttr, topic, attempt):
        """Generate low TTR sentence"""
        if attempt < 2:
            templates = [
                f"In step {step} the {topic} and the {topic} and the {topic}.",
                f"The {topic} is the {topic} and the {topic} is clear.",
            ]
            return random.choice(templates)
        else:
            return self._generate_with_gpt2(step, topic, temperature=0.4, top_p=0.6)
    
    def _generate_high_ttr(self, step, target_ttr, topic, attempt):
        """Generate high TTR sentence"""
        return self._generate_with_gpt2(step, topic, temperature=0.9, top_p=0.95, 
                                       no_repeat_ngram_size=3)
    
    def _generate_with_gpt2(self, step, topic, temperature, top_p, no_repeat_ngram_size=None):
        """Generate with GPT-2"""
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


def encode_and_generate(message, topic="artificial intelligence", n_sentences=100):
    """Full encode-generate-decode pipeline"""
    
    print("=" * 70)
    print("BINARY MESSAGE ENCODING IN TTR MODULATION")
    print("=" * 70)
    print()
    
    # Encode
    encoder = BinaryMessageEncoder(n_sentences=n_sentences)
    target_ttr, binary = encoder.encode_message(message)
    print()
    
    # Generate
    generator = MessageGenerator()
    print()
    print(f"Generating {n_sentences} sentences...")
    
    sentences = []
    actual_ttrs = []
    
    for step in range(n_sentences):
        target = target_ttr[step]
        sentence = generator.generate_sentence(step, target, topic)
        actual = encoder.calculate_ttr(sentence)
        
        sentences.append(sentence)
        actual_ttrs.append(actual)
        
        if (step + 1) % 10 == 0:
            print(f"  [{step+1}/{n_sentences}] Target: {target:.2f} | Actual: {actual:.2f}")
    
    actual_ttr_array = np.array(actual_ttrs)
    print()
    
    # Decode
    decoded_message, decoded_binary = encoder.decode_message(actual_ttr_array)
    
    print("=" * 70)
    print("DECODING RESULTS")
    print("=" * 70)
    print(f"Original message: '{message}'")
    print(f"Original binary:  {binary}")
    print(f"Decoded binary:   {decoded_binary[:len(binary)]}")
    print(f"Decoded message:  '{decoded_message[:len(message)]}'")
    
    # Calculate accuracy
    correct_bits = sum(1 for i in range(min(len(binary), len(decoded_binary))) 
                      if binary[i] == decoded_binary[i])
    accuracy = correct_bits / len(binary) * 100 if binary else 0
    print(f"Bit accuracy: {correct_bits}/{len(binary)} ({accuracy:.1f}%)")
    print()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    steps = np.arange(n_sentences)
    ax1.plot(steps, target_ttr, 'b-', label='Target TTR (Encoded)', linewidth=2, alpha=0.7)
    ax1.plot(steps, actual_ttr_array, 'r--', label='Actual TTR', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Sentence Number')
    ax1.set_ylabel('TTR')
    ax1.set_title(f'Message Encoding: "{message}"')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Show bit boundaries
    for i in range(0, len(binary), 1):
        x_pos = i * encoder.sentences_per_bit
        if x_pos < n_sentences:
            ax1.axvline(x_pos, color='gray', linestyle=':', alpha=0.3)
    
    # Decoded bits visualization
    bit_positions = np.arange(len(decoded_binary))
    bit_values = [int(b) for b in decoded_binary]
    ax2.step(bit_positions, bit_values, 'g-', where='mid', linewidth=2, label='Decoded bits')
    
    # Original bits
    original_values = [int(b) for b in binary] + [0] * (len(decoded_binary) - len(binary))
    ax2.step(bit_positions[:len(binary)], original_values[:len(binary)], 'b--', 
             where='mid', linewidth=2, alpha=0.7, label='Original bits')
    
    ax2.set_xlabel('Bit Position')
    ax2.set_ylabel('Bit Value')
    ax2.set_title(f'Binary Decoding (Accuracy: {accuracy:.1f}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'message_encoding_{message}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: message_encoding_{message}.png")
    
    return sentences, decoded_message, accuracy


if __name__ == "__main__":
    # Test with different messages
    messages = ["HELLO", "AI_RISK", "SECRET"]
    
    for msg in messages:
        sentences, decoded, accuracy = encode_and_generate(msg, n_sentences=100)
        print()
