import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class S1Model:
    def __init__(self, model_name="simplescaling/s1.1-1.5B", device=None, cache_dir="model_cache"):
        # Automatically detect the best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load or create cached model
        self.tokenizer, self.model = self._load_or_cache_model()
        
    def _load_or_cache_model(self):
        """Load model from cache if available, otherwise download and cache it"""
        cache_path = os.path.join(self.cache_dir, self.model_name.replace('/', '_'))
        
        if os.path.exists(cache_path):
            print(f"Loading model from cache: {cache_path}")
            tokenizer = AutoTokenizer.from_pretrained(cache_path)
            model = AutoModelForCausalLM.from_pretrained(
                cache_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
        else:
            print(f"Downloading model {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            # Cache the model
            print(f"Caching model to: {cache_path}")
            tokenizer.save_pretrained(cache_path)
            model.save_pretrained(cache_path)
            
        return tokenizer, model
    
    def query(self, prompt, max_length=100):
        """Generate response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    # Initialize the model
    s1 = S1Model()
    
    # Example query
    prompt = "What is the capital of France?"
    response = s1.query(prompt)
    print(f"\nPrompt: {prompt}\nResponse: {response}") 