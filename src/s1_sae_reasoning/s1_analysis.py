import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import os

class ModelLoader:
    def __init__(self, model_name, device=None, cache_dir="model_cache"):
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
        
        # First load HF config to get model parameters
        self.hf_config = AutoConfig.from_pretrained(model_name)
        
        # Create HookedTransformer config from HF config
        self.config = HookedTransformerConfig(
            n_layers=self.hf_config.num_hidden_layers,
            d_model=self.hf_config.hidden_size,
            n_ctx=self.hf_config.max_position_embeddings,
            d_head=self.hf_config.hidden_size // self.hf_config.num_attention_heads,
            model_name=model_name,
            n_heads=self.hf_config.num_attention_heads,
            act_fn=self.hf_config.hidden_act.lower(),
            d_vocab=self.hf_config.vocab_size,
            device=self.device,
            dtype=torch.float16
        )

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
    
    def query(self, prompt, max_length=25, top_k=2):
        """Generate response for a given prompt and return top k token scores"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Process scores to get top-k values for each step
            top_scores = []
            for score_tensor in outputs.scores:
                # Get top k values and their indices for each step
                values, indices = torch.topk(score_tensor[0], top_k)
                
                # Convert to tokens and probabilities
                tokens = [self.tokenizer.decode([idx.item()]) for idx in indices]
                probs = values.tolist()  # These are already softmaxed
                
                # Store as list of (token, prob) tuples
                step_scores = list(zip(tokens, probs))
                top_scores.append(step_scores)
            
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), top_scores

# Function to compare logits between models
def compare_model_logits(prompt, model1_name="simplescaling/s1.1-1.5B", model2_name="Qwen/Qwen2.5-1.5B-Instruct", max_length=25, top_k=3):
    """Compare top token probabilities between two models"""
    # Initialize both models
    model1 = ModelLoader(model1_name)
    model2 = ModelLoader(model2_name)
    
    # Get responses and token scores
    model1_response, model1_scores = model1.query(prompt, max_length=max_length, top_k=top_k)
    model2_response, model2_scores = model2.query(prompt, max_length=max_length, top_k=top_k)
    
    # Print responses
    print(f"\nPrompt: {prompt}")
    print(f"\n{model1_name} Response: {model1_response}")
    print(f"\n{model2_name} Response: {model2_response}")
    
    # Determine max steps to compare (may be different lengths)
    max_steps = min(len(model1_scores), len(model2_scores))
    
    # Print comparison of top token probabilities for each step
    print("\nComparison of top token probabilities:")
    for i in range(max_steps):
        print(f"\nStep {i+1}:")
        
        # Print model1 scores
        print(f"  {model1_name} Model:")
        for token, score in model1_scores[i]:
            print(f"    '{token}': {score:.4f}")
            
        # Print model2 scores
        print(f"  {model2_name} Model:")
        for token, score in model2_scores[i]:
            print(f"    '{token}': {score:.4f}")
    
    return model1_response, model2_response, model1_scores, model2_scores


# Updated example usage with comparison
if __name__ == "__main__":
    # Example query to compare both models
    prompt = "What is the capital of France?"
    s1_model_name = "simplescaling/s1.1-1.5B"
    qwen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    s1_response, qwen_response, s1_scores, qwen_scores = compare_model_logits(
        prompt, 
        model1_name=s1_model_name, 
        model2_name=qwen_model_name, 
        max_length=25
    )
    