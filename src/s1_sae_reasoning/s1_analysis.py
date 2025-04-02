# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

class ActivationRecorder:
    def __init__(self, model_name="simplescaling/s1.1-32B", device="cuda", layer_indices=None):
        self.model_name = model_name
        self.device = device
        self.layer_indices = layer_indices
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        config = hf_model.config
        
        # Create TransformerLens config
        tl_config = HookedTransformerConfig(
            n_layers=config.num_hidden_layers,
            d_model=config.hidden_size,
            n_ctx=config.max_position_embeddings,
            d_head=config.hidden_size // config.num_attention_heads,
            n_heads=config.num_attention_heads,
            d_mlp=config.intermediate_size,
            d_vocab=config.vocab_size,
            act_fn=config.hidden_act,
            model_name=model_name,
            normalization_type="LN",
            use_hook_tokens=True,  # Enable hooks for token embeddings
            device=device
        )
        
        # Create a TransformerLens model from the HF model
        self.model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            config=tl_config,
        )
        
        # Set model to eval mode
        self.model.eval()
        
        # Storage for activations
        self.activations = {}
        
    def record_activations(self, prompts, save_dir="activations", batch_size=1):
        """
        Record activations for a list of prompts using TransformerLens hooks.
        
        Args:
            prompts: List of text prompts
            save_dir: Directory to save activations
            batch_size: Batch size for processing
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine which activation names to record based on layer indices
        if not self.layer_indices:
            num_layers = self.model.cfg.n_layers
            self.layer_indices = list(range(num_layers-4, num_layers))
        
        # Define hook names to capture
        hook_names = []
        for layer_idx in self.layer_indices:
            # Hook for MLP output
            hook_names.append(f"blocks.{layer_idx}.mlp.output")
            # Hook for attention output 
            hook_names.append(f"blocks.{layer_idx}.attn.output")
            # Hook for residual stream after attention
            hook_names.append(f"blocks.{layer_idx}.hook_resid_post")
        
        # Process prompts in batches
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            batch_activations = {hook_name: [] for hook_name in hook_names}
            
            for prompt in tqdm(batch_prompts, desc=f"Batch {batch_idx//batch_size + 1}"):
                # Tokenize
                tokens = self.model.to_tokens(prompt)
                
                # Define hook function that saves activations
                def save_activation(tensor, hook):
                    # Convert to numpy and save
                    activation = tensor.detach().cpu().numpy()
                    batch_activations[hook.name].append(activation)
                    return tensor  # Pass through the activation unchanged
                
                # Run model with hooks
                with self.model.hooks(hook_names, save_activation):
                    self.model(tokens)
            
            # Save this batch's activations
            self._save_activations(batch_activations, save_dir, batch_idx)
    
    def _save_activations(self, batch_activations, save_dir, batch_idx):
        """Helper method to save activations to disk."""
        for hook_name, activations in batch_activations.items():
            if not activations:
                continue
                
            # Replace dots with underscores for filenames
            safe_name = hook_name.replace(".", "_")
            
            # Concatenate all activations for this hook
            concatenated = np.concatenate(activations, axis=0)
            
            # Save to disk
            save_path = os.path.join(save_dir, f"{safe_name}_batch_{batch_idx}.npy")
            np.save(save_path, concatenated)
            print(f"Saved activations to {save_path}, shape: {concatenated.shape}")
    
    def get_layer_name_mapping(self):
        """Returns a mapping between TransformerLens hook names and user-friendly names"""
        mapping = {}
        for layer_idx in self.layer_indices:
            mapping[f"blocks.{layer_idx}.mlp.output"] = f"mlp_output_layer_{layer_idx}"
            mapping[f"blocks.{layer_idx}.attn.output"] = f"attn_output_layer_{layer_idx}"
            mapping[f"blocks.{layer_idx}.hook_resid_post"] = f"residual_post_layer_{layer_idx}"
        return mapping

# Example usage
if __name__ == "__main__":
    # Initialize the activation recorder 
    recorder = ActivationRecorder(model_name="simplescaling/s1.1-32B", device="cuda")
    
    # Example reasoning prompts
    reasoning_prompts = [
        "Solve this step by step: If 5x + 3 = 18, what is the value of x?",
        "Explain how to determine whether 17 is a prime number.",
        "A train travels at 60 mph. How far will it travel in 2.5 hours? Show your work.",
    ]
    
    # Record activations
    recorder.record_activations(
        prompts=reasoning_prompts,
        save_dir="./transformer_lens_activations",
        batch_size=1  
    )
    
    print("Activation recording complete!")