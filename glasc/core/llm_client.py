"""
GLASC: Global Leverage & Asset Strategy Controller
Module: LLM Client

Wrapper d'abstraction pour le Large Language Model.
Gère le chargement du modèle local (HuggingFace Transformers) ou l'appel API.
"""

import logging
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from glasc.utils.config_loader import load_config

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.config = load_config()
        self.provider = self.config.llm.provider.upper()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if self.provider == "LOCAL":
            self._init_local_model()
        else:
            raise NotImplementedError(f"Provider {self.provider} not yet implemented.")

    def _init_local_model(self):
        model_path = self.config.llm.model_path
        logger.info(f"Loading local model from {model_path}...")
        
        try:
            # Force CPU if needed, or use Auto
            device_map = "auto" if self.config.system.device == "auto" else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float32 # Use float32 for CPU compatibility if bfloat16 issues arise
            )
            
            logger.info("Local model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise e

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Génère une réponse à partir d'un prompt système et utilisateur.
        """
        if self.provider == "LOCAL":
            return self._generate_local(system_prompt, user_prompt)
        return ""

    def _generate_local(self, system_prompt: str, user_prompt: str) -> str:
        # Format ChatML for Qwen
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            do_sample=True # Needed for temperature > 0
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

# Singleton instance setup if needed, but for now we instantiate on demand.
