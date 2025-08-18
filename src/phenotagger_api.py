# -*- coding: utf-8 -*-
"""
PhenoTagger API
Pre-loaded model wrapper for efficient HPO term tagging
"""

import os
import time
import json
import tensorflow as tf
from typing import Dict, List, Tuple, Optional

# Import necessary modules
from nn_model import bioTag_CNN, bioTag_BERT
from dic_ner import dic_ont
from tagging_text import bioTag
from config import PhenoTaggerConfig

class PhenoTaggerAPI:
    """
    Pre-loaded PhenoTagger API for efficient HPO term annotation
    
    This class loads the models once during initialization and provides
    a simple interface for text annotation.
    """
    
    def __init__(self, config: Optional[PhenoTaggerConfig] = None, 
                 model_type: str = None, env_path: str = None):
        """
        Initialize PhenoTagger API with pre-loaded models
        
        Args:
            config: PhenoTaggerConfig instance. If None, creates default config
            model_type: Model type to use ('cnn', 'bioformer', 'pubmedbert', 'biobert')
            env_path: Path to .env file for configuration
        """
        print("Initializing PhenoTagger API...")
        
        # Setup configuration
        if config is None:
            config = PhenoTaggerConfig(env_path=env_path)
        self.config = config
        
        # Override model type if specified
        if model_type:
            self.config.processing_params['model_type'] = model_type
        
        # Setup GPU based on config
        self._setup_gpu()
        
        # Validate paths before loading
        self._validate_setup()
        
        # Load dictionary and models
        self._load_dictionary()
        self._load_model()
        
        print("PhenoTagger API initialized successfully!")
    
    def _setup_gpu(self):
        """Setup GPU memory growth based on configuration"""
        try:
            if self.config.api_settings['gpu_memory_growth']:
                gpu = tf.config.list_physical_devices('GPU')
                print(f"Num GPUs Available: {len(gpu)}")
                if len(gpu) > 0:
                    tf.config.experimental.set_memory_growth(gpu[0], True)
            else:
                print("GPU memory growth disabled by configuration")
        except Exception as e:
            print(f"GPU setup warning: {e}")
    
    def _validate_setup(self):
        """Validate that all required files exist"""
        validation_results = self.config.validate_paths()
        
        missing_files = []
        for path_name, exists in validation_results.items():
            if not exists:
                missing_files.append(path_name)
        
        if missing_files:
            print("Warning: The following files/paths are missing:")
            for missing in missing_files:
                print(f"  - {missing}")
            print("Please check your .env configuration and file paths.")
            # Don't raise error, just warn - some files might be optional
    
    def _load_dictionary(self):
        """Load ontology dictionary"""
        print("Loading ontology dictionary...")
        start_time = time.time()
        
        self.biotag_dic = dic_ont(self.config.ontology_files)
        
        print(f"Dictionary loaded in {time.time() - start_time:.2f} seconds")
    
    def _load_model(self):
        """Load neural network model based on configuration"""
        model_type = self.config.processing_params['model_type']
        model_config = self.config.get_model_config(model_type)
        
        print(f"Loading {model_type} model...")
        start_time = time.time()
        
        # Load model based on type
        if model_type == 'cnn':
            self.nn_model = bioTag_CNN(model_config['vocab_files'])
        else:
            self.nn_model = bioTag_BERT(model_config['vocab_files'])
        
        # Load pre-trained weights
        self.nn_model.load_model(model_config['model_file'])
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def annotate_text(self, text: str, 
                     only_longest: Optional[bool] = None,
                     abbr_recognition: Optional[bool] = None,
                     threshold: Optional[float] = None) -> Dict[str, str]:
        """
        Annotate text with HPO terms
        
        Args:
            text: Input text to annotate
            only_longest: Whether to return only longest concepts (overrides config)
            abbr_recognition: Whether to identify abbreviations (overrides config)
            threshold: ML model threshold (overrides config)
            
        Returns:
            Dictionary with text as key and annotation info as value
            Format: {
                text: "hpo_term1 (HPO:xxxxx);hpo_term2 (HPO:xxxxx);...",
                "hpo_ids": "HPO:xxxxx;HPO:xxxxx;...",
                "raw_results": [...] # Raw tagging results for internal use
            }
        """
        if not text or not text.strip():
            return {
                text: "",
                "hpo_ids": "",
                "raw_results": []
            }
        
        # Use provided parameters or fall back to config defaults
        params = self.config.processing_params.copy()
        if only_longest is not None:
            params['onlyLongest'] = only_longest
        if abbr_recognition is not None:
            params['abbrRecog'] = abbr_recognition
        if threshold is not None:
            params['ML_Threshold'] = threshold
        
        try:
            # Perform tagging
            tag_result = bioTag(
                text,
                self.biotag_dic,
                self.nn_model,
                onlyLongest=params['onlyLongest'],
                abbrRecog=params['abbrRecog'],
                Threshold=params['ML_Threshold']
            )
            
            # Convert results to required format
            hpo_terms, hpo_ids = self._format_results(tag_result)
            
            return {
                text: hpo_terms,
                "hpo_ids": hpo_ids,
                "raw_results": tag_result
            }
            
        except Exception as e:
            print(f"Error during annotation: {e}")
            return {
                text: "",
                "hpo_ids": "",
                "raw_results": []
            }
    
    def _format_results(self, tag_result: List[List[str]]) -> Tuple[str, str]:
        """
        Format tagging results into required string formats
        
        Args:
            tag_result: List of [start, end, hpo_id, score] from bioTag
            
        Returns:
            Tuple of (formatted_terms, hpo_ids_only)
            - formatted_terms: "hpo_term1 (HPO:xxxxx);hpo_term2 (HPO:xxxxx);..."
            - hpo_ids_only: "HPO:xxxxx;HPO:xxxxx;..."
        """
        if not tag_result:
            return "", ""
        
        formatted_terms = []
        hpo_ids = []
        
        for result in tag_result:
            if len(result) >= 3:
                hpo_id = result[2]  # HPO ID
                
                # Collect HPO ID
                hpo_ids.append(hpo_id)
                
                # Get HPO term name from dictionary
                hpo_term = self._get_hpo_term_name(hpo_id)
                
                # Format as "term (HPO:xxxxx)"
                if hpo_term:
                    formatted_term = f"{hpo_term} ({hpo_id})"
                else:
                    formatted_term = f"({hpo_id})"
                
                formatted_terms.append(formatted_term)
        
        return ";".join(formatted_terms), ";".join(hpo_ids)
    
    def _get_hpo_term_name(self, hpo_id: str) -> str:
        """
        Get HPO term name from HPO ID
        
        Args:
            hpo_id: HPO identifier (e.g., "HP:0000001")
            
        Returns:
            HPO term name or empty string if not found
        """
        try:
            if hasattr(self.biotag_dic, 'hpo_word') and hpo_id in self.biotag_dic.hpo_word:
                # Return the first (primary) term name
                term_names = self.biotag_dic.hpo_word[hpo_id]
                if term_names and len(term_names) > 0:
                    return term_names[0]
            return ""
        except Exception:
            return ""
    
    def annotate_batch(self, texts: List[str], **kwargs) -> Dict[str, Dict[str, str]]:
        """
        Annotate multiple texts
        
        Args:
            texts: List of texts to annotate
            **kwargs: Parameters passed to annotate_text
            
        Returns:
            Dictionary mapping each text to its annotation info
            Format: {
                text1: {
                    "hpo_terms": "term1 (HP:xxx);term2 (HP:xxx)",
                    "hpo_ids": "HP:xxx;HP:xxx"
                },
                ...
            }
        """
        results = {}
        
        for text in texts:
            try:
                result = self.annotate_text(text, **kwargs)
                results[text] = {
                    "hpo_terms": result[text],
                    "hpo_ids": result["hpo_ids"]
                }
            except Exception as e:
                print(f"Error annotating text '{text[:50]}...': {e}")
                results[text] = {
                    "hpo_terms": "",
                    "hpo_ids": ""
                }
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about loaded model and configuration
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.config.processing_params['model_type'],
            'threshold': str(self.config.processing_params['ML_Threshold']),
            'only_longest': str(self.config.processing_params['onlyLongest']),
            'abbr_recognition': str(self.config.processing_params['abbrRecog']),
            'api_version': '2.0'
        }


def create_api(model_type: str = None, 
               env_path: str = None,
               threshold: float = None,
               **kwargs) -> PhenoTaggerAPI:
    """
    Convenience function to create PhenoTagger API instance
    
    Args:
        model_type: Type of model to use (overrides .env setting)
        env_path: Path to .env file
        threshold: ML model threshold (overrides .env setting)
        **kwargs: Additional parameters to override
        
    Returns:
        Initialized PhenoTaggerAPI instance
    """
    # Create config from .env
    config = PhenoTaggerConfig(env_path=env_path)
    
    # Override specific parameters
    if model_type:
        config.processing_params['model_type'] = model_type
    if threshold:
        config.processing_params['ML_Threshold'] = threshold
    
    # Update any additional parameters
    config.update_processing_params(**kwargs)
    
    return PhenoTaggerAPI(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the API
    print("Testing PhenoTagger API...")
    
    try:
        # Initialize API
        api = create_api(model_type='bioformer', threshold=0.95)
        
        # Test text
        test_text = "The patient presented with seizures, intellectual disability, and microcephaly."
        
        # Annotate text
        print(f"\nInput text: {test_text}")
        result = api.annotate_text(test_text)
        print(f"HPO annotations: {result[test_text]}")
        
        # Test batch annotation
        test_texts = [
            "Patient has fever and headache.",
            "Observed growth retardation and developmental delay."
        ]
        
        batch_results = api.annotate_batch(test_texts)
        print("\nBatch results:")
        for text, annotation in batch_results.items():
            print(f"Text: {text}")
            print(f"HPO: {annotation}\n")
            
        # Model info
        info = api.get_model_info()
        print("Model info:", info)
        
        # Environment info
        env_info = api.config.get_env_info()
        print("Environment info:", env_info)
        
    except Exception as e:
        print(f"Error testing API: {e}")
        import traceback
        traceback.print_exc()