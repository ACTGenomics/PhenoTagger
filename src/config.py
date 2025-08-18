# -*- coding: utf-8 -*-
"""
PhenoTagger Configuration File
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

class PhenoTaggerConfig:
    """Configuration class for PhenoTagger API"""
    
    def __init__(self, env_path: str = None):
        """
        Initialize configuration from .env file
        
        Args:
            env_path: Path to .env file. If None, looks for .env in current directory
        """
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()  # Load from .env in current directory
        
        # Load configuration from environment variables
        self._load_from_env()
        
        # Build file paths
        self._build_paths()
    
    def _load_from_env(self):
        """Load configuration parameters from environment variables"""
        
        # Helper function to convert string to boolean
        def str_to_bool(value: str) -> bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        
        # Helper function to get env variable with default
        def get_env(key: str, default: Any = None, convert_type: type = str) -> Any:
            value = os.getenv(key, default)
            if convert_type == bool:
                return str_to_bool(str(value))
            elif convert_type == float:
                return float(value)
            elif convert_type == int:
                return int(value)
            return value
        
        # Base paths
        self.base_path = get_env('BASE_PATH', '../')
        self.dict_path = get_env('DICT_PATH', '../dict/')
        self.models_path = get_env('MODELS_PATH', '../models/')
        
        # Processing parameters
        self.processing_params = {
            'model_type': get_env('MODEL_TYPE', 'bioformer'),
            'onlyLongest': get_env('ONLY_LONGEST', False, bool),
            'abbrRecog': get_env('ABBR_RECOGNITION', True, bool),
            'negation': get_env('NEGATION_DETECTION', False, bool),
            'ML_Threshold': get_env('ML_THRESHOLD', 0.95, float)
        }
        
        # File names
        self.file_names = {
            'dic_file': get_env('DIC_FILE', 'noabb_lemma.dic'),
            'word_hpo_file': get_env('WORD_HPO_FILE', 'word_id_map.json'),
            'hpo_word_file': get_env('HPO_WORD_FILE', 'id_word_map.json'),
            'char_vocab_file': get_env('CHAR_VOCAB_FILE', 'char.vocab'),
            'label_vocab_file': get_env('LABEL_VOCAB_FILE', 'lable.vocab'),
            'pos_vocab_file': get_env('POS_VOCAB_FILE', 'pos.vocab'),
            'w2v_file': get_env('W2V_FILE', 'bio_embedding_intrinsic.d200'),
            'cnn_model_file': get_env('CNN_MODEL_FILE', 'cnn_hpo_v1.1.h5'),
            'bioformer_model_file': get_env('BIOFORMER_MODEL_FILE', 'bioformer_PT_v1.2.h5'),
            'pubmedbert_model_file': get_env('PUBMEDBERT_MODEL_FILE', 'pubmedbert_PT.h5'),
            'biobert_model_file': get_env('BIOBERT_MODEL_FILE', 'biobert-PT.h5'),
            'bioformer_checkpoint': get_env('BIOFORMER_CHECKPOINT', 'bioformer-cased-v1.0/'),
            'pubmedbert_checkpoint': get_env('PUBMEDBERT_CHECKPOINT', 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/'),
            'biobert_checkpoint': get_env('BIOBERT_CHECKPOINT', 'biobert-base-cased-v1.2/')
        }
        
        # Model-specific settings
        self.model_settings = {
            'bioformer_lowercase': get_env('BIOFORMER_LOWERCASE', False, bool),
            'pubmedbert_lowercase': get_env('PUBMEDBERT_LOWERCASE', True, bool),
            'biobert_lowercase': get_env('BIOBERT_LOWERCASE', False, bool)
        }
        
        # API settings
        self.api_settings = {
            'version': get_env('API_VERSION', '2.0'),
            'gpu_memory_growth': get_env('GPU_MEMORY_GROWTH', True, bool)
        }
    
    def _build_paths(self):
        """Build full file paths from base paths and file names"""
        
        # Dictionary files
        self.ontology_files = {
            'dic_file': os.path.join(self.dict_path, self.file_names['dic_file']),
            'word_hpo_file': os.path.join(self.dict_path, self.file_names['word_hpo_file']),
            'hpo_word_file': os.path.join(self.dict_path, self.file_names['hpo_word_file'])
        }
        
        # Model configurations for different types
        self.model_configs = {
            'cnn': {
                'vocab_files': {
                    'w2vfile': os.path.join(self.models_path, self.file_names['w2v_file']),
                    'charfile': os.path.join(self.dict_path, self.file_names['char_vocab_file']),
                    'labelfile': os.path.join(self.dict_path, self.file_names['label_vocab_file']),
                    'posfile': os.path.join(self.dict_path, self.file_names['pos_vocab_file'])
                },
                'model_file': os.path.join(self.models_path, self.file_names['cnn_model_file'])
            },
            
            'bioformer': {
                'vocab_files': {
                    'labelfile': os.path.join(self.dict_path, self.file_names['label_vocab_file']),
                    'checkpoint_path': os.path.join(self.models_path, self.file_names['bioformer_checkpoint']),
                    'lowercase': self.model_settings['bioformer_lowercase']
                },
                'model_file': os.path.join(self.models_path, self.file_names['bioformer_model_file'])
            },
            
            'pubmedbert': {
                'vocab_files': {
                    'labelfile': os.path.join(self.dict_path, self.file_names['label_vocab_file']),
                    'checkpoint_path': os.path.join(self.models_path, self.file_names['pubmedbert_checkpoint']),
                    'lowercase': self.model_settings['pubmedbert_lowercase']
                },
                'model_file': os.path.join(self.models_path, self.file_names['pubmedbert_model_file'])
            },
            
            'biobert': {
                'vocab_files': {
                    'labelfile': os.path.join(self.dict_path, self.file_names['label_vocab_file']),
                    'checkpoint_path': os.path.join(self.models_path, self.file_names['biobert_checkpoint']),
                    'lowercase': self.model_settings['biobert_lowercase']
                },
                'model_file': os.path.join(self.models_path, self.file_names['biobert_model_file'])
            }
        }
    
    def reload_from_env(self, env_path: str = None):
        """
        Reload configuration from .env file
        
        Args:
            env_path: Path to .env file
        """
        if env_path:
            load_dotenv(env_path, override=True)
        else:
            load_dotenv(override=True)
        
        self._load_from_env()
        self._build_paths()
    
    def get_model_config(self, model_type: str = None) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        if model_type is None:
            model_type = self.processing_params['model_type']
            
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return self.model_configs[model_type]
    
    def update_processing_params(self, **kwargs):
        """
        Update processing parameters
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if key in self.processing_params:
                self.processing_params[key] = value
            else:
                print(f"Warning: Unknown parameter '{key}' ignored")
    
    def get_env_info(self) -> Dict[str, Any]:
        """
        Get current environment configuration info
        
        Returns:
            Dictionary with current configuration
        """
        return {
            'processing_params': self.processing_params.copy(),
            'base_path': self.base_path,
            'dict_path': self.dict_path,
            'models_path': self.models_path,
            'api_version': self.api_settings['version'],
            'gpu_memory_growth': self.api_settings['gpu_memory_growth']
        }
    
    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that all required paths exist
        
        Returns:
            Dictionary showing which paths exist
        """
        validation_results = {}
        
        # Check base paths
        validation_results['base_path'] = os.path.exists(self.base_path)
        validation_results['dict_path'] = os.path.exists(self.dict_path)
        validation_results['models_path'] = os.path.exists(self.models_path)
        
        # Check dictionary files
        for key, path in self.ontology_files.items():
            validation_results[f'ontology_{key}'] = os.path.exists(path)
        
        # Check model files for current model type
        model_config = self.get_model_config()
        validation_results['current_model_file'] = os.path.exists(model_config['model_file'])
        
        if 'checkpoint_path' in model_config['vocab_files']:
            checkpoint_path = model_config['vocab_files']['checkpoint_path']
            validation_results['current_model_checkpoint'] = os.path.exists(checkpoint_path)
        
        return validation_results