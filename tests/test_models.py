"""
Unit tests for MS-HGNN models
"""

import pytest
import torch
import numpy as np
from models import (
    MSHGNN, MultiTaskMSHGNN, ModalityEncoder,
    CrossModalFusion, HeterogeneousGraphFusion,
    SemanticAttention, UncertaintyFusion
)
from config import get_default_config


class TestModalityEncoder:
    """Test modality encoders"""
    
    def test_encoder_forward(self):
        """Test forward pass of modality encoder"""
        encoder = ModalityEncoder(input_dim=100, output_dim=64)
        x = torch.randn(16, 100)
        output = encoder(x)
        
        assert output.shape == (16, 64)
        assert not torch.isnan(output).any()
    
    def test_encoder_output_dim(self):
        """Test encoder output dimension matches config"""
        encoder = ModalityEncoder(input_dim=131, output_dim=64)
        x = torch.randn(16, 131)
        output = encoder(x)
        
        assert output.shape[-1] == 64


class TestCrossModalAttention:
    """Test cross-modal attention"""
    
    def test_attention_forward(self):
        """Test forward pass of cross-modal attention"""
        attention = CrossModalFusion(dim=64, num_heads=4)
        
        embeddings = {
            'ct': torch.randn(16, 64),
            'pet': torch.randn(16, 64),
            'clinical': torch.randn(16, 64),
            'genomic': torch.randn(16, 64)
        }
        
        output = attention(embeddings)
        
        assert output.shape == (16, 256)
        assert not torch.isnan(output).any()


class TestHeterogeneousGraph:
    """Test heterogeneous graph fusion"""
    
    def test_graph_fusion_forward(self):
        """Test forward pass of graph fusion"""
        fusion = HeterogeneousGraphFusion(hidden_dim=128)
        
        features = {
            'ct': torch.randn(16, 131),
            'pet': torch.randn(16, 131),
            'clinical': torch.randn(16, 45),
            'genomic': torch.randn(16, 50)
        }
        
        output = fusion(features)
        
        assert output.shape == (16, 512)
        assert not torch.isnan(output).any()


class TestSemanticAttention:
    """Test semantic attention mechanism"""
    
    def test_attention_forward(self):
        """Test forward pass of semantic attention"""
        attention = SemanticAttention(hidden_dim=512, semantic_dim=128)
        
        meta_path_embeddings = torch.randn(16, 3, 512)
        embedding, beta = attention(meta_path_embeddings)
        
        assert embedding.shape == (16, 512)
        assert beta.shape == (16, 3)
        assert torch.allclose(beta.sum(dim=1), torch.ones(16), rtol=1e-3)
        assert not torch.isnan(embedding).any()


class TestUncertaintyFusion:
    """Test uncertainty-aware fusion"""
    
    def test_fusion_forward(self):
        """Test forward pass of uncertainty fusion"""
        fusion = UncertaintyFusion(num_modalities=4, gamma=1.0)
        
        modality_outputs = {
            'ct': (torch.randn(16, 1), torch.randn(16, 1).abs()),
            'pet': (torch.randn(16, 1), torch.randn(16, 1).abs()),
            'clinical': (torch.randn(16, 1), torch.randn(16, 1).abs()),
            'genomic': (torch.randn(16, 1), torch.randn(16, 1).abs())
        }
        
        output = fusion(modality_outputs)
        
        assert output['prediction'].shape == (16, 1)
        assert output['variance'].shape == (16, 1)
        assert output['confidences'].shape == (16, 4)
        assert output['weights'].shape == (16, 4)
        assert output['ci_lower'].shape == (16, 1)
        assert output['ci_upper'].shape == (16, 1)


class TestMSHGNN:
    """Test complete MS-HGNN model"""
    
    def test_model_forward(self):
        """Test forward pass of MS-HGNN"""
        config = get_default_config()['model']
        model = MSHGNN(config)
        
        batch = {
            'ct': torch.randn(16, 131),
            'pet': torch.randn(16, 131),
            'clinical': torch.randn(16, 45),
            'genomic': torch.randn(16, 50)
        }
        
        output = model(batch)
        
        assert 'survival_prediction' in output
        assert 'recurrence_prediction' in output
        assert output['survival_prediction'].shape == (16, 1)
        assert output['recurrence_prediction'].shape == (16, 1)
        
    def test_model_with_missing_modality(self):
        """Test MS-HGNN with missing modality"""
        config = get_default_config()['model']
        model = MSHGNN(config)
        
        batch = {
            'ct': torch.randn(16, 131),
            'pet': torch.randn(16, 131),
            'clinical': torch.randn(16, 45)
            # genomic is missing
        }
        
        output = model(batch)
        
        assert 'survival_prediction' in output
        assert 'recurrence_prediction' in output
        assert output['survival_prediction'].shape == (16, 1)


class TestMultiTaskMSHGNN:
    """Test multi-task MS-HGNN"""
    
    def test_multitask_forward(self):
        """Test forward pass of multi-task model"""
        config = get_default_config()['model']
        model = MultiTaskMSHGNN(config)
        
        batch = {
            'ct': torch.randn(16, 131),
            'pet': torch.randn(16, 131),
            'clinical': torch.randn(16, 45),
            'genomic': torch.randn(16, 50),
            'survival_time': torch.randn(16),
            'survival_event': torch.randint(0, 2, (16,)).float(),
            'recurrence': torch.randint(0, 2, (16,)).float()
        }
        
        output = model(batch)
        
        assert 'survival_prediction' in output
        assert 'recurrence_prediction' in output
        assert 'loss' in output
        assert not torch.isnan(output['loss'])
    
    def test_multitask_loss_without_labels(self):
        """Test multi-task model without labels"""
        config = get_default_config()['model']
        model = MultiTaskMSHGNN(config)
        
        batch = {
            'ct': torch.randn(16, 131),
            'pet': torch.randn(16, 131),
            'clinical': torch.randn(16, 45),
            'genomic': torch.randn(16, 50)
            # No labels
        }
        
        output = model(batch)
        
        assert 'survival_prediction' in output
        assert 'loss' is None
        assert 'survival_loss' is None
        assert 'recurrence_loss' is None


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
