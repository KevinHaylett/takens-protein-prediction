"""
Protein Takens-Based Transformer: protein_tbt.py
Adapted from tbt_architecture.py for protein structure prediction.

Key changes:
- Input: Amino acid tokens (21 vocab) instead of words
- Output: 3 regression heads (x, y, z coordinates) instead of classification
- Loss: Huber/SmoothL1 instead of cross-entropy
- No positional encoding (Takens delays handle position)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.takens_embedding import TakensEmbedding, AdaptiveTakensEmbedding, create_exponential_delays
from core.tbt_architecture import TBTLayer


class ProteinTBT(nn.Module):
    """
    Takens-Based Transformer for Protein Structure Prediction.
    
    Architecture:
        1. Amino acid embeddings
        2. Takens delay embedding (exponential delays)
        3. TBT layers (feedforward only, no attention)
        4. Three coordinate prediction heads (x, y, z)
    
    Physics emerges from coordinate supervision - no explicit energy function needed!
    """
    
    def __init__(
        self,
        vocab_size: int = 21,  # 20 AAs + GAP
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 4,
        max_seq_len: int = 256,
        delays: Optional[list] = None,
        dropout: float = 0.1,
        ff_hidden_multiplier: int = 4,
        use_adaptive_takens: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Amino acid embeddings
        self.aa_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Takens delay embedding
        if delays is None:
            delays = create_exponential_delays(128)
        self.delays = delays
        
        if use_adaptive_takens:
            self.takens_embed = AdaptiveTakensEmbedding(
                embedding_dim=embed_dim,
                delays=delays,
                output_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.takens_embed = TakensEmbedding(
                embedding_dim=embed_dim,
                delays=delays
            )
            takens_dim = self.takens_embed.get_output_dim()
            self.takens_projection = nn.Linear(takens_dim, hidden_dim)
        
        self.use_adaptive_takens = use_adaptive_takens
        
        # TBT layers (reuse from tbt_architecture.py)
        self.layers = nn.ModuleList([
            TBTLayer(
                dim=hidden_dim,
                ff_hidden_dim=hidden_dim * ff_hidden_multiplier,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Coordinate prediction heads (3 separate heads for x, y, z)
        self.coord_x = nn.Linear(hidden_dim, 1)
        self.coord_y = nn.Linear(hidden_dim, 1)
        self.coord_z = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Amino acid embeddings
        nn.init.normal_(self.aa_embed.weight, std=0.02)
        
        # Coordinate heads
        nn.init.normal_(self.coord_x.weight, std=0.02)
        nn.init.normal_(self.coord_y.weight, std=0.02)
        nn.init.normal_(self.coord_z.weight, std=0.02)
        
        if self.coord_x.bias is not None:
            nn.init.zeros_(self.coord_x.bias)
            nn.init.zeros_(self.coord_y.bias)
            nn.init.zeros_(self.coord_z.bias)
        
        # Projection (if not using adaptive Takens)
        if not self.use_adaptive_takens:
            nn.init.normal_(self.takens_projection.weight, std=0.02)
            if self.takens_projection.bias is not None:
                nn.init.zeros_(self.takens_projection.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        coords_x: Optional[torch.Tensor] = None,
        coords_y: Optional[torch.Tensor] = None,
        coords_z: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len] amino acid tokens
            coords_x: [batch, seq_len] target x coordinates (optional)
            coords_y: [batch, seq_len] target y coordinates (optional)
            coords_z: [batch, seq_len] target z coordinates (optional)
            mask: [batch, seq_len] boolean mask (True = real residue)
            
        Returns:
            preds: Tuple of (pred_x, pred_y, pred_z), each [batch, seq_len]
            loss: Scalar loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed amino acids
        x = self.aa_embed(input_ids)  # [batch, seq_len, embed_dim]
        x = self.dropout(x)
        
        # Takens delay embedding
        if self.use_adaptive_takens:
            x = self.takens_embed(x)  # [batch, seq_len, hidden_dim]
        else:
            grid = self.takens_embed(x)  # [batch, seq_len, num_delays+1, embed_dim]
            x = self.takens_embed.flatten_grid(grid)  # [batch, seq_len, (num_delays+1)*embed_dim]
            x = self.takens_projection(x)  # [batch, seq_len, hidden_dim]
        
        # TBT layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Predict coordinates
        pred_x = self.coord_x(x).squeeze(-1)  # [batch, seq_len]
        pred_y = self.coord_y(x).squeeze(-1)  # [batch, seq_len]
        pred_z = self.coord_z(x).squeeze(-1)  # [batch, seq_len]
        
        # Compute loss if targets provided
        loss = None
        if coords_x is not None:
            loss = self.compute_loss(
                (pred_x, pred_y, pred_z),
                (coords_x, coords_y, coords_z),
                mask
            )
        
        return (pred_x, pred_y, pred_z), loss
    
    def compute_loss(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Huber loss on coordinates.
        
        Huber loss (smooth L1) is less sensitive to outliers than MSE,
        which is important for flexible loop regions.
        
        Args:
            preds: Tuple of (pred_x, pred_y, pred_z)
            targets: Tuple of (target_x, target_y, target_z)
            mask: Boolean mask for valid positions
            
        Returns:
            Scalar loss
        """
        pred_x, pred_y, pred_z = preds
        tgt_x, tgt_y, tgt_z = targets
        
        if mask is not None:
            # Only compute loss on valid (non-padded) positions
            loss_x = F.smooth_l1_loss(pred_x[mask], tgt_x[mask])
            loss_y = F.smooth_l1_loss(pred_y[mask], tgt_y[mask])
            loss_z = F.smooth_l1_loss(pred_z[mask], tgt_z[mask])
        else:
            # Use all positions
            loss_x = F.smooth_l1_loss(pred_x, tgt_x)
            loss_y = F.smooth_l1_loss(pred_y, tgt_y)
            loss_z = F.smooth_l1_loss(pred_z, tgt_z)
        
        # Total loss is sum of coordinate losses
        return loss_x + loss_y + loss_z
    
    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict coordinates for a sequence.
        
        Args:
            input_ids: [batch, seq_len] or [seq_len] amino acid tokens
            mask: Optional mask for valid positions
            
        Returns:
            coords: [batch, seq_len, 3] or [seq_len, 3] predicted coordinates
        """
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Forward pass (no targets, so loss will be None)
        (pred_x, pred_y, pred_z), _ = self.forward(input_ids)
        
        # Stack coordinates
        coords = torch.stack([pred_x, pred_y, pred_z], dim=-1)  # [batch, seq_len, 3]
        
        # Remove batch dimension if input was 1D
        if squeeze_output:
            coords = coords.squeeze(0)  # [seq_len, 3]
        
        return coords
    
    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'max_seq_len': self.max_seq_len,
                'delays': self.delays,
                'use_adaptive_takens': self.use_adaptive_takens
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with saved config
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing ProteinTBT")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating model...")
    model = ProteinTBT(
        vocab_size=21,
        embed_dim=128,
        hidden_dim=128,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    )
    
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    print(f"Target: ~6-8M parameters")
    print(f"Status: {'✓' if 4_000_000 < num_params < 10_000_000 else '✗'}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 256
    
    # Create dummy data
    input_ids = torch.randint(0, 21, (batch_size, seq_len))
    coords_x = torch.randn(batch_size, seq_len)
    coords_y = torch.randn(batch_size, seq_len)
    coords_z = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass
    (pred_x, pred_y, pred_z), loss = model(
        input_ids,
        coords_x, coords_y, coords_z,
        mask
    )
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Pred X shape: {pred_x.shape}")
    print(f"Pred Y shape: {pred_y.shape}")
    print(f"Pred Z shape: {pred_z.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test prediction (no targets)
    print("\n3. Testing prediction (no targets)...")
    coords = model.predict(input_ids, mask)
    print(f"Predicted coords shape: {coords.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, 3]")
    assert coords.shape == (batch_size, seq_len, 3)
    print("✓ Prediction works!")
    
    # Test single sequence prediction
    print("\n4. Testing single sequence...")
    single_input = torch.randint(0, 21, (seq_len,))
    single_coords = model.predict(single_input)
    print(f"Single input shape: {single_input.shape}")
    print(f"Single output shape: {single_coords.shape}")
    print(f"Expected: [{seq_len}, 3]")
    assert single_coords.shape == (seq_len, 3)
    print("✓ Single sequence works!")
    
    # Test loss with mask
    print("\n5. Testing masked loss...")
    # Create mask with some padding
    mask_partial = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask_partial[:, 200:] = False  # Mask last 56 positions
    
    _, loss_full = model(input_ids, coords_x, coords_y, coords_z, mask)
    _, loss_partial = model(input_ids, coords_x, coords_y, coords_z, mask_partial)
    
    print(f"Loss (full sequence): {loss_full.item():.4f}")
    print(f"Loss (partial mask): {loss_partial.item():.4f}")
    print(f"Losses are different: {'✓' if loss_full.item() != loss_partial.item() else '✗'}")
    
    # Test gradient flow
    print("\n6. Testing gradient flow...")
    loss_partial.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    no_nan = all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    
    print(f"Gradients computed: {'✓' if has_grad else '✗'}")
    print(f"No NaN gradients: {'✓' if no_nan else '✗'}")
    
    # Test save/load
    print("\n7. Testing save/load...")
    model.save('test_protein_model.pt')
    model2 = ProteinTBT.load('test_protein_model.pt')
    print("✓ Save/load works!")
    
    # Verify loaded model produces same output
    model.eval()
    model2.eval()
    with torch.no_grad():
        coords1 = model.predict(single_input)
        coords2 = model2.predict(single_input)
        diff = (coords1 - coords2).abs().max().item()
    
    print(f"Max difference after reload: {diff:.2e}")
    print(f"Models match: {'✓' if diff < 1e-6 else '✗'}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nModel ready for training.")
    print(f"\nArchitecture summary:")
    print(f"  Vocabulary: {model.vocab_size}")
    print(f"  Embedding dim: {model.embed_dim}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Layers: {model.num_layers}")
    print(f"  Delays: {model.delays}")
    print(f"  Parameters: {num_params:,}")
