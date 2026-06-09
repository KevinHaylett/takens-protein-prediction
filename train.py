"""
train.py - Protein Structure Prediction Training
=================================================
Takens-Based Transformer (MARINA Architecture)
Kevin R. Haylett, PhD - Manchester, UK

Usage:
    python train.py

All paths and parameters are set in config.py.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from protein.protein_encoder import ProteinEncoder
from protein.protein_dataset import ProteinDataset, collate_batch
from pipeline.pdb_to_training import PDBToTrainingConverter
from protein_tbt import ProteinTBT
from core.takens_embedding import create_exponential_delays


def convert_csv_to_training_format():
    print("=" * 70)
    print("Converting CSV to Training Format")
    print("=" * 70)
    converter = PDBToTrainingConverter(
        max_seq_len=config.MAX_SEQ_LEN,
        center_coords=True,
        normalize_coords=False
    )
    Path(config.TRAINING_DIR).mkdir(parents=True, exist_ok=True)
    converter.convert_directory(
        input_dir=config.CSV_DIR,
        output_dir=config.TRAINING_DIR,
        save_format='pt'
    )
    print(f"Processed data saved to: {config.TRAINING_DIR}")


def compute_rmsd(preds, targets, mask):
    pred_x, pred_y, pred_z = preds
    tgt_x, tgt_y, tgt_z = targets
    sq_diff = ((pred_x-tgt_x)**2 + (pred_y-tgt_y)**2 + (pred_z-tgt_z)**2)
    if mask is not None:
        sq_diff = sq_diff[mask]
    return torch.sqrt(sq_diff.mean()).item()


def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss, total_rmsd, n = 0.0, 0.0, 0
    print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
    print("-" * 70)
    for idx, batch in enumerate(loader):
        ids  = batch['input_ids'].to(config.DEVICE)
        cx   = batch['coords_x'].to(config.DEVICE)
        cy   = batch['coords_y'].to(config.DEVICE)
        cz   = batch['coords_z'].to(config.DEVICE)
        mask = batch['mask'].to(config.DEVICE)
        preds, loss = model(ids, cx, cy, cz, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        with torch.no_grad():
            rmsd = compute_rmsd(preds, (cx, cy, cz), mask)
        total_loss += loss.item(); total_rmsd += rmsd; n += 1
        if (idx + 1) % config.LOG_INTERVAL == 0 or (idx + 1) == len(loader):
            print(f"  Batch {idx+1}/{len(loader)} - Loss: {total_loss/n:.4f}, RMSD: {total_rmsd/n:.2f}A")
    return total_loss / n, total_rmsd / n


def validate(model, loader):
    model.eval()
    total_loss, total_rmsd, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(config.DEVICE)
            cx   = batch['coords_x'].to(config.DEVICE)
            cy   = batch['coords_y'].to(config.DEVICE)
            cz   = batch['coords_z'].to(config.DEVICE)
            mask = batch['mask'].to(config.DEVICE)
            preds, loss = model(ids, cx, cy, cz, mask)
            rmsd = compute_rmsd(preds, (cx, cy, cz), mask)
            total_loss += loss.item(); total_rmsd += rmsd; n += 1
    return total_loss / n, total_rmsd / n


def save_results(history, best_val_loss, best_val_rmsd, total_time):
    results = {
        "date": datetime.now().isoformat(),
        "architecture": "Takens-Based Transformer (MARINA)",
        "training_config": {
            "max_seq_len": config.MAX_SEQ_LEN,
            "embed_dim": config.EMBED_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "num_layers": config.NUM_LAYERS,
            "max_delay": config.MAX_DELAY,
            "batch_size": config.BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "learning_rate": config.LEARNING_RATE,
        },
        "results": {
            "best_val_loss": round(best_val_loss, 6),
            "best_val_rmsd_A": round(best_val_rmsd, 4),
            "training_time_min": round(total_time / 60, 2),
        },
        "loss_curves": history
    }
    out = Path(config.CHECKPOINT_DIR) / "training_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out}")


def main():
    print("=" * 70)
    print("PROTEIN FOLDING TRAINING")
    print("Takens-Based Transformer (MARINA)")
    print("=" * 70)
    print(f"  CSV data:       {config.CSV_DIR}")
    print(f"  Max seq length: {config.MAX_SEQ_LEN}")
    print(f"  Batch size:     {config.BATCH_SIZE}")
    print(f"  Epochs:         {config.NUM_EPOCHS}")
    print(f"  Device:         {config.DEVICE}")

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    convert_csv_to_training_format()

    encoder = ProteinEncoder()
    print(f"Vocabulary size: {encoder.get_vocab_size()}")

    class ProteinDatasetFixed(ProteinDataset):
        def _load_file(self, file_path):
            try:
                return torch.load(file_path, weights_only=False)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return None

    full_dataset = ProteinDatasetFixed(
        data_dir=config.TRAINING_DIR, encoder=encoder,
        max_seq_len=config.MAX_SEQ_LEN, file_format='pt',
        filter_by_length=False, verbose=True
    )

    if len(full_dataset) == 0:
        print(f"\nERROR: No proteins found in {config.TRAINING_DIR}")
        return

    print(f"Total proteins: {len(full_dataset)}")

    if len(full_dataset) < 10:
        print(f"Small dataset - using all proteins for train and validation")
        train_dataset = val_dataset = full_dataset
    else:
        val_size = max(1, int(config.VAL_SPLIT * len(full_dataset)))
        train_dataset, val_dataset = random_split(
            full_dataset, [len(full_dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED)
        )

    print(f"Training: {len(train_dataset)}  Validation: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_batch, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_batch, num_workers=config.NUM_WORKERS)

    delays = create_exponential_delays(config.MAX_DELAY)
    print(f"Takens delays: {delays}")

    model = ProteinTBT(
        vocab_size=encoder.get_vocab_size(),
        embed_dim=config.EMBED_DIM, hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS, max_seq_len=config.MAX_SEQ_LEN,
        delays=delays, dropout=config.DROPOUT, use_adaptive_takens=True
    ).to(config.DEVICE)
    print(f"Parameters: {model.get_num_params():,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
        betas=(0.9, 0.999), weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=config.LEARNING_RATE * 0.1)

    history = {'train_loss': [], 'train_rmsd': [], 'val_loss': [], 'val_rmsd': [], 'learning_rate': []}
    best_val_loss = best_val_rmsd = float('inf')
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_rmsd = train_epoch(model, train_loader, optimizer, epoch)
        val_loss, val_rmsd = validate(model, val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_rmsd'].append(train_rmsd)
        history['val_loss'].append(val_loss)
        history['val_rmsd'].append(val_rmsd)
        history['learning_rate'].append(lr)
        print(f"\nEpoch {epoch} ({time.time()-t0:.1f}s):")
        print(f"  Train - Loss: {train_loss:.4f}, RMSD: {train_rmsd:.2f}A")
        print(f"  Val   - Loss: {val_loss:.4f},   RMSD: {val_rmsd:.2f}A")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_val_rmsd = val_rmsd
            model.save(Path(config.CHECKPOINT_DIR) / 'best_model.pt')
            print(f"  New best model - RMSD: {best_val_rmsd:.2f}A")
        print("=" * 70)

    total_time = time.time() - start_time
    print(f"\nTraining complete: {total_time/60:.1f} min, best RMSD: {best_val_rmsd:.2f}A")
    save_results(history, best_val_loss, best_val_rmsd, total_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
