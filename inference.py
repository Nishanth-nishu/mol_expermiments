#!/usr/bin/env python3
"""
inference.py — Generate 3D conformers from SMILES strings using a trained model checkpoint.

This script takes SMILES strings, extracts the heavy-atom graph, uses the diffusion model
to generate 3D coordinates, adds hydrogen atoms back to the generated 3D structure,
and exports the result as SDF and PDB.

Usage:
    source venv/bin/activate
    python inference.py --smiles "CCO" "c1ccccc1" --checkpoint checkpoints/exp_G_heavy_atom_sota_ddp_best.pt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter, RWMol, Atom

# Import the model architecture
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.conformer_diffusion import ConformerDiffusion
from models.attn_conformer_diffusion import AttnConformerDiffusion
from models.flow_matching import FlowMatchingConformer

BOND_TYPE_TO_RDKIT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}

def load_model(ckpt_path, device):
    """Load the model from the checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    hidden_dim = cfg.get("hidden_dim", 384)
    num_layers = cfg.get("num_layers", 8)
    time_dim   = cfg.get("time_dim", 256)
    timesteps  = cfg.get("timesteps", 1000)

    num_rbf    = cfg.get("num_rbf", 32)

    # We assume ConformerDiffusion for Exp G SOTA. Can be made dynamic.
    # Check the name from the checkpoint if possible, else default to ConformerDiffusion
    model = ConformerDiffusion(num_timesteps=timesteps, hidden_dim=hidden_dim,
                               num_layers=num_layers, time_dim=time_dim, num_rbf=num_rbf)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded model from {ckpt_path} (epoch={ckpt.get('epoch','?')})")
    return model

def smiles_to_batch(smiles_list):
    """Convert SMILES strings to a heavy-atom PyTorch Geometric batch."""
    atom_types = []
    edge_index_src = []
    edge_index_dst = []
    bond_types = []
    batch_idx = []
    
    mols = []
    offset = 0
    
    for bi, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Failed to parse SMILES: {smi}")
            continue
            
        # Strip hydrogens since the model generates heavy atoms only
        mol = Chem.RemoveHs(mol)
        mols.append((bi, smi, mol))
        
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())
            batch_idx.append(bi)
            
        for bond in mol.GetBonds():
            s = bond.GetBeginAtomIdx() + offset
            d = bond.GetEndAtomIdx() + offset
            b_type = bond.GetBondTypeAsDouble()
            b_int = {1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}.get(b_type, 1)
            
            # Add bidirectional edges
            edge_index_src.extend([s, d])
            edge_index_dst.extend([d, s])
            bond_types.extend([b_int, b_int])
            
        offset += n_atoms
        
    batch = {
        "atom_types": torch.tensor(atom_types, dtype=torch.long),
        "edge_index": torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long),
        "bond_types": torch.tensor(bond_types, dtype=torch.long),
        "batch_idx": torch.tensor(batch_idx, dtype=torch.long),
    }
    return batch, mols

def build_3d_mol_with_hydrogens(mol_2d, coords):
    """Takes an RDKit mol (heavy atoms), assigns 3D coords, and adds H atoms."""
    # Build 3D conformer for heavy atoms
    conf = Chem.Conformer(mol_2d.GetNumAtoms())
    from rdkit.Geometry import Point3D
    for i, (x, y, z) in enumerate(coords.tolist()):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    
    mol_2d.AddConformer(conf, assignId=True)
    
    # Add hydrogens. addCoords=True geometrically places H atoms based on heavy atom positions.
    mol_with_hs = Chem.AddHs(mol_2d, addCoords=True)
    
    # Optional: Slightly relax the hydrogen atoms to fix any bad geometry 
    # without moving the heavy atoms (which the model generated).
    try:
        AllChem.MMFFOptimizeMolecule(mol_with_hs, maxIters=100)
    except Exception:
        pass
        
    return mol_with_hs

def main():
    parser = argparse.ArgumentParser(description="Generate 3D conformers from SMILES")
    parser.add_argument("--smiles", nargs="+", required=True, help="List of SMILES strings")
    parser.add_argument("--samples", type=int, default=1, help="Number of conformers to generate per SMILES")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/exp_G_heavy_atom_sota_ddp_best.pt", help="Path to model checkpoint")
    parser.add_argument("--outdir", type=str, default="generated_conformers", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM sampling steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.outdir, exist_ok=True)
    model = load_model(args.checkpoint, device)

    # Prepare input by duplicating SMILES for the requested number of samples
    expanded_smiles = []
    for smi in args.smiles:
        expanded_smiles.extend([smi] * args.samples)

    print(f"Parsing {len(args.smiles)} SMILES strings... (Generating {args.samples} conformer(s) each, {len(expanded_smiles)} total)")
    batch, mols = smiles_to_batch(expanded_smiles)
    
    if len(mols) == 0:
        print("No valid SMILES provided.")
        return

    # Move to device
    at = batch["atom_types"].to(device)
    ei = batch["edge_index"].to(device)
    bt = batch["bond_types"].to(device)
    bi = batch["batch_idx"].to(device)

    # Generate heavy atom coordinates
    print(f"Generating 3D heavy-atom coordinates (steps={args.steps})...")
    with torch.no_grad():
        coords_all = model.ddim_sample(at, ei, bt, bi, num_steps=args.steps).cpu()

    # Process and save
    print(f"Adding hydrogens and saving results to {args.outdir}/ ...")
    sdf_path = os.path.join(args.outdir, "conformers.sdf")
    sdf_writer = SDWriter(sdf_path)

    for i, (mol_idx, smi, mol_2d) in enumerate(mols):
        mask = batch["batch_idx"] == mol_idx
        coords = coords_all[mask]
        
        # Build full 3D molecule with hydrogens
        final_mol = build_3d_mol_with_hydrogens(mol_2d, coords)
        
        # Determine base molecule ID and sample ID
        base_mol_id = (i // args.samples) + 1
        sample_id = (i % args.samples) + 1
        
        # Write SDF
        mol_name = f"Mol_{base_mol_id}_Conf_{sample_id}"
        final_mol.SetProp("_Name", mol_name)
        final_mol.SetProp("SMILES", smi)
        sdf_writer.write(final_mol)
        
        # Write PDB
        pdb_path = os.path.join(args.outdir, f"mol_{base_mol_id}_conf_{sample_id}.pdb")
        Chem.MolToPDBFile(final_mol, pdb_path)
        print(f"  -> Saved {mol_name} ({smi}) to {pdb_path}")

    sdf_writer.close()
    print(f"All molecules saved to {sdf_path}")

if __name__ == "__main__":
    main()
