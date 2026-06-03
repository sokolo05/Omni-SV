import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from model_arch import OmniSVModel
from mgo_loss import MGOLoss
from utils import param_groups_l2_dual

class OmniSVDataset(Dataset):
    def __init__(self, record_ids, label_dict, feat_dirs):
        self.record_ids = record_ids
        self.label_dict = label_dict
        self.feat_dirs = feat_dirs

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        rid = self.record_ids[idx]
        try:
            x_syn = np.load(os.path.join(self.feat_dirs['syn'], f"{rid}.npy"))
            x_evo = np.load(os.path.join(self.feat_dirs['evo'], f"{rid}.npy"))
            x_omics = np.load(os.path.join(self.feat_dirs['omics'], f"{rid}.npy"))
            label = self.label_dict[rid]
            return (torch.from_numpy(x_syn).float(), 
                    torch.from_numpy(x_evo).float(), 
                    torch.from_numpy(x_omics).float(), 
                    torch.tensor(label).long())
        except Exception:
            return (torch.zeros(1024, 512), torch.zeros(1024, 2052), 
                    torch.zeros(1024, 14), torch.tensor(0).long())

def evaluate_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_syn, x_evo, x_omics, labels in loader:
            x_syn, x_evo, x_omics, labels = x_syn.to(device), x_evo.to(device), x_omics.to(device), labels.to(device)
            logits, _, _, _, _ = model(x_syn, x_evo, x_omics)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    return macro_f1, accuracy

def run_cross_validation(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Training initiated on: {device}")
    
    with open(args.label_json, 'r') as f:
        label_dict = json.load(f)
    
    record_ids = np.array(list(label_dict.keys()))
    feat_dirs = {'syn': args.syn_dir, 'evo': args.evo_dir, 'omics': args.omics_dir}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    final_test_f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(record_ids)):
        print(f"\n" + "="*20 + f" FOLD {fold+1} " + "="*20)
        
        val_size = len(train_idx) // 8
        val_idx = train_idx[:val_size]
        actual_train_idx = train_idx[val_size:]

        train_loader = DataLoader(OmniSVDataset(record_ids[actual_train_idx], label_dict, feat_dirs), 
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(OmniSVDataset(record_ids[val_idx], label_dict, feat_dirs), 
                                batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(OmniSVDataset(record_ids[test_idx], label_dict, feat_dirs), 
                                 batch_size=args.batch_size, shuffle=False)

        # 1. Corrected class instantiation
        model = OmniSVModel(d_model=args.d_model).to(device)
        criterion = MGOLoss(lambda1=args.lambda1, lambda2=args.lambda2, d_model=args.d_model).to(device)
        
        # 2. Optimized parameter group assembly incorporating learnable loss parameters
        param_groups = param_groups_l2_dual(
            model=model, 
            loss_fn=criterion, 
            backbone_wd=args.weight_decay, 
            head_wd=args.weight_decay * 10
        )
        optimizer = optim.AdamW(param_groups, lr=args.lr)
        
        best_val_f1 = 0.0
        best_model_path = os.path.join(args.save_dir, f"{args.model_name}_fold{fold+1}.pth")
        
        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for x_syn, x_evo, x_omics, labels in pbar:
                x_syn, x_evo, x_omics, labels = x_syn.to(device), x_evo.to(device), x_omics.to(device), labels.to(device)
                optimizer.zero_grad()
                
                logits, f_local, h_evo, f_final, h_omics = model(x_syn, x_evo, x_omics)
                loss, _, _, _ = criterion(logits, labels, f_local, h_evo, f_final, h_omics)
                
                loss.backward()
                optimizer.step()
            
            val_f1, val_acc = evaluate_metrics(model, val_loader, device)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"[*] New Best F1: {val_f1:.4f} - Model saved to {best_model_path}")

        model.load_state_dict(torch.load(best_model_path))
        test_f1, test_acc = evaluate_metrics(model, test_loader, device)
        print(f"[*] Fold {fold+1} Result: Test F1={test_f1:.4f}, Accuracy={test_acc:.4f}")
        final_test_f1_scores.append(test_f1)

    print("\n" + "="*20 + " CROSS VALIDATION COMPLETE " + "="*20)
    print(f"Average Macro-F1: {np.mean(final_test_f1_scores):.4f} (+/- {np.std(final_test_f1_scores):.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omni-SV 5-Fold Cross Validation Pipeline")
    
    parser.add_argument("--label_json", type=str, required=True)
    parser.add_argument("--syn_dir", type=str, required=True)
    parser.add_argument("--evo_dir", type=str, required=True)
    parser.add_argument("--omics_dir", type=str, required=True)
    
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_name", type=str, default="OmniSV_Best")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.1)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    run_cross_validation(args)
