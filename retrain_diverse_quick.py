"""Quick retrain from cached diverse features - no extraction needed."""
import logging, sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_detection_app.models.detector import VoiceAuthenticityNet

cache = Path("data/features_diverse_v1.npz")
data = np.load(str(cache))
X, y = data["X"], data["y"]
logger.info(f"Loaded {len(X)} samples G:{int((y==0).sum())} S:{int((y==1).sum())}")

# Balance 1:1
genuine_idx = np.where(y==0)[0]; synthetic_idx = np.where(y==1)[0]
np.random.seed(42)
min_c = min(len(genuine_idx), len(synthetic_idx))
gb = np.random.choice(genuine_idx, size=min_c, replace=False)
sb = np.random.choice(synthetic_idx, size=min_c, replace=False)
idx = np.concatenate([gb, sb]); np.random.shuffle(idx)
Xb, yb = X[idx], y[idx]
logger.info(f"Balanced {len(Xb)}")

X_train, X_val, y_train, y_val = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VoiceAuthenticityNet(input_size=64, hidden_sizes=[512,256,128,64], dropout=0.3).to(device)
logger.info(f"Params {sum(p.numel() for p in model.parameters()):,}")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
best=0; patience=0; save_path=Path("voice_detection_app/trained_model.pth")
for epoch in range(300):
    model.train()
    t_loss=0; correct=0; total=0
    for bx,by in train_loader:
        bx,by=bx.to(device),by.to(device)
        optimizer.zero_grad()
        out=model(bx).squeeze(-1)
        loss=criterion(out,by); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        t_loss+=loss.item()
        correct+=((torch.sigmoid(out)>0.5).float()==by).sum().item(); total+=by.size(0)
    scheduler.step()
    model.eval()
    v_correct=0; v_total=0; all_probs=[]; all_labels=[]
    with torch.no_grad():
        for bx,by in val_loader:
            bx,by=bx.to(device),by.to(device)
            out=model(bx).squeeze(-1); probs=torch.sigmoid(out)
            v_correct+=((probs>0.5).float()==by).sum().item(); v_total+=by.size(0)
            all_probs.extend(probs.cpu().numpy()); all_labels.extend(by.cpu().numpy())
    import numpy as np
    all_probs=np.array(all_probs); all_labels=np.array(all_labels)
    try: auc=roc_auc_score(all_labels, all_probs)
    except: auc=0
    if (epoch+1)%10==0 or epoch==0:
        logger.info(f"Epoch {epoch+1:3d} | T L:{t_loss/len(train_loader):.4f} A:{correct/total:.4f} | V A:{v_correct/v_total:.4f} AUC:{auc:.4f}")
    combined=(v_correct/v_total)*0.5+auc*0.5
    if combined>best:
        best=combined; patience=0; torch.save(model.state_dict(), str(save_path))
    else:
        patience+=1
        if patience>=40:
            logger.info(f"Early stop {epoch+1}"); break
logger.info(f"Best {best:.4f}")
# final eval
model.load_state_dict(torch.load(str(save_path), map_location=device, weights_only=True))
model.eval()
all_probs=[]; all_labels=[]
with torch.no_grad():
    for bx,by in val_loader:
        bx,by=bx.to(device),by.to(device)
        out=model(bx).squeeze(-1); probs=torch.sigmoid(out)
        all_probs.extend(probs.cpu().numpy()); all_labels.extend(by.cpu().numpy())
import numpy as np
all_probs=np.array(all_probs); all_labels=np.array(all_labels)
binary=(all_probs>0.5).astype(int)
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
try: logger.info(f"AUC {roc_auc_score(all_labels, all_probs):.4f}")
except: pass
print(classification_report(all_labels, binary, target_names=["Genuine","Synthetic"]))
print(confusion_matrix(all_labels, binary))
logger.info(f"G mean {all_probs[all_labels==0].mean():.4f} S mean {all_probs[all_labels==1].mean():.4f}")
# ONNX
import onnx
model_cpu=VoiceAuthenticityNet(input_size=64, hidden_sizes=[512,256,128,64], dropout=0.0)
model_cpu.load_state_dict(model.state_dict()); model_cpu.eval()
torch.onnx.export(model_cpu, torch.randn(1,64), "voice_detection_app/trained_model.onnx", opset_version=14, input_names=["features"], output_names=["synthetic_prob"], dynamic_axes={"features":{0:"batch"},"synthetic_prob":{0:"batch"}}, dynamo=False)
logger.info("ONNX done")
