import torch
import matplotlib.pyplot as plt
from MemSE import ROOT
from MemSE.nas import AccuracyDataset, ResNetArchEncoder, AccuracyPredictor


torch.backends.cudnn.benchmark = True

save_path = ROOT / 'experiments/conference_2/results'
dset = AccuracyDataset(save_path)
#dset.merge_acc_dataset()
encoder = ResNetArchEncoder()
tloader, vloader, bacc, bpow = dset.build_acc_data_loader(encoder)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
net = AccuracyPredictor(encoder).to(device)
loaded = torch.load(save_path / 'trained_predictor.pth')
net.load_state_dict(loaded['net_dict'])
net.eval()

plt.plot(loaded['train_loss'], label='Train loss')
plt.plot(loaded['val_loss'], label='Val loss')
plt.title('Loss evolution during training')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.savefig(f'{save_path}/predictor_train_loss.png')
plt.close()

pow_abs = []
pow_data = []
prec_abs = []
prec_data = []
with torch.no_grad():
    losses = []
    for inp, labels in vloader:
        inp, labels = inp.to(device), labels.to(device)
        out = net(inp)
        pow_abs.extend(labels[:, 1].cpu().tolist())
        pow_data.extend(out[:, 1].cpu().tolist())
        prec_abs.extend(labels[:, 0].cpu().tolist())
        prec_data.extend(out[:, 0].cpu().tolist())


plt.scatter(pow_abs, pow_data)
plt.title('Validation power prediction vs simulated')
plt.ylabel('Predicted power')
plt.xlabel('Simulated power')
plt.savefig(f'{save_path}/predictor_power.png')
plt.close()
plt.scatter(prec_abs, prec_data)
plt.title('Validation precision prediction vs simulated')
plt.ylabel('Predicted precision')
plt.xlabel('Simulated precision')
plt.savefig(f'{save_path}/predictor_prec.png')
plt.close()
