import torch
import torch.nn as nn
from argparse import ArgumentParser
from MemSE import ROOT
from MemSE.nas import AccuracyDataset, ResNetArchEncoder, AccuracyPredictorFactory

from tqdm import tqdm

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument('--predictor', default='AccuracyPredictor')
args, _ = parser.parse_known_args()

save_path = ROOT / 'experiments/conference_2/results'
dset = AccuracyDataset(save_path)
#dset.merge_acc_dataset()
encoder = ResNetArchEncoder()
tloader, vloader, bacc, bpow = dset.build_acc_data_loader(encoder)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
net = AccuracyPredictorFactory[args.predictor](encoder).to(device)

opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

epoch_train_loss, epoch_val_loss = [], []

with tqdm(total=200) as t:
    for _ in range(200):
        losses = []
        for inp, labels in tloader:
            inp, labels = inp.to(device), labels.to(device)
            out = net(inp)
            opt.zero_grad()
            loss = torch.mean((labels - out)**2)
            losses.append(loss.item())
            loss.backward()
            opt.step()
        # scheduler.step()
        epoch_train_loss.append(torch.mean(torch.tensor(losses)).item())
        with torch.no_grad():
            losses = []
            for inp, labels in vloader:
                inp, labels = inp.to(device), labels.to(device)
                out = net(inp)
                loss = torch.mean((labels - out)**2)
                losses.append(loss.item())
            scheduler.step(torch.mean(torch.tensor(losses)).item())
            epoch_val_loss.append(torch.mean(torch.tensor(losses)).item())
            t.set_postfix({"val_loss": torch.mean(torch.tensor(losses)).item(),})
        t.update()

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

torch.save({'net_dict': net.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'pow_ref': torch.tensor(pow_abs),
            'pow_pred': torch.tensor(pow_data),
            'prec_ref': torch.tensor(prec_abs),
            'prec_pred': torch.tensor(prec_data)},
           save_path / f'trained_{args.predictor}.pth')