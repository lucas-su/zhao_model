import os, sys
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import gc

import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.vgg import VGG, make_layers

from pprint import pprint, pformat
from torch.utils.data import DataLoader
import numpy as np

# from torchinfo import summary

import customLayers
from datasetbuilders import iitaff, sunrgbd, umd
from arg_interpret import arg_int
from stat_functions import conf_scores_weighted

def print_memstats(location = None):
    for obj in gc.get_objects():
        torch.cuda.memory_summary()
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if location:
                    print(f"Memstats at {location}")
                print(type(obj), obj.size())
        except:
            pass

class MyResNet(ResNet):
    def __init__(self):

        if dcnn == "resnet50":
            super(MyResNet, self).__init__(Bottleneck, [3, 4, 6, 3]) # resnet50
        elif dcnn == "resnet18":
            super(MyResNet, self).__init__(BasicBlock, [2, 2, 2, 2]) # resnet18
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        """
        disabling layers 3 and 4 yields input into coordaspp which is closer to zhao 
        but this also yields memory issues with umd (specifically feature size 17) in relation module 
        where an FC layer of shape WxHx2xfeature_size is used. Only disabling layer 4 is a compromise, still close
        to the shape described in the paper, and with fewer memory issues
        """
        x = self.layer3(x)
        # x = self.layer4(x)
        return x

class MyVGG(VGG):
    def __init__(self):
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        super(MyVGG, self).__init__(make_layers(self.cfg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for y in self.features:
        #     x = y(x)
        x = self.features[:23](x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dcnn, nchannels):
        super().__init__()

        # encoder
        if dcnn == "resnet50":
            self.model_conv = MyResNet()
            self.model_conv.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict())
        elif dcnn == "VGG":
            self.model_conv = MyVGG()
            self.model_conv.load_state_dict(torchvision.models.vgg16(pretrained=True).state_dict())
        else:
            self.model_conv = MyResNet()
            self.model_conv.load_state_dict(torchvision.models.resnet18(pretrained=True).state_dict())

        if not train_dcnn:
            for param in self.model_conv.parameters():
                param.requires_grad = False

        if dcnn == 'resnet50':
            self.coordASPP = customLayers.CoordASPP.CoordASPP(self.model_conv.layer3[-1].bn3.num_features, 256,
                                                              [1, 2, 3, 6])  # 512 for resnet18, 2048 for resnet50 todo changed
        elif dcnn == 'resnet18':
            self.coordASPP = customLayers.CoordASPP.CoordASPP(self.model_conv.layer4[-1].bn2.num_features, 256,
                                                              [1, 2, 3, 6])  # 512 for resnet18, 2048 for resnet50
        elif dcnn == "VGG":
            self.coordASPP = customLayers.CoordASPP.CoordASPP(512, 256, [1, 2, 3, 6])
        else:
            raise NotImplementedError  # for other models (squeezenet, alexnet etc.)
        self.upsample_encoder = nn.Upsample(scale_factor=4)

        self.rescale_conv = nn.Conv2d(256, nchannels, 3, padding="same")
        self.batch_norm_enc = nn.BatchNorm2d(nchannels)
        self.dropout_enc = nn.Dropout()
        self.relu_enc = torch.nn.ReLU()

    def forward(self, x):
        x = self.model_conv(x)
        x = self.coordASPP(x)

        x = self.upsample_encoder(x)
        x = self.rescale_conv(x)
        if norm:
            x = self.batch_norm_enc(x)
        if activation:
            x = self.relu_enc(x)
        if dropout:
            x = self.dropout_enc(x)
        # print_memstats('encoder forward')
        return x

class Decoder(nn.Module):
    def __init__(self, dataset, nchannels, noutputs):
        super().__init__()
        # decoder after relation and elm
        self.oselm = customLayers.oselm.OSELM(dataset=dataset, norm=norm, dropout=dropout, activation=activation )
        self.relation = customLayers.relationshipModule.RelationshipAwareModule(dataset=dataset, norm=norm, dropout=dropout, activation=activation)

        self.conv4 = nn.Conv2d(nchannels, noutputs, 4) # to n channels for mask output, change here if feature map size is larger than nobjects. Change kernel size to 3 for layer 3 and 4 in dcnn
        self.batch_norm_dec = nn.BatchNorm2d(noutputs)
        self.dropout_dec = nn.Dropout()
        self.upsample_decoder = nn.Upsample(scale_factor=4) # revert to same resolution as mask

        self.nchannels = nchannels

    def forward(self, x):
        oselm = self.oselm(x)
        omega_oselm = torch.mul(oselm, torch.ones(x.shape[1], device=oselm.device) * 0.1) # use mul here becuase lambda_ is scalar
        r_a__objectLabels = self.relation(x)
        Wfusion = torch.add(omega_oselm, r_a__objectLabels).add(torch.ones(self.nchannels).to(omega_oselm.device))
        Wfusion = Wfusion.unsqueeze(2).unsqueeze(3)
        x = torch.multiply(x,Wfusion)
        x = self.conv4(x)
        if norm:
            x = self.batch_norm_dec(x)
        if dropout:
            x = self.dropout_dec(x)
        if activation:
            x = torch.relu(x)
        x = self.upsample_decoder(x)
        # print_memstats('decoder forward')
        return x, r_a__objectLabels, oselm

class ZhaoModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        if dataset == 'umd':
            self.nchannels = 17
            self.noutputs = 8
        elif dataset == 'iitaff':
            self.nchannels = 10
            self.noutputs = 10
        self.encoder = Encoder(dcnn, self.nchannels)
        self.decoder = Decoder(dataset, self.nchannels, self.noutputs)

        self.loss_seg = smp.losses.FocalLoss(smp.losses.MULTILABEL_MODE, gamma=1)
        self.loss_r_aware = smp.losses.FocalLoss(smp.losses.MULTILABEL_MODE, gamma=1)

        self.alpha = 10

    def forward(self, image):
        image = image.float()
        x = self.encoder(image)

        mask, r_a__objectLabels, oselm = self.decoder(x)
        mask = torch.softmax(mask, dim=1)
        return mask, r_a__objectLabels, oselm

    def shared_step(self, batch, stage):
        # print(torch.cuda.memory_allocated())
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4
        h, w = image.shape[2:]
        # assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"].float()
        # mask = mask[:, 0:14].contiguous() # for use with fewer channels than objects
        objectLabel = batch["object"]
        # objectLabel = objectLabel[:,:14].contiguous()

        logits_mask, r_a__objectLabels, oselm = self.forward(image)

        loss_seg = self.loss_seg(logits_mask, mask)
        loss_r = self.loss_r_aware(r_a__objectLabels, objectLabel)
        l2_lambda = 0.00000001
        # l2_reg = torch.tensor(0., device="cuda")
        l2_reg = 0
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.linalg.norm(param)**2
        loss_reg = (l2_lambda * l2_reg)
        # loss_reg = l2_reg
        loss = self.alpha * loss_seg + loss_r + loss_reg # reg implemented in adam function instead of manually

        masksize = logits_mask.size()
        weighted_conf_metrics = np.zeros((masksize[0], masksize[1],5))
        for b in range(masksize[0]):
            for i in range(masksize[1]):
                weighted_conf_metrics[b][i] = conf_scores_weighted(logits_mask[b][i], mask[b][i])
        logits_mask = logits_mask.sigmoid()
        # pred_mask = prob_mask
        pred_mask = (logits_mask > 0.5).float()


        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multilabel")

        return {
            "loss": loss,
            "loss_seg": loss_seg.detach(),
            "loss_r": loss_r.detach(),
            # "loss_reg": loss_reg.detach(),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "tpw": weighted_conf_metrics.sum(0).T[0],
            "fpw": weighted_conf_metrics.sum(0).T[1],
            "tnw": weighted_conf_metrics.sum(0).T[2],
            "fnw": weighted_conf_metrics.sum(0).T[3],
            "q": np.nanmean(weighted_conf_metrics, axis=0).T[4]
        }

    def shared_epoch_end(self, outputs, stage):

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs]).long()
        fp = torch.cat([x["fp"] for x in outputs]).long()
        fn = torch.cat([x["fn"] for x in outputs]).long()
        tn = torch.cat([x["tn"] for x in outputs]).long()

        tpw = np.array([x["tpw"] for x in outputs]).sum(0)
        fpw = np.array([x["fpw"] for x in outputs]).sum(0)
        fnw = np.array([x["fnw"] for x in outputs]).sum(0)
        tnw = np.array([x["tnw"] for x in outputs]).sum(0)
        q_macro = np.nanmean(np.array([x["q"] for x in outputs]), axis=0)

        loss_total = ([x["loss"].item() for x in outputs])
        loss_seg = ([x["loss_seg"].item() for x in outputs])
        loss_r = ([x["loss_r"].item() for x in outputs])
        # loss_reg = ([x["loss_reg"].item() for x in outputs])

        # aggregate intersection and union over whole dataset and then compute IoU score.
        # images without some target influence per_image_iou more than dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        none_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)

        # average over all batches
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)

        per_label_iou_none = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        # fbeta_mean = np.mean(list(per_label_fbeta_none.cpu()))

        metrics = {
            f"{stage}_loss_total": np.mean(loss_total),
            f"{stage}_loss_seg": np.mean(loss_seg),
            f"{stage}_loss_r": np.mean(loss_r),
            # f"{stage}_loss_reg": np.mean(loss_reg),
            f"{stage}_iou_dataset": dataset_iou,
            f"{stage}_iou_none": none_iou,
        }
        fbeta_w = []
        for i, iou in enumerate(per_label_iou_none):
            metrics[f"{stage}_iou_none_label_{i}"] = iou
        for i, (tpw_i, fpw_i, fnw_i, tnw_i) in enumerate(zip(tpw, fpw, fnw, tnw)):
            beta = 1
            eps = np.spacing(1)
            R = tpw_i/(fnw_i + tpw_i + eps)
            P = tpw_i/(fpw_i + tpw_i + eps)
            Q = (1 + beta ** 2) * (R * P) / (eps + R + (beta * P))
            fbeta_w.append(Q)
            metrics[f"{stage}_fbeta_weighted_label_{i}"] = Q
            # include macro averaged q per image
            metrics[f"{stage}_fbeta_macro_weighted_label_{i}"] = q_macro[i]

        metrics[f"{stage}_fbeta_weighted_mean"] = np.mean(fbeta_w)

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1*10**-8)
        return torch.optim.Adam(self.parameters(), lr=0.0001)


if __name__ == "__main__":

    args = dict(arg.split("=") for arg in sys.argv[1:])

    print(f"device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    dataset, dcnn, train_test_mode, train_dcnn, use_gpu, norm, dropout, activation = arg_int(args)

    if dataset == 'sunrgbd':
        if os.path.exists("devmode"):
            root = "/media/luc/data/sunrgbd"
        else:
            root = "/home/schootuiterkampl/sunrgbd"

        train_dataset = sunrgbd.sunrgbd(root, "train")
        valid_dataset = sunrgbd.sunrgbd(root, "valid")
        test_dataset = sunrgbd.sunrgbd(root, "test")

    elif dataset == 'iitaff':
        if os.path.exists("devmode"):
            root = "/media/luc/data/iitaff"
        else:
            root = "/home/schootuiterkampl/iitaff"

        train_dataset = iitaff.iitaff(root, "train")
        valid_dataset = iitaff.iitaff(root, "valid")
        test_dataset = iitaff.iitaff(root, "test")
    elif dataset == 'umd':
        if os.path.exists("devmode"):
            root = "/media/luc/data/UMD/part-affordance-dataset"
        else:
            root = "/home/schootuiterkampl/part-affordance-dataset"

        train_dataset = umd.umd(root, "train")
        valid_dataset = umd.umd(root, "valid")
        test_dataset = umd.umd(root, "test")

    else:
        raise ValueError

    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()

    if os.path.exists("devmode"):
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
        valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=n_cpu)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=n_cpu)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16)
        valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=n_cpu)

    model = ZhaoModel()
    # summary(model)
    if os.path.exists("devmode"):
        trainer = pl.Trainer(
            gpus=(torch.cuda.device_count() if use_gpu else 0),
            max_epochs=15,
        )
    else:
        trainer = pl.Trainer(
            gpus=(torch.cuda.device_count() if use_gpu else 0),
            max_epochs=15,
        )

    if train_test_mode == 'train':
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
    else:
        torch.load("/home/luc/Documents/iitaff_resnet50_model_state_dict")

    model.eval()
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=True)
    pprint(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=True)
    pprint(test_metrics)

    name_template = f'{dataset}_{dcnn}-{"trained" if train_dcnn else "frozen"}_' \
                    f'{"GPU" if use_gpu else "CPU"}_'\
                    f'{"BNorm" if norm else "no-BNorm"}_'\
                    f'{"Dropout" if dropout else "no-Dropout"}_'\
                    f'{"Activation" if activation else "no-Activation"}'

    torch.save(model.state_dict(), f'{name_template}_model_state_dict')


    with open(f'{name_template}_metrics.txt', 'w') as file:
        file.write(pformat(valid_metrics))
        file.write(pformat(test_metrics))

