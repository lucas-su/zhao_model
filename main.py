import os, sys
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn

import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from pprint import pprint
from torch.utils.data import DataLoader
import numpy as np

from torchinfo import summary

import customLayers
from datasetbuilders import iitaff, sunrgbd

from stat_functions import conf_scores_weighted

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
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class Encoder(nn.Module):
    def __init__(self, dcnn):
        super().__init__()
        # encoder
        self.model_conv = MyResNet()
        if dcnn == "resnet50":
            self.model_conv.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict())
        else:
            self.model_conv.load_state_dict(torchvision.models.resnet18(pretrained=True).state_dict())

        # self.model_conv = torchvision.models.alexnet(pretrained=True)
        # self.model_conv = torchvision.models.squeezenet1_0(pretrained=True)
        # self.model_conv = torchvision.models.vgg16(pretrained=True)

        if dcnn[:6] == 'resnet':
            self.coordASPP = customLayers.CoordASPP.CoordASPP(self.model_conv.layer4[2].bn3.num_features, 256,
                                                              [1, 2, 3, 6])  # 512 for resnet18, 2048 for resnet50
        else:
            raise NotImplementedError  # for other models (squeezenet, alexnet etc.)
        self.upsample_encoder = nn.Upsample(scale_factor=4)

        self.rescale_conv = nn.Conv2d(256, 10, 3, padding="same")
        self.batch_norm_enc = nn.BatchNorm2d(10)
        self.dropout_enc = nn.Dropout()
        self.relu_enc = torch.nn.ReLU()

    def forward(self, x):
        cnnblock = self.model_conv(x)
        coordaspp = self.coordASPP(cnnblock)

        upsampled = self.upsample_encoder(coordaspp)
        encoder = self.rescale_conv(upsampled)
        encoder = self.batch_norm_enc(encoder)
        encoder = self.relu_enc(encoder)
        encoder = self.dropout_enc(encoder)
        return encoder

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # decoder after relation and elm
        self.oselm = customLayers.oselm.OSELM()
        self.relation = customLayers.relationshipModule.RelationshipAwareModule()

        self.conv4 = nn.Conv2d(10, 10, 3, padding='same') # to 10 channel for mask output
        self.batch_norm_dec = nn.BatchNorm2d(10)
        self.dropout_dec = nn.Dropout()
        self.upsample_decoder = nn.Upsample(scale_factor=8) # revert to some resolution to see output images


    def forward(self, x):
        oselm = self.oselm(x)
        relation = self.relation(x)
        y = torch.add(oselm, relation)
        y = y.unsqueeze(2).unsqueeze(3)
        x = torch.multiply(x,y)
        x = self.conv4(x)
        x = self.batch_norm_dec(x)
        # x = self.dropout_dec(x)
        x = torch.relu(x)
        x = self.upsample_decoder(x)
        return x

class ZhaoModel(pl.LightningModule):

    def __init__(self, in_channels, out_classes, **kwargs):
        super().__init__()
        self.encoder = Encoder(dcnn=dcnn)
        self.decoder = Decoder()

        # dice loss taken from segmentation_models_pytorch
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) # binary because one hot is implemented in dataset already
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.loss_fn = smp.losses.FocalLoss(smp.losses.MULTILABEL_MODE, gamma=1) # focal loss with gamma=1 is cross entropy loss which is what was used in paper


    def forward(self, image):
        image = image.float()
        x = self.encoder(image)

        mask = self.decoder(x)
        mask = torch.softmax(mask, dim=1)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        masksize = logits_mask.size()
        weighted_conf_metrics = np.zeros((masksize[0], masksize[1],4))
        for b in range(masksize[0]):
            for i in range(masksize[1]):
                weighted_conf_metrics[b][i] = conf_scores_weighted(logits_mask[b][i], mask[b][i])
        # prob_mask = logits_mask.sigmoid()
        pred_mask = (logits_mask > 0.5).float()

        # tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multilabel")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "tpw": weighted_conf_metrics.sum(0).T[0],
            "fpw": weighted_conf_metrics.sum(0).T[1],
            "tnw": weighted_conf_metrics.sum(0).T[2],
            "fnw": weighted_conf_metrics.sum(0).T[3],
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

        loss = ([x["loss"].item() for x in outputs])

        # aggregate intersection and union over whole dataset and then compute IoU score.
        # images without some target influence per_image_iou more than dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        none_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)

        # average over all batches
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)

        fbeta_w = []
        for tpw_i, fpw_i, fnw_i, tnw_i in zip(tpw, fpw, fnw, tnw):
            beta = 1
            R = tpw_i/(fnw_i + tpw_i)
            P = tpw_i/(fpw_i + tpw_i)
            eps = np.spacing(1)
            Q = (1 + beta ** 2) * (R * P) / (eps + R + (beta * P))
            fbeta_w.append(Q)

        # beta = 1
        # beta_tp = (1 + beta ** 2) * tp
        # beta_fn = (beta ** 2) * fn
        # score = beta_tp / (beta_tp + beta_fn + fp)
        # class_weights = torch.tensor(class_weights).to(tp.device)
        # per_label_fbeta_weighted = (score * class_weights)
        #
        # per_label_fbeta_none = smp.metrics.fbeta_score(tp, fp, fn, tn, reduction=None)
        per_label_iou_none = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        # fbeta_mean = np.mean(list(per_label_fbeta_none.cpu()))
        metrics = {
            f"{stage}_loss": np.mean(loss),
            f"{stage}_iou_dataset": dataset_iou,
            f"{stage}_iou_none": none_iou,
            f"{stage}_fbeta_none": np.mean(fbeta_w),
            f"{stage}_fbeta_none_label_0": fbeta_w[0],
            f"{stage}_fbeta_none_label_1": fbeta_w[1],
            f"{stage}_fbeta_none_label_2": fbeta_w[2],
            f"{stage}_fbeta_none_label_3": fbeta_w[3],
            f"{stage}_fbeta_none_label_4": fbeta_w[4],
            f"{stage}_fbeta_none_label_5": fbeta_w[5],
            f"{stage}_fbeta_none_label_6": fbeta_w[6],
            f"{stage}_fbeta_none_label_7": fbeta_w[7],
            f"{stage}_fbeta_none_label_8": fbeta_w[8],
            f"{stage}_fbeta_none_label_9": fbeta_w[9],
            f"{stage}_iou_none_label_0": per_label_iou_none[0],
            f"{stage}_iou_none_label_1": per_label_iou_none[1],
            f"{stage}_iou_none_label_2": per_label_iou_none[2],
            f"{stage}_iou_none_label_3": per_label_iou_none[3],
            f"{stage}_iou_none_label_4": per_label_iou_none[4],
            f"{stage}_iou_none_label_5": per_label_iou_none[5],
            f"{stage}_iou_none_label_6": per_label_iou_none[6],
            f"{stage}_iou_none_label_7": per_label_iou_none[7],
            f"{stage}_iou_none_label_8": per_label_iou_none[8],
            f"{stage}_iou_none_label_9": per_label_iou_none[9]
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
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
        return torch.optim.Adam(self.parameters(), lr=0.0001)


if __name__ == "__main__":

    if sys.argv.__len__() == 1:
        dataset = 'sunrgbd'
    else:
        dataset = sys.argv[1] # options 'sunrgbd' 'iitaff'

    if sys.argv.__len__() < 3:
        dcnn = "resnet50"
    else:
        dcnn = sys.argv[2] # options 'resnet50' 'resnet18'

    if sys.argv.__len__() < 4:
        test_test_mode = 'train'
    else:
        test_test_mode = sys.argv[3] # options 'train' 'test'

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
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
        valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    model = ZhaoModel(in_channels=3, out_classes=10)
    summary(model)
    if os.path.exists("devmode"):
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=15,
        )
    else:
        trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=15,
        )

    if test_test_mode == 'train':
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

    torch.save(model.state_dict(), f'{dataset}_{dcnn}_model_state_dict')