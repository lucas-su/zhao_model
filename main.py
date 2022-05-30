import os, sys
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn

import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.vgg import VGG, make_layers

from pprint import pprint
from torch.utils.data import DataLoader
import numpy as np

from torchinfo import summary

import customLayers
from datasetbuilders import iitaff, sunrgbd, umd

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
        # x = self.layer3(x) # layers disabled because table indicates the dcnn stops at 512 channels
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
    def __init__(self, dcnn):
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

        # self.model_conv = torchvision.models.alexnet(pretrained=True)
        # self.model_conv = torchvision.models.squeezenet1_0(pretrained=True)
        # self.model_conv = torchvision.models.vgg16(pretrained=True)

        if dcnn == 'resnet50':
            self.coordASPP = customLayers.CoordASPP.CoordASPP(self.model_conv.layer2[-1].bn3.num_features, 256,
                                                              [1, 2, 3, 6])  # 512 for resnet18, 2048 for resnet50
        elif dcnn == 'resnet18':
            self.coordASPP = customLayers.CoordASPP.CoordASPP(self.model_conv.layer4[-1].bn2.num_features, 256,
                                                              [1, 2, 3, 6])  # 512 for resnet18, 2048 for resnet50
        elif dcnn == "VGG":
            self.coordASPP = customLayers.CoordASPP.CoordASPP(512, 256, [1, 2, 3, 6])
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
    def __init__(self, dataset):
        super().__init__()
        # decoder after relation and elm
        self.oselm = customLayers.oselm.OSELM(dataset=dataset)
        self.relation = customLayers.relationshipModule.RelationshipAwareModule(dataset=dataset)

        self.conv4 = nn.Conv2d(10, 10, 3) # to 10 channel for mask output
        self.batch_norm_dec = nn.BatchNorm2d(10)
        self.dropout_dec = nn.Dropout()
        self.upsample_decoder = nn.Upsample(scale_factor=2) # revert to some resolution to see output images
        if dataset == 'umd':
            out_features = 7
        elif dataset == 'iitaff':
            out_features = 10
        else:
            raise ValueError
        self.lambda_ = torch.ones(out_features, device='cuda') * 0.1

    def forward(self, x):
        oselm = self.oselm(x)
        omega_oselm = torch.mul(oselm, self.lambda_) # use mul here becuase lambda_ is scalar
        r_a__objectLabels = self.relation(x)
        Wfusion = torch.add(omega_oselm, r_a__objectLabels).add(torch.ones(10).to("cuda"))
        Wfusion = Wfusion.unsqueeze(2).unsqueeze(3)
        x = torch.multiply(x,Wfusion)
        x = self.conv4(x)
        x = self.batch_norm_dec(x)
        x = self.dropout_dec(x)
        x = torch.relu(x)
        x = self.upsample_decoder(x)
        return x, r_a__objectLabels, oselm


class ZhaoModel(pl.LightningModule):

    def __init__(self, in_channels, out_classes, **kwargs):
        super().__init__()
        self.encoder = Encoder(dcnn=dcnn)
        self.decoder = Decoder(dataset=dataset)

        self.loss_seg = smp.losses.FocalLoss(smp.losses.MULTILABEL_MODE, gamma=1)
        self.loss_r_aware = smp.losses.FocalLoss(smp.losses.MULTILABEL_MODE, gamma=1)
        self.gamma_R_theta = torch.nn.MSELoss()
        self.alpha = 10


    def forward(self, image):
        image = image.float()
        x = self.encoder(image)

        mask, r_a__objectLabels, oselm = self.decoder(x)
        mask = torch.softmax(mask, dim=1)
        return mask, r_a__objectLabels, oselm

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4
        h, w = image.shape[2:]
        # assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"]
        objectLabel = batch["object"]

        regularized_mask = torch.flatten(mask,start_dim=2).any(dim=2).float() # number of times each affordance is present


        logits_mask, r_a__objectLabels, oselm = self.forward(image)

        loss_seg = self.loss_seg(logits_mask, mask)
        loss_r = self.loss_r_aware(r_a__objectLabels, objectLabel)
        loss_reg = self.gamma_R_theta(logits_mask, mask.float())
        loss = self.alpha * loss_seg + loss_r + loss_reg

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
            "loss_seg": loss_seg,
            "loss_r": loss_r,
            "loss_reg": loss_reg,
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

        loss_total = ([x["loss"].item() for x in outputs])
        loss_seg = ([x["loss_seg"].item() for x in outputs])
        loss_r = ([x["loss_r"].item() for x in outputs])
        loss_reg = ([x["loss_reg"].item() for x in outputs])

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

        per_label_iou_none = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        # fbeta_mean = np.mean(list(per_label_fbeta_none.cpu()))
        metrics = {
            f"{stage}_loss_total": np.mean(loss_total),
            f"{stage}_loss_seg": np.mean(loss_seg),
            f"{stage}_loss_r": np.mean(loss_r),
            f"{stage}_loss_reg": np.mean(loss_reg),
            f"{stage}_iou_dataset": dataset_iou,
            f"{stage}_iou_none": none_iou,
            f"{stage}_fbeta_weighted_mean": np.mean(fbeta_w),
            f"{stage}_fbeta_weighted_label_0": fbeta_w[0],
            f"{stage}_fbeta_weighted_label_1": fbeta_w[1],
            f"{stage}_fbeta_weighted_label_2": fbeta_w[2],
            f"{stage}_fbeta_weighted_label_3": fbeta_w[3],
            f"{stage}_fbeta_weighted_label_4": fbeta_w[4],
            f"{stage}_fbeta_weighted_label_5": fbeta_w[5],
            f"{stage}_fbeta_weighted_label_6": fbeta_w[6],
            f"{stage}_fbeta_weighted_label_7": fbeta_w[7],
            f"{stage}_fbeta_weighted_label_8": fbeta_w[8],
            f"{stage}_fbeta_weighted_label_9": fbeta_w[9],
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

    args = dict(arg.split("=") for arg in sys.argv[1:])

    if "dataset" in args.keys():
        dataset = args["dataset"] # options 'sunrgbd' 'iitaff' 'umd'
    else:
        print("using default dataset sunrgbd")
        dataset = 'sunrgbd'

    if "dcnn_type" in args.keys():
        dcnn = args["dcnn_type"] # options 'resnet50' 'resnet18' 'vgg'
    else:
        print("using default dcnn type resnet50")
        dcnn = "resnet50"

    if "train_test_mode" in args.keys():
        test_test_mode = args["train_test_mode"] # options 'train' 'test'
    else:
        print("training network by default")
        test_test_mode = 'train'

    if "train_dcnn" in args.keys():
        train_dcnn = True if args["train_dcnn"].lower() == 'true' else False # options true, false
    else:
        print("training dcnn layers by default")
        train_dcnn = True

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
            root = "/media/luc/data/UMD"
        else:
            root = "/home/schootuiterkampl/UMD"

        train_dataset = umd.umd(root, "train")
        valid_dataset = umd.umd(root, "test") ##################### valid == test because umd does not provide valid set
        test_dataset = umd.umd(root, "test")

    else:
        raise ValueError

    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    # assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames)) #disable for umd
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

    torch.save(model.state_dict(), f'{dataset}_{dcnn}_{"dcnn-trained" if train_dcnn else "dcnn-untrained"}_model_state_dict')