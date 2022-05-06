import os, sys
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from pprint import pprint
from torch.utils.data import DataLoader
import numpy as np

from torchinfo import summary

import custom_layers
import sunrgbd, iitaff

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

class ZhaoModel(pl.LightningModule):

    def __init__(self, in_channels, out_classes, **kwargs):
        super().__init__()

        #encoder
        self.model_conv = MyResNet()
        if dcnn == "resnet50":
            self.model_conv.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict())
        else:
            self.model_conv.load_state_dict(torchvision.models.resnet18(pretrained=True).state_dict())

        # self.model_conv = torchvision.models.alexnet(pretrained=True)
        # self.model_conv = torchvision.models.squeezenet1_0(pretrained=True)
        # self.model_conv = torchvision.models.vgg16(pretrained=True)


        if dcnn[:6] == 'resnet':
            self.coordASPP = custom_layers.CoordASPP(self.model_conv.layer4[2].bn3.num_features,256,[1,2,3,6]) # 512 for resnet18, 2048 for resnet50
        else:
            raise NotImplementedError # for other models (squeezenet, alexnet etc.)
        self.upsample_encoder = nn.Upsample(scale_factor=4)

        self.rescale_conv = nn.Conv2d(256, 10, 3 ,padding="same")

        # elm
        self.elmPooling = nn.AvgPool2d(16) # is 56 in paper but changed to 15 to accomodate smaller resnet input shape
        self.elm1 = nn.Linear(40,1000)
        self.elm2 = nn.Linear(1000, 10, bias=False)

        # relation
        self.conv3 = nn.Conv2d(10,1,3, padding="same")
        self.att_w_c = torch.nn.Parameter(torch.zeros(20,32,32))
        self.att_b_c = torch.nn.Parameter(torch.ones(20,32,32))
        self.att_w_i = torch.nn.Parameter(torch.zeros(20,32,32))
        self.att_b_i = torch.nn.Parameter(torch.ones(20,32,32))

        self.fc3 = nn.Linear(20480, 128) # there's a large dimension change in the paper in the FC layer, it makes most sense to me to flatten first, hence the large input size
        self.fc4 = nn.Linear(128, 10)

        # decoder after relation and elm
        self.conv4 = nn.Conv2d(10, 10, 3, padding='same') # to 10 channel for mask output
        self.upsample_decoder = nn.Upsample(scale_factor=8) # revert to some resolution to see output images


        # dice loss taken from segmentation_models_pytorch
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) # binary because one hot is implemented in dataset already
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, classes=10)

    def encoder(self, x):
        # cnn block
        # summary(self.model_conv)
        cnnblock = self.model_conv(x)
        coordaspp = self.coordASPP(cnnblock)

        upsampled = self.upsample_encoder(coordaspp)
        encoder = self.rescale_conv(upsampled)
        return encoder

    def decoder(self, x):
        oselm = self.oselm(x)
        relation = self.relation(x)
        y = torch.add(oselm, relation)
        y = y.unsqueeze(2).unsqueeze(3)
        x = torch.multiply(x,y)
        x = self.conv4(x)
        x = self.upsample_decoder(x)
        return x

    def oselm(self, x):
        x = self.elmPooling(x)
        x = torch.flatten(x,start_dim=1)
        x = self.elm1(x)
        x = F.leaky_relu(x)
        x = self.elm2(x)
        return x

    def attention(self,x,y):
        x = torch.tanh(torch.add(torch.multiply(
                                    self.att_w_c,
                                    torch.cat((x,y), dim=1)
                                    ),
                                self.att_b_c)
                        )
        w = torch.softmax(self.att_w_i*x+self.att_b_i,dim=0)
        return w

    def relation(self,x):
        y = self.conv3(x)
        y = torch.tile(y,[10,1,1])
        y = self.attention(x,y)
        y = torch.flatten(y,start_dim=1)
        y = self.fc3(y)
        y = torch.tanh(y)
        y = self.fc4(y)
        y = torch.softmax(y, dim=1)
        return y

    def forward(self, image):

        image = image.float()
        x = self.encoder(image)

        mask = self.decoder(x)
        mask = torch.softmax(mask, dim=1)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)


        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multilabel", num_classes=10)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = ([x["loss"].item() for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # aggregate intersection and union over whole dataset and then compute IoU score.
        # images without some target influence per_image_iou more than dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        none_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)

        # class weights are total pixel frequencies in dataset
        class_weights = [486325588, 30685374, 1266505, 19379653, 2220869, 14231953, 2904292, 2229592, 2449537, 17317197]

        per_label_iou = np.mean([list(smp.metrics.iou_score(tp_i, fp_i, fn_i, tn_i, reduction="weighted", class_weights=class_weights)) for tp_i, fp_i, fn_i, tn_i in zip(tp, fp, fn, tn)], axis=0)

        metrics = {
            f"{stage}_loss": np.mean(loss),
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_none_iou": none_iou,
            f"{stage}_label_0_iou": per_label_iou[0],
            f"{stage}_label_1_iou": per_label_iou[1],
            f"{stage}_label_2_iou": per_label_iou[2],
            f"{stage}_label_3_iou": per_label_iou[3],
            f"{stage}_label_4_iou": per_label_iou[4],
            f"{stage}_label_5_iou": per_label_iou[5],
            f"{stage}_label_6_iou": per_label_iou[6],
            f"{stage}_label_7_iou": per_label_iou[7],
            f"{stage}_label_8_iou": per_label_iou[8],
            f"{stage}_label_9_iou": per_label_iou[9]
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

    if os.path.exists("devmode"):
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=5,
        )
    else:
        trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=15,
        )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=True)
    pprint(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=True)
    pprint(test_metrics)

    torch.save(model.state_dict(), f'{dataset}_{dcnn}_model_state_dict')