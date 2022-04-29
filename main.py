import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock

from pprint import pprint
from torch.utils.data import DataLoader

from torchinfo import summary

import custom_layers
import sunrgbd

class MyResNet18(ResNet):
    def __init__(self):
        #todo look into basicblock shape
        super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])

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

        self.model_conv = MyResNet18()
        self.model_conv.load_state_dict(torchvision.models.resnet18(pretrained=True).state_dict())

        # self.model_conv = torchvision.models.alexnet(pretrained=True)
        # self.model_conv = torchvision.models.squeezenet1_0(pretrained=True)
        # self.model_conv = torchvision.models.vgg16(pretrained=True)

        self.coordASPP = custom_layers.CoordASPP(512,256,[1,2,3,6])
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
        self.conv4 = nn.Conv2d(10, 1, 3, padding='same') # to 1 channel for mask output
        self.upsample_decoder = nn.Upsample(scale_factor=8) # revert to some resolution to see output images

        # dice loss taken from segmentation_models_pytorch
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

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
        # y = torch.sigmoid(y) # no reason for sigmoid, not in paper
        y = torch.tile(y,[10,1,1])
        y = self.attention(x,y)
        y = torch.flatten(y,start_dim=1)
        y = self.fc3(y)
        y = torch.tanh(y)
        y = self.fc4(y)
        y = torch.softmax(y, dim=1)
        return y

    def forward(self, image):
        # normalize image here? - from petmodel
        # image = (image - self.mean) / self.std
        image = image.float()
        x = self.encoder(image)

        mask = self.decoder(x)
        # mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        # assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

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

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
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

    if os.path.exists("devmode"):
        root = "/media/luc/data/sunrgbd"
    else:
        root = "/home/schootuiterkampl/sunrgbd/"

    train_dataset = sunrgbd.sunrgbd(root, "train")
    valid_dataset = sunrgbd.sunrgbd(root, "valid")
    test_dataset = sunrgbd.sunrgbd(root, "test")


    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    train_dataset.__getitem__(1)

    model = ZhaoModel(in_channels=3, out_classes=23)
    # summary(model, input_size=(train_dataloader.batch_size, 3, 256, 256))
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
    )
    print('fit')
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)

    torch.save(model.state_dict(), '.')

    # batch = next(iter(test_dataloader))
    # with torch.no_grad():
    #     model.eval()
    #     logits = model(batch["image"])
    # pr_masks = logits.sigmoid()
    #
    # for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
    #     plt.figure(figsize=(10, 5))
    #
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    #     plt.title("Image")
    #     plt.axis("off")
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(gt_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    #     plt.title("Ground truth")
    #     plt.axis("off")
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(pr_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    #     plt.title("Prediction")
    #     plt.axis("off")
    #
    #     plt.show()