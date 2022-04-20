import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

from pprint import pprint
from torch.utils.data import DataLoader

from torchinfo import summary

import custom_layers
import sunrgbd


class ZhaoModel(pl.LightningModule):

    def __init__(self, in_channels, out_classes, **kwargs):
        super().__init__()

        #encoder
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 512, 5)
        # todo: how many conv layers are there?

        self.coordASPP = custom_layers.CoordASPP(512,256,[1,2,3,6]) # coordaspp first layer dimensions not the same as others
        self.upsample = nn.Upsample(scale_factor=4)

        # elm
        self.elmPooling = nn.AvgPool2d(56)
        self.elm1 = nn.Linear(16*28*4,1000)
        self.elm2 = nn.Linear(1000, 10, bias=False)

        # relation
        self.conv3 = nn.Conv2d(10,1,3)
        self.att_w_c = torch.nn.Parameter()
        self.att_b_c = torch.nn.Parameter()
        self.att_w_i = torch.nn.Parameter()
        self.att_b_i = torch.nn.Parameter()
        # self.conv4 = nn.Conv2d(112*112*10*2, 3) # image says conv layers but table says FC layers
        self.fc3 = nn.Linear(10*2, 128)
        self.fc4 = nn.Linear(128, out_classes)


        # dice loss taken from segmentation_models_pytorch
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def encoder(self, x):
        # cnn block
        cnn = self.pool(F.relu(self.conv1(x)))
        cnn = self.pool(F.relu(self.conv2(cnn)))

        # convaspp block
        convaspp = self.coordASPP(cnn)
        encoder = self.upsample(convaspp)

        return encoder

    def decoder(self, x):
        y = torch.add(self.oselm(x), self.relation(x))
        x = torch.multiply(x,y)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def oselm(self, x):
        x = self.elmPooling(x)
        x = self.elm1(x)
        x = F.leaky_relu(x)
        x = self.elm2(x)
        return x

    def attention(self,x,y):
        x = torch.tanh(torch.add(torch.multiply(
                                    self.att_w_c,
                                    torch.concat(x,y)
                                    ),
                                self.att_b_c)
                        )
        w = torch.softmax(self.att_w_i*x+self.att_b_i,None)
        return w


    def relation(self,x):
        y = self.conv3(x)
        y = torch.sigmoid(y)
        torch.tile(y,[112,112,10])
        y = self.attention(x,y)
        x = torch.multiply(x,y)
        return x

    def forward(self, image):
        # normalize image here? - from petmodel
        # image = (image - self.mean) / self.std
        x = self.encoder(image)
        #upsample?
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
    # download data
    root = "/media/luc/data/sunrgbd"

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

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

    model = ZhaoModel("Zhao", "resnet34", in_channels=3, out_classes=23)
    summary(model, input_size=(train_dataloader.batch_size, 3, 256, 256))
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

    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.show()