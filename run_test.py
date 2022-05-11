from main import ZhaoModel
import sys, os, torch
from datasetbuilders import iitaff, sunrgbd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pprint import pprint


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
    with open("/home/luc/Documents/iitaff_resnet50_model_state_dict") as file:
        torch.load(file)

    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=True)
    pprint(test_metrics)