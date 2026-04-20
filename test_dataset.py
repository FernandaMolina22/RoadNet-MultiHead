# test_dataset.py (extended sanity test)
"""
1. Data pipeline validation (PRIMARY purpose)

This part checks:

DataLoader works 
Shapes are correct 
Masks are valid (0/1) 
Topology labels load correctly 

2. Model forward-pass validation (SECONDARY purpose)

This part checks:

Model accepts the data 
Output shapes match expectations 
Multi-head outputs are correct 

"""



import torch
from dataset_awr_seg import get_dataloaders
from model_unet_single import build_unet_single
from model_unet_multihead import build_unet_multihead

DATA_DIR = r"C:\Users\mfmr2\Documents\Documentos Windows\POLIMI\Thesis\Project\Amazon Wild Roads Dataset"


def main():
    # ----------------------------------------------------
    # 1. Test dataloaders (mask-only, return_topology=False)
    # ----------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=4,
        num_workers=4,
        return_topology=False,  # default, but explicit for clarity
    )
    images, masks = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch masks  shape:", masks.shape)
    print("Mask values min/max:", masks.min().item(), masks.max().item())

    # ----------------------------------------------------
    # 2. Test UNetSingle and UNetMultiHead forward passes
    #    using mask-only batch
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dev = images.to(device)

    # 2a. UNetSingle
    model_single = build_unet_single(device)
    with torch.no_grad():
        logits_single = model_single(images_dev)
    print("UNetSingle output shape:", logits_single.shape)

    # 2b. UNetMultiHead
    model_multi = build_unet_multihead(device)
    with torch.no_grad():
        outputs_multi = model_multi(images_dev)
    for head_name, tensor in outputs_multi.items():
        print(f"UNetMultiHead '{head_name}' shape:", tensor.shape)

    # ----------------------------------------------------
    # 3. Test dataloaders with topology=True
    #    Expect: batch = (images, masks, centers, junctions)
    # ----------------------------------------------------
    print("\nTesting loader with topology=True...")
    train_loader_topo, _, _ = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=2,      # small batch is enough for sanity check
        num_workers=0,     # simpler for quick test
        return_topology=True,
    )

    batch = next(iter(train_loader_topo))
    print("len(batch) with topology:", len(batch))

    imgs, masks_t, centers, junctions = batch
    print("Images shape:     ", imgs.shape)
    print("Masks shape:      ", masks_t.shape)
    print("Centers shape:    ", centers.shape)
    print("Junctions shape:  ", junctions.shape)


if __name__ == "__main__":
    main()
