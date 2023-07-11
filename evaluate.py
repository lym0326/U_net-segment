import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from hausdorff import hausdorff_distance
from utils.dice_score import multiclass_dice_coeff, dice_coeff, diceCoeffv2, jaccardv2


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    train_class_dices = np.array([0] * 3, dtype=float)
    train_class_jaccard = np.array([0] * 3, dtype=float)
    train_class_hsdf = np.array([0] * 3, dtype=float)
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, 4).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 4).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:,...], mask_true[:, 1:,...], reduce_batch_first=False)

            class_dice = []
            val_jaccard_arr = []
            val_hsdf_arr = []
            for i in range(1, 4):
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['

                cur_dice = diceCoeffv2(mask_pred[:, i:i + 1,...],
                                       mask_true[:, i:i + 1,...]).cpu().item()
                class_dice.append(cur_dice)
                val_jaccard_arr.append(jaccardv2(mask_pred[:, i:i + 1,...], mask_true[:, i:i + 1,...]).cpu())
                val_hsdf_arr.append(
                    hausdorff_distance(np.array(mask_pred[0, i:i + 1,...].squeeze().cpu()), np.array(mask_true[0, i:i + 1,...].squeeze().cpu()),
                                       distance='manhattan'))

            train_class_dices += np.array(class_dice)
            train_class_jaccard += np.array(val_jaccard_arr)
            train_class_hsdf += np.array(val_hsdf_arr)

        train_class_dices = train_class_dices / max(num_val_batches, 1)
        train_class_jaccard = train_class_jaccard / max(num_val_batches, 1)
        train_class_hsdf = train_class_hsdf / max(num_val_batches, 1)
        print(
            'dice_lvm: {:.4} - dice_lv: {:.4} - dice_rv: {:.4}'.format(
                train_class_dices[0], train_class_dices[1],
                train_class_dices[2]))
        print(
            'dice_lvm: {:.4} - dice_lv: {:.4} - dice_rv: {:.4}'.format(
                train_class_jaccard[0], train_class_jaccard[1],
                train_class_jaccard[2]))
        print(
            'dice_lvm: {:.4} - dice_lv: {:.4} - dice_rv: {:.4}'.format(
                train_class_hsdf[0], train_class_hsdf[1],
                train_class_hsdf[2]))
    net.train()
    return dice_score / max(num_val_batches, 1)
