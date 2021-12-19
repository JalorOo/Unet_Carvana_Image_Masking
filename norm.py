import torch


def dice_coef(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().detach().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().detach().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)