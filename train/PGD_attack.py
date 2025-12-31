import torch
from data.dataset_test import Data
import torch.nn as nn
from PIL import Image


def attack(
    image_vis,
    image_ir,
    image_gt,
    model,
    loss,
    step_size=1 / 255,
    total_steps=3,
    epsilon=4 / 255,
):
    """
    Projected Gradient Descent (PGD) adversarial attack.
    
    Args:
        image_vis: Visible image tensor
        image_ir: Infrared image tensor
        image_gt: Ground truth image tensor
        model: Model to attack
        loss: Loss function
        step_size: Step size for gradient update
        total_steps: Number of attack iterations
        epsilon: Maximum perturbation bound (L∞ norm)
    
    Returns:
        adv_img_vis: Adversarial visible image
        adv_img_ir: Adversarial infrared image
    """
    model.eval()

    # random start in [-epsilon, epsilon]
    adv_img_vis = (image_vis + torch.empty_like(image_vis).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()
    adv_img_ir  = (image_ir  + torch.empty_like(image_ir ).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()

    adv_img_vis.requires_grad_(True)
    adv_img_ir.requires_grad_(True)

    for _ in range(total_steps):
        model.zero_grad(set_to_none=True)
        if adv_img_vis.grad is not None:
            adv_img_vis.grad = None
        if adv_img_ir.grad is not None:
            adv_img_ir.grad = None

        logits = model(adv_img_vis, adv_img_ir)
        loss_total = loss(logits, image_gt)
        loss_total.backward()

        with torch.no_grad():

            assert adv_img_vis.grad is not None and adv_img_ir.grad is not None
            adv_img_vis = adv_img_vis + step_size * adv_img_vis.grad.sign()
            adv_img_ir  = adv_img_ir  + step_size * adv_img_ir .grad.sign()

            adv_img_vis = image_vis + (adv_img_vis - image_vis).clamp(-epsilon, epsilon)
            adv_img_ir  = image_ir  + (adv_img_ir  - image_ir ).clamp(-epsilon, epsilon)

            adv_img_vis.clamp_(0, 1)
            adv_img_ir.clamp_(0, 1)

        adv_img_vis = adv_img_vis.detach().requires_grad_(True)
        adv_img_ir  = adv_img_ir .detach().requires_grad_(True)

    return adv_img_vis, adv_img_ir


if __name__ == "__main__":

    dataset = Data(mode='train', img_dir='/data/ykx/MSRS/train')
    image_ir, image_vis = dataset[0]['ir'], dataset[0]['y']
    image_vis.unsqueeze_(0)
    image_ir.unsqueeze_(0)
    image_vis = image_vis
    image_ir = image_ir
    
    image_gt = (image_vis + image_ir) / 2


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(2, 1, 3, 1, 1)
        def forward(self, x, y):
            x = torch.cat((x, y), 1)
            return self.conv(x)

    model = Net()
    model.eval()

    fused = model(image_vis, image_ir)    

    loss = nn.MSELoss()
    attack_image_vis, attack_image_ir = attack(image_vis, image_ir, image_gt, model, loss)

    Image.fromarray(image_ir.squeeze().detach().cpu().numpy()*255).convert('L').save('output/image_ir.png')
    Image.fromarray(image_vis.squeeze().detach().cpu().numpy()*255).convert('L').save('output/image_vis.png')
    Image.fromarray(attack_image_vis.squeeze().detach().cpu().numpy()*255).convert('L').save('output/attack_image_vis.png')
    Image.fromarray(attack_image_ir.squeeze().detach().cpu().numpy()*255).convert('L').save('output/attack_image_ir.png')
    
    print(attack_image_vis.shape, attack_image_ir.shape)


