import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from .gradcam import GradCAM


def visualize_cam(mask, img, alpha=1.0):
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result
    

def unnormalize(image, mean, std, out_type='array'):
    if type(image) == torch.Tensor:
        image = np.transpose(image.clone().numpy(), (1, 2, 0))
    
    normal_image = image * std + mean
    if out_type == 'tensor':
        return torch.Tensor(np.transpose(normal_image, (2, 0, 1)))
    elif out_type == 'array':
        return normal_image
    return None  # No valid value given


def to_numpy(tensor):
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


class GradCAMView:

    def __init__(self, model, layers, device, mean, std):
        self.model = model
        self.layers = layers
        self.device = device
        self.mean = mean
        self.std = std
        self._gradcam()
        self.grad = self.gradcam.copy()
        self.views = []

    def _gradcam(self):
        self.gradcam = {}
        for layer in self.layers:
            self.gradcam[layer] = GradCAM(self.model, layer)
    
    def _cam_image(self, norm_image):
        norm_image_cuda = norm_image.clone().unsqueeze_(0).to(self.device)
        heatmap, result = {}, {}
        for layer, gc in self.gradcam.items():
            mask, _ = gc(norm_image_cuda)
            cam_heatmap, cam_result = visualize_cam(
                mask,
                unnormalize(norm_image, self.mean, self.std, out_type='tensor').clone().unsqueeze_(0).to(self.device)
            )
            heatmap[layer], result[layer] = to_numpy(cam_heatmap), to_numpy(cam_result)
        return {
            'image': unnormalize(norm_image, self.mean, self.std),
            'heatmap': heatmap,
            'result': result
        }
    
    def _plot_view(self, view, fig, row_num, ncols, metric):
        sub = fig.add_subplot(row_num, ncols, 1)
        sub.axis('off')
        plt.imshow(view['image'])
        sub.set_title(f'{metric.title()}:')
        for idx, layer in enumerate(self.layers):
            sub = fig.add_subplot(row_num, ncols, idx + 2)
            sub.axis('off')
            plt.imshow(view[metric][layer])
            sub.set_title(layer)
    
    def cam(self, norm_image_list):
        for norm_image in norm_image_list:
            self.views.append(self._cam_image(norm_image))
    
    def plot(self, plot_path):
        for idx, view in enumerate(self.views):
            # Initialize plot
            fig = plt.figure(figsize=(10, 10))

            # Plot view
            self._plot_view(view, fig, 1, len(self.layers) + 1, 'heatmap')
            self._plot_view(view, fig, 2, len(self.layers) + 1, 'result')
            
            # Set spacing and display
            fig.tight_layout()
            plt.show()

            # Save image
            fig.savefig(f'{plot_path}_{idx + 1}.png', bbox_inches='tight')

            # Clear cache
            plt.clf()
    
    def __call__(self, norm_image_list, plot_path):
        self.cam(norm_image_list)
        self.plot(plot_path)