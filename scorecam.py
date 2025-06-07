# part of the code is from https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""
Created on Wed Apr 29 16:11:20 2020

@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import re

from misc_functions import (get_example_params, save_class_activation_images,
                            preprocess_image, read_dot_images, apply_colormap_on_image)
import os, sys
from cornet.cornet_z import CORnet_Z


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
        Performs a forward pass through convolutional layers, hooks into the target layer,
        and registers a backward hook to capture gradients.

        :param x: Input tensor
        :return: (target layer activation, final conv output)
        """
        conv_output = None

        # Iterate through convolutional layers
        for name, layer in zip(['V1', 'V2', 'V4', 'IT'], [self.model.V1, self.model.V2, self.model.V4, self.model.IT]):
            x = layer(x)
            if name == self.target_layer:
                conv_output = x  # Store activations at the target layer

        return conv_output, x

    def forward_pass(self, x):
        """
        Runs a full forward pass, capturing activations at the target layer.

        :param x: Input tensor
        :return: (target layer activations, final model output)
        """
        # Forward pass through convolutional layers
        conv_output, x = self.forward_pass_on_convolutions(x)

        # Forward pass through the classifier
        x = self.model.decoder(x)

        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        # print(model_output)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            # print(input_image.shape)
            saliency_map = F.interpolate(saliency_map, size=(300, 300), mode='bilinear', align_corners=False)
            # saliency_map = F.interpolate(saliency_map, size=(256, 256), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.Resampling.LANCZOS))/255
        return cam


def produce_scorecam(img_path, model):
    original_image = Image.open(img_path).convert('RGB')
    prep_img = preprocess_image(original_image)
    cam_list = []
    for layer in ['V1', 'V2', 'V4', 'IT']:
        score_cam = ScoreCam(model, target_layer=layer)
        cam_list.append(score_cam.generate_cam(prep_img, target_class=None))
    return np.array(cam_list)


def produce_scorecam_and_viz(img_path, model):
    cam_list = produce_scorecam(img_path, model)

    # fig, ax = plt.subplots(1, 8, figsize=(12, 2))
    fig, ax = plt.subplots(1, 4, figsize=(8, 2))
    # ax[0].imshow(original_image)
    # ax[0].set_title('Original Image')
    for i in range(4):
        # ax[i*2].imshow(cam_list[i])
        ax[i].imshow(cam_list[i])
        # _, heatmap_on_image = apply_colormap_on_image(original_image, cam_list[i], 'viridis')
        # ax[i*2+1].imshow(heatmap_on_image)
    for a in ax:
        a.axis('off')
    fig.tight_layout(pad=0.2)
    return fig


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    model_name = "untrained" # "dewind" or "imagenet" or "untrained"
    pretrained_model = CORnet_Z()
    if model_name == "imagenet":
        state_dict = torch.load('./cornet/cornet_z-5c427c9c.pth', weights_only=False)['state_dict']
        pretrained_model.load_state_dict(state_dict)
    if model_name == "dewind":
        out_features = 10
        pretrained_model.decoder.linear = nn.Linear(512, out_features)
        state_dict = torch.load('./cornet/cornetz_testolin_dewind.ckpt')['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'model.' in k or 'module.' in k:
                new_state_dict[k.replace('model.', '').replace('module.', '')] = v
            else:
                new_state_dict[k] = v
        del state_dict
        pretrained_model.load_state_dict(new_state_dict)

    pretrained_model.eval()
    # print(pretrained_model)

    # # Produce scorecam
    labels = ['6', '7', '9', '10', '12', '14', '17', '20', '24', '29']
    for label in labels:
        img_dir = f'./data/images/test_dewind_small/{label}/'
        # img_dir = f'./data/images/{label}/'
        img_list = sorted(os.listdir(img_dir), key=natsort)
        cam_list = []
        for img in tqdm(img_list):
            cam_list.append(produce_scorecam(img_dir + img, pretrained_model))
        np.save(f'./results/scorecam/DeWind_test_small_array/dewind/{label}.npy', np.array(cam_list))

    # # save fig
    #         fig = produce_scorecam(img_dir + img, pretrained_model)
    #         fig.savefig(f'../results/number_neurons/scorecam/{label}/' + img)
    # # fig = produce_scorecam('/home/nhut/Desktop/number_neurons/data/images/test_dewind_small/6/1.png', pretrained_model)
    # # fig.savefig('../results/number_neurons/tmp.png')


    # # Example on 1 image
    # img_path = '../input_images/Test.png'
    # # target_class = 4  # 29 DeWind 6 ImageNet 2 Untrained
    # file_name_to_export = 'Test_full.png'
    # # Read image
    # original_image = Image.open(img_path).convert('RGB')
    # # Process image
    # prep_img = preprocess_image(original_image)
    # # Define model
    # for layer in ['V1', 'V2', 'V4', 'IT']:
    #     score_cam = ScoreCam(pretrained_model, target_layer=layer)
    #     # Generate cam mask
    #     cam = score_cam.generate_cam(prep_img, target_class=None)
    #     # Save mask
    #     save_class_activation_images(original_image, cam, 'scorecam/' + file_name_to_export[:-4] + '_' + layer)
