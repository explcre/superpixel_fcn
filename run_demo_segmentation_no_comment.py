import argparse
import os
import torch.backends.cudnn as cudnn
import models
import torchvision.transforms as transforms
import flow_transforms
#from scipy.ndimage import imread #commented
import imageio

#from scipy.misc import imsave #commented
from loss import *
import time
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
#import os
#import torch.backends.cudnn as cudnn
#import models
#import torchvision.transforms as transforms
#import flow_transforms
#import imageio
from loss import *
import time
import random
from glob import glob
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
# import sys
# sys.path.append('../cython')
# from connectivity import enforce_connectivity
#import cv2
#import numpy as np
#import matplotlib.pyplot as plt
from collections import Counter


#2023-5-19-3
def superpixel_covered_area(mask_image, superpixel_mask, threshold):
    final_set = []
    unique_spixels = np.unique(superpixel_mask)

    # If mask_image has more than 1 channel (is a color image), convert it to grayscale
    if mask_image.ndim > 2:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Check and correct if mask_image and spixel_mask have different shapes
    if mask_image.shape != superpixel_mask.shape:
        mask_image = cv2.resize(mask_image, (superpixel_mask.shape[1], superpixel_mask.shape[0]))

    for spixel_id in unique_spixels:
        # Create a mask for the current superpixel
        spixel_mask = np.where(superpixel_mask == spixel_id, 1, 0)
        print("spixel_mask.min(), spixel_mask.max()")
        print(spixel_mask.min(), spixel_mask.max())

        # If the last dimension of mask_image is 1, squeeze it to make mask_image 2D
        if mask_image.shape[-1] == 1:
            mask_image = np.squeeze(mask_image, axis=-1)
        
        # Find the overlap between the superpixel and the mask_image
        overlap = np.logical_and(spixel_mask, mask_image)
        # Calculate the ratio
        ratio = np.sum(overlap) / np.sum(spixel_mask)
        

        print(f"overlap shape: {overlap.shape}")
        print(f"spixel_mask shape: {spixel_mask.shape}")
        print(f"mask_image shape: {mask_image.shape}")
        print(f"overlap dtype: {overlap.dtype}")
        print(f"spixel_mask dtype: {spixel_mask.dtype}")
        print(f"mask_image dtype: {mask_image.dtype}")
        ############
        overlap = overlap.astype('uint8')
        spixel_mask = spixel_mask.astype('uint8')
        # mask_image is already uint8
        print("*"*100,"after uint8 conversion")
        print(f"overlap shape: {overlap.shape}")
        print(f"spixel_mask shape: {spixel_mask.shape}")
        print(f"mask_image shape: {mask_image.shape}")
        print(f"overlap dtype: {overlap.dtype}")
        print(f"spixel_mask dtype: {spixel_mask.dtype}")
        print(f"mask_image dtype: {mask_image.dtype}")

        
       
        
        # Ensure the dimensions are correct for cv2.merge
        overlap = np.expand_dims(overlap.astype(np.uint8) * 255, axis=-1)
        # spixel_mask = np.expand_dims(spixel_mask * 255, axis=-1)
        spixel_mask = np.expand_dims(spixel_mask.astype(np.uint8) * 255, axis=-1)
        #mask_image = np.expand_dims(mask_image *255, axis=-1)
        mask_image = np.expand_dims(mask_image, axis=-1)
        print("*"*100,"after np.expand_dims")
        print(f"overlap shape: {overlap.shape}")
        print(f"spixel_mask shape: {spixel_mask.shape}")
        print(f"mask_image shape: {mask_image.shape}")
        print(f"overlap dtype: {overlap.dtype}")
        print(f"spixel_mask dtype: {spixel_mask.dtype}")
        print(f"mask_image dtype: {mask_image.dtype}")

        # Print and save debug information
        print(f"Superpixel {spixel_id}: Ratio = {ratio}")
        debug_image = cv2.merge([overlap, spixel_mask, mask_image])
        
        cv2.imwrite(f"debug_overlap_{spixel_id}.png", debug_image)

        if ratio >= threshold:
            final_set.append(spixel_id)

    return final_set


#2023-5-19
def visualize_selected_superpixels(img, superpixel_mask, final_set):
    mask_selected = np.zeros_like(superpixel_mask)

    for spixel_id in final_set:
        mask_selected[superpixel_mask == spixel_id] = 1

    # Resize the img array to match the dimensions of mask_selected
    img = cv2.resize(img, (mask_selected.shape[1], mask_selected.shape[0]), interpolation=cv2.INTER_AREA)

    # Create a color mask for the selected superpixels
    color_mask = np.zeros_like(img)
    color_mask[mask_selected == 1] = [255, 0, 0]  # Assign red color to the selected superpixels

    # Combine the original image with the color mask
    img_selected = cv2.addWeighted(img, 1, color_mask, 0.5, 0)

    # Save debug image
    cv2.imwrite("debug_visualization.png", img_selected)

    return img_selected



def calculate_mask_distribution(mask_image_path):
    # Read the mask image
    mask_image = cv2.imread(mask_image_path, 0)  # Read as grayscale (single channel)

    # Calculate the distribution of pixel values
    pixel_counts = dict(Counter(mask_image.flatten()))

    # Calculate statistical measures
    values = list(pixel_counts.keys())
    counts = list(pixel_counts.values())

    max_value = np.max(values)
    min_value = np.min(values)
    q1_value = np.percentile(values, 25)
    mean_value = np.mean(values)
    q3_value = np.percentile(values, 75)
    median_value = np.median(values)

    # Create a histogram plot of the distribution with 50 bins
    plt.hist(mask_image.flatten(), bins=50, color='blue')
    plt.title('Mask Image Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.grid(True)

    # Save the histogram as an image
    histogram_path = 'histogram.png'
    plt.savefig(histogram_path)
    plt.close()

    # Construct the result dictionary
    result = {
        'distribution': pixel_counts,
        'max': max_value,
        'min': min_value,
        'q1': q1_value,
        'mean': mean_value,
        'q3': q3_value,
        'median': median_value,
        'histogram_path': histogram_path
    }

    return result


def generate_segmentation_path(input_path, output_dir, output_ext="png"):
    input_filename = os.path.basename(input_path)
    file_root, _ = os.path.splitext(input_filename)
    output_filename = f"{file_root}.{output_ext}"
    output_path = os.path.join(output_dir, 'segmentation', output_filename)
    return output_path


'''
def generate_segmentation_path(input_path, output_dir):
    input_filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, 'segmentation', input_filename)
    return output_path
'''


'''
input_path = "/scratch/bbsb/xu10/superpixel_fcn/demo/inputs/xx.jpg"
output_dir = "/scratch/bbsb/xu10/superpixel_fcn/demo"

output_path = generate_output_path(input_path, output_dir)
print(output_path)
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='./demo/inputs', help='path to images folder')
parser.add_argument('--data_suffix',  default='jpg', help='suffix of the testing image')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model',
                                    default= './pretrain_ckpt/SpixelNet_bsd_ckpt.tar')
parser.add_argument('--output', metavar='DIR', default= './demo' , help='path to output folder')

parser.add_argument('--downsize', default=16, type=float,help='superpixel grid cell, must be same as training setting')

parser.add_argument('-nw', '--num_threads', default=1, type=int,  help='num_threads')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

args = parser.parse_args()

random.seed(100)
@torch.no_grad()
def test(args, model, img_paths, save_path, idx):
      # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    # may get 4 channel (alpha channel) for some format
    #img_ = imread(load_path)[:, :, :3]
    img_ = imageio.imread(load_path)[:, :, :3]
    H, W, _ = img_.shape
    H_, W_  = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)

    # get spixel id
    n_spixl_h = int(np.floor(H_ / args.downsize))
    n_spixl_w = int(np.floor(W_ / args.downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    n_spixel =  int(n_spixl_h * n_spixl_w)


    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=( H_,W_), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel,  b_enforce_connect=True)

    # ************************ Save all result********************************************
    # save img, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    # img_save = (ori_img + mean_values).clamp(0, 1)
    # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))


    # save spixel viz
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    imageio.imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))

    # save the unique maps as csv, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'map_csv')):
    #     os.makedirs(os.path.join(save_path, 'map_csv'))
    # output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
    #   # plus 1 to make it consistent with the toolkit format
    # np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i',delimiter=",")


    if idx % 10 == 0:
        print("processing %d"%idx)
    ###########added##########################
    # Load the mask image
    output_dir = "/scratch/bbsb/xu10/superpixel_fcn/demo"
    mask_image_path = generate_segmentation_path(img_paths[idx],output_dir)#'path/to/mask/image'
    mask_image = cv2.imread(mask_image_path)
    print("mask image",mask_image_path)
    print(mask_image)
    mask_distribution_result=calculate_mask_distribution(mask_image_path)
    print(mask_distribution_result)
    # Calculate the final set of superpixels
    threshold = 0.5
    final_set = superpixel_covered_area(mask_image, spixel_label_map, threshold)

    # Visualize the selected superpixels on the original image
    img_selected = visualize_selected_superpixels(img_, spixel_label_map, final_set)

    # Save the selected superpixels visualization
    spixl_save_name_selected = os.path.join(save_path, 'spixel_viz', imgId + '_selected_sPixel_final.png')
    imageio.imsave(spixl_save_name_selected, img_selected)
    #######################################
    return toc


def main_seg():
    # ... (your existing code)

    # Load the mask image
    mask_image_path = 'path/to/mask/image'
    mask_image = cv2.imread(mask_image_path)

    # Calculate the final set of superpixels
    threshold = 0.5
    final_set = superpixel_covered_area(mask_image, spixel_label_map, threshold)

    # Visualize the selected superpixels on the original image
    img_selected = visualize_selected_superpixels(img_, spixel_label_map, final_set)

    # Save the selected superpixels visualization
    spixl_save_name_selected = os.path.join(save_path, 'spixel_viz', imgId + '_selected_sPixel.png')
    imageio.imsave(spixl_save_name_selected, img_selected)


def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))

    save_path = args.output
    print('=> will save everything to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tst_lst = glob(args.data_dir + '/*.' + args.data_suffix)
    tst_lst.sort()

    if len(tst_lst) == 0:
        print('Wrong data dir or suffix!')
        exit(1)

    print('{} samples found'.format(len(tst_lst)))

    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']]( data = network_data).cuda()
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True

    mean_time = 0
    for n in range(len(tst_lst)):
      time = test(args, model, tst_lst, save_path, n)
      mean_time += time
    print("avg_time per img: %.3f"%(mean_time/len(tst_lst)))



    

if __name__ == '__main__':
    main()
