import sys
from PIL import Image, ImageDraw, ImageFont
import torch
import random
import numpy as np


def mark_indices(draw, indices, color):
    for idx in indices:
        x = idx[1]
        y = idx[0]
        draw.rectangle([x, y, x + 5, y + 5], fill=color)
        # circle_radius = 2
        # draw.ellipse((x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius), fill=color, outline="white")
def sample_method(mask):
    nonzero_indices = torch.nonzero(mask, as_tuple=False)
    chosen_indices = random.sample(range(len(nonzero_indices)), k=16000)
    indices = nonzero_indices[chosen_indices]
    return indices

def mark_samples(input_image_path, input_mask_path, output_image_path):
    # Open the input image
    img_masked = Image.open(input_image_path)
    img_masked.putalpha(255)
    img_random = Image.open(input_image_path)
    img_random.putalpha(255)
    mask = Image.open(input_mask_path)
    draw_masked = ImageDraw.Draw(img_masked)
    draw_random = ImageDraw.Draw(img_random)


    torch_mask = torch.tensor(np.array(mask)).squeeze()
    masked_indices = sample_method(torch_mask)
    random_indices = sample_method(torch.ones_like(torch_mask))



    # Mark masked indices in green and random indices in red
    mark_indices(draw_masked, masked_indices, (0, 255, 0))  # Green
    mark_indices(draw_random, random_indices, (255, 0, 0))  # Red
    # Save the modified image
    img_masked.save(output_image_path + "/marked_masked_img.png")
    img_random.save(output_image_path + "/marked_random_img.png")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py input_image_path output_image_path")
        sys.exit(1)

    input_image_path = sys.argv[1]
    input_mask_path = sys.argv[2]
    output_image_path = sys.argv[3]

    mark_samples(input_image_path, input_mask_path, output_image_path)
