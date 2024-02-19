import sys
from PIL import Image, ImageDraw, ImageFont


def add_text_to_image(input_image_path, output_image_path):
    # Open the input image
    img = Image.open(input_image_path)
    draw = ImageDraw.Draw(img)

    # Load a font
    font_path = "/home/leo/nerfstudio_reg/nerfstudio/renders/chunkfive/ChunkFive-Regular.otf"  # Specify the path to your font file
    font_size = 100  # Specify the font size
    font = ImageFont.truetype(font_path, font_size)

    offset = 50

    # Add "A" text on the lower left bottom
    text_a = "A"
    text_bbox_a = draw.textbbox((0, 0), text_a, font=font)

    # Add "B" text on the upper right
    text_b = "B"
    text_bbox_b = draw.textbbox((0, 0), text_b, font=font)

    # Create a white circle
    circle_radius = max(max(text_bbox_a), max(text_bbox_b)) * 0.6
    circle_radius_outer = max(max(text_bbox_a), max(text_bbox_b)) * 0.63
    circle_center_a = (offset+(text_bbox_a[2])/2, img.height - (text_bbox_a[3]/3) - offset)
    draw.ellipse((circle_center_a[0] - circle_radius_outer, circle_center_a[1] - circle_radius_outer,
                  circle_center_a[0] + circle_radius_outer, circle_center_a[1] + circle_radius_outer),
                 fill=(0, 0, 0))
    draw.ellipse((circle_center_a[0] - circle_radius, circle_center_a[1] - circle_radius,
                  circle_center_a[0] + circle_radius, circle_center_a[1] + circle_radius),
                 fill=(255, 255, 255))

    circle_center_b = (img.width - text_bbox_b[2]/2 - offset, text_bbox_b[1]*1.4 + offset)
    draw.ellipse((circle_center_b[0] - circle_radius_outer, circle_center_b[1] - circle_radius_outer,
                  circle_center_b[0] + circle_radius_outer, circle_center_b[1] + circle_radius_outer),
                 fill=(0, 0, 0))
    draw.ellipse((circle_center_b[0] - circle_radius, circle_center_b[1] - circle_radius,
                  circle_center_b[0] + circle_radius, circle_center_b[1] + circle_radius),
                 fill=(255, 255, 255))



    draw.text((offset, img.height - text_bbox_a[3] - offset), text_a, fill=(0, 127, 0), font=font)
    draw.text((img.width - text_bbox_b[2] - offset, -text_bbox_b[1] + offset), text_b, fill=(256, 0, 0), font=font)

    # Save the modified image
    img.save(output_image_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_image_path output_image_path")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    add_text_to_image(input_image_path, output_image_path)
