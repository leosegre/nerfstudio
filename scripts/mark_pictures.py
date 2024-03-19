import sys
from PIL import Image, ImageDraw, ImageFont


def add_text_to_image(input_image_path, output_image_path, delta_t=None,delta_r=None):
    # Open the input image
    img = Image.open(input_image_path)
    draw = ImageDraw.Draw(img)

    # Load a font
    font_path = "/home/leo/nerfstudio_reg/nerfstudio/renders/chunkfive/ChunkFive-Regular.otf"  # Specify the path to your font file
    font_size = 100  # Specify the font size
    font = ImageFont.truetype(font_path, font_size)
    font2 = ImageFont.truetype(font_path, 70)
    font3 = ImageFont.truetype(font_path, 60)


    offset = 50

    # Add "A" text on the lower left bottom
    text_a = "A"
    text_bbox_a = draw.textbbox((0, 0), text_a, font=font)

    # Add "B" text on the upper right
    text_b = "B"
    text_bbox_b = draw.textbbox((0, 0), text_b, font=font)

    # Create a white circle
    circle_radius = max(max(text_bbox_a), max(text_bbox_b)) * 0.63
    circle_radius_outer = max(max(text_bbox_a), max(text_bbox_b)) * 0.63
    circle_center_a = (offset+(text_bbox_a[2])/2, img.height - (text_bbox_a[3]/3) - offset)
    # draw.ellipse((circle_center_a[0] - circle_radius_outer, circle_center_a[1] - circle_radius_outer,
    #               circle_center_a[0] + circle_radius_outer, circle_center_a[1] + circle_radius_outer),
    #              fill=(0, 0, 0))
    # draw.ellipse((circle_center_a[0] - circle_radius, circle_center_a[1] - circle_radius,
    #               circle_center_a[0] + circle_radius, circle_center_a[1] + circle_radius),
    #              fill=(255, 255, 255), width=3, outline="black")

    circle_center_b = (img.width - text_bbox_b[2]/2 - offset, text_bbox_b[1]*1.4 + offset)
    # draw.ellipse((circle_center_b[0] - circle_radius_outer, circle_center_b[1] - circle_radius_outer,
    #               circle_center_b[0] + circle_radius_outer, circle_center_b[1] + circle_radius_outer),
    #              fill=(0, 0, 0))
    # draw.ellipse((circle_center_b[0] - circle_radius, circle_center_b[1] - circle_radius,
    #               circle_center_b[0] + circle_radius, circle_center_b[1] + circle_radius),
    #              fill=(255, 255, 255), width=3, outline="black")



    draw.text((offset, img.height - text_bbox_a[3] - offset), text_a, fill=(0, 127, 0), font=font, stroke_width=3, stroke_fill=3)
    # draw.text((offset+8, img.height - text_bbox_a[3] - offset + 2), text_b, fill=(255, 0, 0), font=font, stroke_width=3, stroke_fill=3)
    draw.text((img.width - text_bbox_b[2] - offset, -text_bbox_b[1] + offset), text_b, fill=(255, 0, 0), font=font, stroke_width=3, stroke_fill=3)

    if delta_r is not None:
        text_t_r = "Translation Error: " + str(delta_t) + "\n" + "Rotation Error: " + str(delta_r + u'\N{DEGREE SIGN}')
        text_bbox_t_r = draw.textbbox((0, 0), text_t_r, font=font2)
        # text_r = "Rotation Error: " + str(delta_r)
        # text_bbox_r = draw.textbbox((0, 0), text_r, font=font)
        print(text_bbox_t_r)
        xy=(text_bbox_t_r[0] + img.width/2 - text_bbox_t_r[2]/2,
                        text_bbox_t_r[1] + img.height - text_bbox_t_r[3] - offset,
                        text_bbox_t_r[2]/2 + img.width/2,
                        text_bbox_t_r[3] + img.height - text_bbox_t_r[3] - offset)
        draw.rounded_rectangle(xy, fill="white", outline="black", width=3, radius=10)
        xy_text = (xy[0] + 60,
                   xy[1] ,
                   xy[2] ,
                   xy[3])
        draw.multiline_text(xy_text, text_t_r, font=font3, fill="black", align="center")

    # Save the modified image
    img.save(output_image_path)


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        print("Usage: python script.py input_image_path output_image_path")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    if len(sys.argv) == 5:
        delta_t = sys.argv[3]
        delta_r = sys.argv[4]
    else:
        delta_t = None
        delta_r = None

    add_text_to_image(input_image_path, output_image_path, delta_t, delta_r)
