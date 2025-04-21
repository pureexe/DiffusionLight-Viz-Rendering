import argparse
import os 
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from functools import partial
import skimage
import ezexr

INPUT_DIR = "example/blender_rendered"
OUTPUT_DIR = "output/cropped"

def reexpose_hdr(hdrim, percentile=90, max_mapping=0.8, alpha=None):
    """
    :param img: HDR image
    :param percentile:
    :param max_mapping:
    :return:
    """
    r_percentile = np.percentile(hdrim, percentile)
    if alpha==None:
        alpha = max_mapping / (r_percentile + 1e-10)
    return alpha * hdrim, alpha

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR ,help='input directory that contain render file') 
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,help='input directory that contain render file')
    parser.add_argument("--subdir", type=str, default="mirror,matte_silver,diffuse" ,help='subdirectory in input directory')
    parser.add_argument('--clip', action='store_true', help='clip image to 0-1')
    parser.add_argument("--mode", type=str, default="front" ,help='cropping mode default or front')

    parser.add_argument('--gamma', type=float, default=2.4, help='gamma value' )
    parser.add_argument('--max_mapping', type=float, default=0.9, help='max mapping value' )
    parser.add_argument('--percentile', type=float, default=97.5, help='max mapping value' )
    parser.add_argument('--use_whitebg', type=int, default=1, help='apply white background (0 to not apply, default 1)' )

    return parser

#=============================================
class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img
# =============================================
def process_image(args, path):
    image_path = os.path.join(args.input_dir, path[0], path[1])
    image = ezexr.imread(image_path)
    if args.mode == "front":
        image = image[128:412,338:622]
    else:
        image = image[130:410,340:620]
    if image.shape[2] == 4:
        mask_alpha = image[...,3]
    if args.clip:
        image = np.clip(image, 0, 1)
    else: 
        tonemapper = TonemapHDR(args.gamma, args.percentile, args.max_mapping)
        #image, _ = reexpose_hdr(image[...,:3])
        image, _,_ = tonemapper(image[...,:3])
    
    image = np.concatenate([image, mask_alpha[...,None]], axis=2)
    if args.use_whitebg:
        # Assume image is your [H, W, 4] NumPy array with values in [0, 1]
        rgb = image[..., :3]
        alpha = image[..., 3:]

        # Blend with white background
        white_bg = np.ones_like(rgb)
        image = rgb * alpha + white_bg * (1 - alpha)
    
    image = skimage.img_as_ubyte(image)
    skimage.io.imsave(os.path.join(args.output_dir, path[0], path[1].replace(".exr", ".png")), image)
    return None

def main():
    args  = create_argparser().parse_args()
    files_queue = []
    subdirs = [f.strip() for f in args.subdir.split(",")]
    for subdir in subdirs:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
        subdir_path = os.path.join(args.input_dir, subdir)
        files = sorted(os.listdir(subdir_path))
        files = [f for f in files if f.endswith(".exr")]
        files_queue = files_queue + [(subdir, f) for f in files]
    
    fn = partial(process_image, args)
    fn(files_queue[0])
    with Pool(16) as p:
        r = list(tqdm(p.imap(fn, files_queue), total=len(files_queue)))
        
if __name__ == "__main__":
    main()
