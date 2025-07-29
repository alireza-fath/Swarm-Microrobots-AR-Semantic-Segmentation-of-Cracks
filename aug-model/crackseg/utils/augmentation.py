import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import linalg, stats

def plt_display(images):
    """Display a single image or ndarray of images using matplotlib."""
    # If images is a single image and not a list/array, make it a single-item list
    images = images.squeeze()
    rgb = images.shape[-1] == 3

    if type(images) == np.ndarray:
        if images.ndim == 3 + rgb:
            images = list(images)
        else:
            images = [images]

    elif type(images) != list:
        images = [images]
    
    fig, axs = plt.subplots(len(images), 1, figsize=(10, 10*len(images)))
    if len(images) == 1:
        axs = [axs]
    for ax, im in zip(axs, images):
        ax.grid(False)
        if im.ndim == 2:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()


def load_images(path, num=None):
    """Load a single image or folder of images from a path."""
    if os.path.isfile(path):
        image = cv2.imread(path)
        return image
    
    list_dir = os.listdir(path)
    num = len(list_dir) if num == None else num
    images = []
    
    for file_name, i in zip(list_dir, range(num)):
        image = cv2.imread(os.path.join(path, file_name), )
        if image is not None:
            images.append(image)
            
    return images

def load_images_and_masks(path_images, path_masks, num=None, shuffle=True, replace=False):
    """Load a matched set of images and masks"""
    common_files = list(set(os.listdir(path_images)) & set(os.listdir(path_masks)))
    
    if shuffle:
        if replace:
            chosen_files = np.random.choice(common_files, size=num, replace=True)
        else:
            chosen_files = np.random.choice(common_files, size=min(num, len(common_files)), replace=False)
    else:
        chosen_files = common_files[:min(num, len(common_files))]
    
    # images = [cv2.imread(os.path.join(path_images, f)) for f in chosen_files]
        
    images = [np.asarray(Image.open(os.path.join(path_images, f))) for f in chosen_files]
    # masks = [np.asarray(Image.open(os.path.join(path_masks, f)).convert('L')) for f in chosen_files]
    masks = [cv2.imread(os.path.join(path_masks, f), 0) for f in chosen_files]
    
    return images, masks


def open_image_np(path):
    im = Image.open(path)
    return np.asarray(im)


def resize_image(image, mask, W, H):
    H_im, W_im, _ = image.shape

    if H_im == H and W_im == W:
        return image

    r = max(H / H_im, W / W_im)
    H_scale = round(H_im * r)
    W_scale = round(W_im * r)
    scaled_image = cv2.resize(image, (W_scale, H_scale))
    scaled_mask = cv2.resize(mask, (W_scale, H_scale))

    Y_min = (H_scale - H) // 2
    X_min = (W_scale - W) // 2

    scaled_image = scaled_image[Y_min : Y_min + H, X_min : X_min + W, :]
    scaled_mask = scaled_mask[Y_min : Y_min + H, X_min : X_min + W]

    return scaled_image, scaled_mask


def resize_images(images, masks, W, H):
    resized_images = np.zeros((len(images), H, W, 3), dtype=np.uint8)
    resized_masks = np.zeros((len(masks), H, W), dtype=np.uint8)

    for idx, (image, mask) in enumerate(zip(images, masks)):
        resized_images[idx], resized_masks[idx] = resize_image(image, mask, W, H)

    return resized_images, resized_masks


def transform_perspective_image(image, mask, theta = 0.35):
    """Transform perspective of an image based on randomly selected new corners."""
    H, W = image.shape[:2]
    
    dx_tl, dx_bl, dx_br, dx_tr = np.random.randint(0, theta * W, 4)
    dy_tl, dy_bl, dy_br, dy_tr = np.random.randint(0, theta * H, 4)

    to_points = np.float32([[0, 0], [0, H - 1], [W - 1, H - 1], [W - 1, 0]])
    from_points = np.float32([[dx_tl, dy_tl], 
                            [dx_bl, H - dy_bl], 
                            [W - dx_br, H - dy_br], 
                            [W - dx_tr, dy_tr]])
    
    transform_matrix = cv2.getPerspectiveTransform(from_points, to_points)
    transformed_image = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))
    transformed_mask = cv2.warpPerspective(mask, transform_matrix, (image.shape[1], image.shape[0]))

    return transformed_image, transformed_mask


def transform_perspective(images, masks, theta = 0.35):
    adjusted_images = []
    adjusted_masks = []

    for idx in range(len(images)): 
        adjusted_img, adjusted_mask = transform_perspective_image(images[idx], masks[idx], theta)
        adjusted_images.append(adjusted_img)
        adjusted_masks.append(adjusted_mask)

    return adjusted_images, adjusted_masks



def compute_pca(images):
    """Compute PCA parameters based on sample for adjusting color distribution."""
    if type(images) == list:
        images = np.array(images)
    images_norm = images / 255
    cov = np.cov(images_norm.reshape((-1, 3)), rowvar=False)
    lam, p= linalg.eigh(cov)

    return lam, p


def recolor(images, lam, p, sigma=0.1): 

    adjusted_images = np.array(images)

    for idx in range(images.shape[0]):
        alpha = np.random.normal(0, sigma, 3)
        adjustment = (np.dot(p, (lam * alpha)) * 255).astype(np.int16)
        adjustment = np.tile(adjustment, images.shape[1:-1] + (1,))

        adjusted = cv2.add(images[idx], adjustment, dtype=cv2.CV_8U)
        adjusted_images[idx] = adjusted

    return adjusted_images


def apply_gradient(images, direction="horizontal", grad_min=-50, grad_max=50, randomize_grad=True):
    # adjusted_images = np.array(images)
    adjusted_images = []


    for idx in range(len(images)):
        height, width, channels= images[idx].shape
        grad_start = np.random.uniform(grad_min, grad_max, 1)[0] if randomize_grad else grad_min
        grad_end = np.random.uniform(grad_min, grad_max, 1)[0] if randomize_grad else grad_max

        if direction == "horizontal":
            grad = np.tile(np.linspace(grad_start, grad_end, num=width, dtype=images[idx].dtype), height).reshape([height, width, 1])
            grad = np.repeat(grad, 3, axis=2)
            # grad = grad.astype(np.uint8)

            # adjusted_images[idx] = cv2.add(images[idx], grad, dtype=cv2.CV_8U)
            # print(f"image resolution: {images[idx].shape}")
            # print(f"grad resolution: {grad.shape}")
            adjusted_images.append(cv2.add(images[idx], grad))

        if direction == "vertical":
            return NotImplementedError

    return adjusted_images


def random_crop_image(image, mask, crop_size):
    """Crop a random selection of an image"""

    height, width = image.shape[:2]
    crop_height, crop_width = crop_size

    if (crop_height > height) or (crop_width > crop_width):
        raise ValueError("Crop size must be smaller than the image dimensions")
    
    # Randomly choose the top-left corner of the cropping box
    y = np.random.randint(0, height - crop_height + 1)
    x = np.random.randint(0, width - crop_width + 1)

    return image[y:y+crop_height, x:x+crop_width], mask[y:y+crop_height, x:x+crop_width]


def random_crop(images, masks, crop_size=(1500, 1500)):
    adjusted_images = []
    adjusted_masks = []

    for idx in range(len(images)): 
        adjusted_img, adjusted_mask = random_crop_image(images[idx], masks[idx], crop_size)
        adjusted_images.append(adjusted_img)
        adjusted_masks.append(adjusted_mask)

    return adjusted_images, adjusted_masks


def get_split_slices(image, split_size):
    """Get slices for splitting an image into equally sized smaller images."""
    image_height, image_width = image.shape[:2]
    split_height, split_width = split_size
    slices = []

    for y in range(0, image_height - split_height, split_height):
        for x in range(0, image_width - split_width, split_width):
            slices.append(np.s_[y:y+split_height, x:x+split_width])

    return slices


def weighted_crop(image, mask, crop_size):
    """Crop an image based on a weighted distribution."""
    slices = get_split_slices(image, crop_size)
    max_crop = max(slices, key=lambda s: np.sum(mask[s]))

    return image[max_crop], mask[max_crop]