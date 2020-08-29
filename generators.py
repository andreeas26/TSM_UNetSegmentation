import pathlib as pt
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from keras.preprocessing import image

def augmentator(images, masks):
    """
    Function to do data augmentation on images and masks as well
    
    Args:
        images(numpy array):
        masks(numpy array)

    """
    spatial_aug = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertical flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale=(0.8, 1.2),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-20, 20),
            # shear=(-20, 20),
            order=[1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=125,  # if mode is constant, use a cval between 0 and 255
            mode="reflect",
            name="Affine")
    ], random_order=True)

    blur_aug = iaa.Sequential([
        # Blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.OneOf([
                          iaa.GaussianBlur(sigma=(0, 0.5)),
                          iaa.AverageBlur(k=(3, 7)),
                          iaa.MedianBlur(k=(3, 7)),
                      ])
                      )

    ], random_order=True)

    elastic_aug = iaa.Sometimes(0.5, [iaa.ElasticTransformation(alpha=(30, 60), sigma=10)])

    other_aug = iaa.Sequential([
        iaa.Sometimes(0.5, [
            iaa.OneOf([
                iaa.contrast.CLAHE(clip_limit=2),
                iaa.contrast.GammaContrast(gamma=(0.5, 2.0))
            ]),
            # change brightness of images
            iaa.Add((-40, 40))
        ])
    ], random_order=True)

    # Freeze randomization to apply same to labels
    spatial_det = spatial_aug.to_deterministic()
    elastic_det = elastic_aug.to_deterministic()

    # when input mask is float32, the no channels must be 3 as it would be 3 classes.
    segmaps = [SegmentationMapOnImage(m, nb_classes=3, shape=images[i].shape) for i, m in enumerate(masks)]

    aug_images, aug_masks = spatial_det.augment_images(images), spatial_det.augment_segmentation_maps(segmaps=segmaps)
    aug_images, aug_masks = elastic_det.augment_images(aug_images), elastic_det.augment_segmentation_maps(segmaps=aug_masks)
    aug_images = blur_aug.augment_images(aug_images)
    aug_images = other_aug.augment_images(aug_images)

    # convert seg_maps into numpy arrays with shape (H,W,1)
    aug_masks = [np.expand_dims(m.arr[:, :, 0], axis=2) for m in aug_masks]

    return aug_images, aug_masks


def image_mask_generator_imgaug(img_df, mask_df, subset, batch_size, target_size, data_aug=False):#, pretrained_network=None):
    """
    Generator for images and masks that can perform data augmentation using the imgaug library
    
    Args:
        img_df(DataFrame):
        mask_df(DataFrame):
        batch_size(int):
        target_size(tuple or list):
        data_aug(bool):
    
    Returns:
        tuple contains:
          - x_batch: 
          - y_batch:
    """

    img_df  = img_df.loc[img_df['subset'] == subset, :]
    mask_df = mask_df.loc[mask_df['subset'] == subset, :]

    assert len(img_df) == len(mask_df), "The number of images should be equal with the number of masks."

    list_IDs = np.arange(0, len(img_df))
    random.shuffle(list_IDs)

    while True:

        random.shuffle(list_IDs)

        for start in range(0, len(list_IDs), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(list_IDs))
            batch_IDs = list_IDs[start:end]

            for idx in batch_IDs:

                assert pt.Path(img_df.iloc[idx]['file_path']).stem == \
                       pt.Path(mask_df.iloc[idx]['file_path']).stem, "Image and mask filename don't match."

                img = cv2.imread(img_df.iloc[idx]['file_path'])
                img = cv2.resize(img, (target_size[0], target_size[1]))

                mask = cv2.imread(mask_df.iloc[idx]['file_path'])
                mask = cv2.resize(mask, (target_size[0], target_size[1]))
                mask = (mask * 1 / 255).astype(np.float32)

                if not data_aug:
                    mask = np.expand_dims(mask[:, :, 0], axis=2)

                assert img.dtype != np.float32, "Invalid type: {} for image {}".format(img.dtype, img_files[id])

                x_batch.append(img)
                y_batch.append(mask)

            if data_aug:
                x_batch, y_batch = augmentator(x_batch, y_batch)

            # hash = random.randint(1, 100)
            # cv2.imwrite("aug_img_{}_{}.jpg".format(hash, data_aug), x_batch[0])
            # cv2.imwrite("aug_mask_{}_{}.jpg".format(hash, data_aug), y_batch[0] * 255)

            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32)  # scaling already done in line 166

            yield x_batch, y_batch
