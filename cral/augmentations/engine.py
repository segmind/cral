from abc import ABC, abstractmethod

import numpy as np
from albumentations import BboxParams, Compose


class Base(ABC):

    def __init__(self, aug):
        self.aug = aug

    @abstractmethod
    def apply():
        pass


class Classification(Base):
    """Class to apply augentations to Classification datasets.

    Attributes:
        aug (list(augmentations)): List of Albumentations Augmentations
        augmentation (function): function to apply list of agumentations on
                images
        p (float): probability of applying the transformation
    """

    def __init__(self, aug, p=0.5):
        """

        Args:
            aug (list(augmentations)): List of Albumentations Augmentations
            p (float, optional): probability of applying the transformation
        """
        self.aug = aug
        self.p = p
        self.augmentation = self.get_aug()

    def get_aug(self):
        """returns a function that applies the list of transformations.

        Returns:
            function: function to apply list of agumentations on images
        """
        return Compose(transforms=self.aug, p=self.p)

    def apply(self, image):
        """Applies the augmentation on image.

        Args:
            image (nparray): input image

        Returns:
            image: Augmented image
        """
        return self.augmentation(image=image)['image']


class ObjectDetection(Base):
    """Class to apply augentations to ObjectDetection datasets.

    Attributes:
        annotation_format (str): can be one of 'coco','pascal'
        aug (list(augmentations)): List of Albumentations Augmentations
        p (float): probability of applying the transformation
    """

    def __init__(self, aug, annotation_format, p=0.5):
        """

        Args:
            aug (list(augmentations)): List of Albumentations Augmentations
            annotation_format (str): can be one of 'coco','pascal'
            p (float , optional): probability of applying the transformation
        """
        self.aug = aug
        self.p = p

        assert annotation_format in ('coco', 'pascal_voc', 'yolo')
        self.annotation_format = annotation_format
        self.augmentation = self.get_aug('category_id')

    def get_aug(self, label_field, min_area=0.0, min_visibility=0.0):
        """returns a function that applies the list of transformations.

        Args:
            label_field (str): The feild in the dictionary that contains the
                    name labels
            min_area (float, optional): minimum area of bbox that is considered
            min_visibility (float, optional): minimum area of bbox to be
                     visible

        Returns:
            function: function to apply list of agumentations on images and
                     bboxes
        """
        return Compose(
            self.aug,
            bbox_params=BboxParams(
                format=self.annotation_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=[label_field]),
            p=self.p)

    def apply(self, image, bboxes, labels):
        """applies augentations to ObjectDetection datasets.

        Args:
            image (list(images)): list of images
            bboxes (list(bboxes)): list of boxxes
            labels (list(labels)): list of labels
            annotation_format (str): can be one of 'coco','pascal_voc'

        Returns:
            list(images),list(bbox_,list(labels):return list of images bboxes
                    and labels
        """
        # augmentation=self.get_aug(annotation_format,'category_id')
        annotation = {'image': image, 'bboxes': bboxes, 'category_id': labels}
        augmented_annotation = self.augmentation(**annotation)
        return augmented_annotation['image'], np.array(
            augmented_annotation['bboxes']), np.array(
                augmented_annotation['category_id'])


class Segmentation(Base):
    """Class to apply augentations to ObjectDetection datasets.

    Attributes:
        aug (list(augmentations)): List of Albumentations Augmentations
        augmentation (function): function to apply list of agumentations
                on images
        p (float): probability of applying the transformation
    """

    def __init__(self, aug, p=0.5):
        """

        Args:
            aug (list(augmentations)): List of Albumentations Augmentations
            p (float, optional): probability of applying the transformation
        """
        self.aug = aug
        self.p = p
        self.augmentation = self.get_aug()

    def get_aug(self):
        """returns a function that applies the list of transformations
        Returns:
            function: function to apply list of agumentations on images and
                 bboxes
        """
        return Compose(self.aug, p=self.p)

    def apply(self, image, mask):
        """applies augentations to Segmentation datasets.

        Args:
            image (list(images)): list of images
            mask (list(masks)): list of masks

        Returns:
            list(image),list(mask): list of augmented images and masks
        """
        augmented = self.augmentation(image=image, mask=mask)
        return augmented['image'], augmented['mask']
