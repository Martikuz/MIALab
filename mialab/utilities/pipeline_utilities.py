"""This module contains utility classes and functions."""
import enum
import os
import typing as t

import numpy as np
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc

# import matplotlib.pyplot as plt

atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2
    # atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a.nii.gz'))  # changed by Martin
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))
    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images have not the same image properties')

    # print('*** load_atlas_images')

    # # Plot variables
    # at1 = sitk.GetArrayFromImage(atlas_t1)
    # at2 = sitk.GetArrayFromImage(atlas_t2)
    #
    # # show image
    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle('atlas')
    # slice = 100
    # axs[0].imshow(at1[slice, :, :])
    # axs[0].title.set_text('atlas t1')
    # axs[1].imshow(at2[slice, :, :])
    # axs[1].title.set_text('atlas t2')
    # plt.show()
    #
    # # histoplot
    # histoplt = np.histogram(at1, bins=10, range=None, normed=None, weights=None, density=None)
    # _ = plt.hist(histoplt, bins='auto')
    # plt.title("atlas Histogram with 'auto' bins")
    # plt.show()
    #
    # # histoplot better
    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle('atlas histograms')
    # data0 = at1[at1 > at1.mean()].flatten()
    # data1 = at2[at2 > at2.mean()].flatten()
    # colors = ['b', 'r']
    # axs[0].hist(data0, 400, color=colors[0])
    # axs[0].title.set_text('atlas_t1')
    # axs[1].hist(data1, 400, color=colors[1])
    # axs[1].title.set_text('atlas_t2')
    # axs[0].grid(True)
    # axs[1].grid(True)
    # plt.show()


class FeatureImageTypes(enum.Enum):
    """Represents the feature image types."""

    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5
    # print('*** FeatureImageTypes')


class FeatureExtractor:
    """Represents a feature extractor."""

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        self.coordinates_feature = kwargs.get('coordinates_feature', False)
        self.intensity_feature = kwargs.get('intensity_feature', False)
        self.gradient_intensity_feature = kwargs.get('gradient_intensity_feature', False)
        # print('*** init FeatureExtractor')

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        # todo: add T2w features
        # warnings.warn('No features from T2-weighted image extracted.')

        if self.coordinates_feature:
            atlas_coordinates = fltr_feat.AtlasCoordinates()
            self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1w])
            # atlas for T2w seems to stay the same...

        if self.intensity_feature:
            self.img.feature_images[FeatureImageTypes.T1w_INTENSITY] = self.img.images[structure.BrainImageTypes.T1w]
            self.img.feature_images[FeatureImageTypes.T2w_INTENSITY] = self.img.images[structure.BrainImageTypes.T2w]

        if self.gradient_intensity_feature:
            # compute gradient magnitude images
            self.img.feature_images[FeatureImageTypes.T1w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1w])
            self.img.feature_images[FeatureImageTypes.T2w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2w])

        self._generate_feature_matrix()
        # print('*** execute FeatureExtractor')

        return self.img

    def _generate_feature_matrix(self):
        """Generates a feature matrix."""

        mask = None
        if self.training:
            # generate a randomized mask where 1 represents voxels used for training
            # the mask needs to be binary, where the value 1 is considered as a voxel which is to be loaded
            # we have following labels:
            # - 0 (background)
            # - 1 (white matter)
            # - 2 (grey matter)
            # - 3 (Hippocampus)
            # - 4 (Amygdala)
            # - 5 (Thalamus)

            # you can exclude background voxels from the training mask generation
            # mask_background = self.img.images[structure.BrainImageTypes.BrainMask]
            # and use background_mask=mask_background in get_mask()

            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                self.img.images[structure.BrainImageTypes.GroundTruth],
                [0, 1, 2, 3, 4, 5],
                [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02])

            # convert the mask to a logical array where value 1 is False and value 0 is True
            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # generate features
        data = np.concatenate(
            [self._image_as_numpy_array(image, mask) for id_, image in self.img.feature_images.items()],
            axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

        # print('*** generate_feature_matrix FeatureExtractor')

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray = None):
        """Gets an image as numpy array where each row is a voxel and each column is a feature.

        Args:
            image (sitk.Image): The image.
            mask (np.ndarray): A mask defining which voxels to return. True is background, False is a masked voxel.

        Returns:
            np.ndarray: An array where each row is a voxel and each column is a feature.
        """

        number_of_components = image.GetNumberOfComponentsPerPixel()  # the number of features for this image
        no_voxels = np.prod(image.GetSize())
        image = sitk.GetArrayFromImage(image)

        if mask is not None:
            no_voxels = np.size(mask) - np.count_nonzero(mask)

            if number_of_components == 1:
                masked_image = np.ma.masked_array(image, mask=mask)
            else:
                # image is a vector image, make a vector mask
                vector_mask = np.expand_dims(mask, axis=3)  # shape is now (z, x, y, 1)
                vector_mask = np.repeat(vector_mask, number_of_components,
                                        axis=3)  # shape is now (z, x, y, number_of_components)
                masked_image = np.ma.masked_array(image, mask=vector_mask)

            image = masked_image[~masked_image.mask]

        # print('*** image_as_numpy_array FeatureExtractor')

        return image.reshape((no_voxels, number_of_components))

    # print('*** FeatureExtractor')


def pre_process(id_: str, paths: dict, **kwargs) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.

    Returns:
        (structure.BrainImage):
    """

    print('-' * 10, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    transform = sitk.ReadTransform(path_to_transform)
    img = structure.BrainImage(id_, path, img, transform)

    # load reference image
    # TODO
    reference_image_dir_path = 'E:/01_Programme/Git/00_MyFolder/MIALab/data/train/100307/'
    reference_t1_path = os.path.join(reference_image_dir_path, 'T1native.nii.gz')
    reference_t2_path = os.path.join(reference_image_dir_path, 'T2native.nii.gz')
    reference_brain_mask_path = os.path.join(reference_image_dir_path, 'Brainmasknative.nii.gz')
    reference_transform_path = 'E:/01_Programme/Git/00_MyFolder/MIALab/data/train/100307/affine.txt'
    reference_img = structure.BrainImage('Reference',
                                         reference_image_dir_path,
                                         {structure.BrainImageTypes.T1w: sitk.ReadImage(reference_t1_path),
                                          structure.BrainImageTypes.T2w: sitk.ReadImage(reference_t2_path),
                                          structure.BrainImageTypes.BrainMask: sitk.ReadImage(reference_brain_mask_path)},
                                          sitk.ReadTransform(reference_transform_path))

    # construct pipeline for brain mask registration
    # we need to perform this before the T1w and T2w pipeline because the registered mask is used for skull-stripping
    pipeline_brain_mask = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_brain_mask.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                                      len(pipeline_brain_mask.filters) - 1)

    # execute pipeline on the brain mask image
    img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
        img.images[structure.BrainImageTypes.BrainMask])

    # construct pipeline for the reference image
    pipeline_ref_brain_mask = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_ref_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_ref_brain_mask.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, reference_img.transformation, True),
                                          len(pipeline_ref_brain_mask.filters) - 1)
    reference_img.images[structure.BrainImageTypes.BrainMask] = pipeline_ref_brain_mask.execute(reference_img.images[structure.BrainImageTypes.BrainMask])

    # construct pipeline for T1w reference image pre-processing
    pipeline_t1_ref = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t1_ref.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1_ref.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, reference_img.transformation),
                                  len(pipeline_t1_ref.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t1_ref.add_filter(fltr_prep.SkullStripping())
        pipeline_t1_ref.set_param(fltr_prep.SkullStrippingParameters(reference_img.images[structure.BrainImageTypes.BrainMask]),
                                  len(pipeline_t1_ref.filters) - 1)
    reference_img.images[structure.BrainImageTypes.T1w] = pipeline_t1_ref.execute(reference_img.images[structure.BrainImageTypes.T1w])
    # normalization_parameters_t1 = fltr_prep.NormalizationParameters(reference_img.images[structure.BrainImageTypes.T1w])
    # normalization_parameters_t1 = fltr_prep.NormalizationParameters(reference_img.images[structure.BrainImageTypes.T1w], img.id_)

    # construct pipeline for T1w image pre-processing
    pipeline_t1 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t1.add_filter(fltr_prep.SkullStripping())
        pipeline_t1.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t1.filters) - 1)

    if kwargs.get('normalization_pre', False):
        filter_params = fltr_prep.NormalizationParameters(reference_img.images[structure.BrainImageTypes.T1w], img.id_, 'T1w')
        if kwargs.get('white_stripes', False):
            pipeline_t1.add_filter(fltr_prep.WhiteStripesT1())
            pipeline_t1.set_param(filter_params, len(pipeline_t1.filters) - 1)
        elif kwargs.get('no_normalization', False):
            pipeline_t1.add_filter(fltr_prep.NoNormalization())
            pipeline_t1.set_param(filter_params, len(pipeline_t1.filters) - 1)
        elif kwargs.get('histogram_matching_1', False):
            pipeline_t1.add_filter(fltr_prep.HistogramMatching())
            pipeline_t1.set_param(filter_params, len(pipeline_t1.filters) - 1)
        elif kwargs.get('histogram_matching_2', False):
            pipeline_t1.add_filter(fltr_prep.ImageNormalization())
            pipeline_t1.set_param(filter_params, len(pipeline_t1.filters) - 1)
        elif kwargs.get('z_score', False):
            pipeline_t1.add_filter(fltr_prep.ZScore())
            pipeline_t1.set_param(filter_params, len(pipeline_t1.filters) - 1)
        # pipeline_t1.set_param(normalization_parameters_t1, len(pipeline_t1.filters) - 1)

    # execute pipeline on the T1w image
    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1w])

    # construct pipeline for T2w reference image pre-processing
    # TODO Remove
    pipeline_t2_ref = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t2_ref.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2_ref.set_param(fltr_prep.ImageRegistrationParameters(atlas_t2, reference_img.transformation),
                                  len(pipeline_t2_ref.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t2_ref.add_filter(fltr_prep.SkullStripping())
        pipeline_t2_ref.set_param(fltr_prep.SkullStrippingParameters(reference_img.images[structure.BrainImageTypes.BrainMask]),
                                  len(pipeline_t2_ref.filters) - 1)
    reference_img.images[structure.BrainImageTypes.T2w] = pipeline_t2_ref.execute(reference_img.images[structure.BrainImageTypes.T2w])
    # normalization_parameters_t2 = fltr_prep.NormalizationParameters(reference_img.images[structure.BrainImageTypes.T2w])
    # normalization_parameters_t2 = fltr_prep.NormalizationParameters(reference_img.images[structure.BrainImageTypes.T2w], img.id_)

    # construct pipeline for T2w image pre-processing
    pipeline_t2 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2.set_param(fltr_prep.ImageRegistrationParameters(atlas_t2, img.transformation),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t2.add_filter(fltr_prep.SkullStripping())
        pipeline_t2.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t2.filters) - 1)

    if kwargs.get('normalization_pre', False):
        filter_params = fltr_prep.NormalizationParameters(reference_img.images[structure.BrainImageTypes.T2w], img.id_, 'T2w')
        if kwargs.get('white_stripes', False):
            pipeline_t2.add_filter(fltr_prep.WhiteStripesT2())
            pipeline_t2.set_param(filter_params, len(pipeline_t2.filters) - 1)
        elif kwargs.get('no_normalization', False):
            pipeline_t2.add_filter(fltr_prep.NoNormalization())
            pipeline_t2.set_param(filter_params, len(pipeline_t2.filters) - 1)
        elif kwargs.get('histogram_matching_1', False):
            pipeline_t2.add_filter(fltr_prep.HistogramMatching())
            pipeline_t2.set_param(filter_params, len(pipeline_t2.filters) - 1)
        elif kwargs.get('histogram_matching_2', False):
            pipeline_t2.add_filter(fltr_prep.ImageNormalization())
            pipeline_t2.set_param(filter_params, len(pipeline_t2.filters) - 1)
        elif kwargs.get('z_score', False):
            pipeline_t2.add_filter(fltr_prep.ZScore())
            pipeline_t2.set_param(filter_params, len(pipeline_t2.filters) - 1)
        # pipeline_t2.set_param(normalization_parameters_t2, len(pipeline_t2.filters) - 1)

    # execute pipeline on the T2w image
    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2w])

    # construct pipeline for ground truth image pre-processing
    pipeline_gt = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_gt.add_filter(fltr_prep.ImageRegistration())
        pipeline_gt.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                              len(pipeline_gt.filters) - 1)

    # execute pipeline on the ground truth image
    img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
        img.images[structure.BrainImageTypes.GroundTruth])

    # update image properties to atlas image properties after registration
    img.image_properties = conversion.ImageProperties(img.images[structure.BrainImageTypes.T1w])

    # extract the features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()

    img.feature_images = {}  # we free up memory because we only need the img.feature_matrix
    # for training of the classifier

    # print('*** pre_process')

    return img


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation.

    Args:
        img (structure.BrainImage): The image.
        segmentation (sitk.Image): The segmentation (label image).
        probability (sitk.Image): The probabilities images (a vector image).

    Returns:
        sitk.Image: The post-processed image.
    """

    print('-' * 10, 'Post-processing', img.id_)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())
    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1w],
                                                     img.images[structure.BrainImageTypes.T2w],
                                                     probability), len(pipeline.filters) - 1)

    # print('*** post_process')

    return pipeline.execute(segmentation)


def init_evaluator() -> eval_.Evaluator:
    """Initializes an evaluator.

    Returns:
        eval.Evaluator: An evaluator.
    """

    # initialize metrics
    # todo: add hausdorff distance, 95th percentile (see metric.HausdorffDistance)
    # mk: added code that renames HDRFDST to HDRFDST95
    # mk: added code that add the element VolumeSimilarity to the list with:
    #     metric.VolumeSimilarity() according the pymia documentation on:
    #     https://pymia.readthedocs.io/en/latest/examples.evaluation.basic.html
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'),
               metric.VolumeSimilarity()]
    # warnings.warn('Initialized evaluation with the Dice coefficient. Do you know other suitable metrics?')

    # define the labels to evaluate
    labels = {1: 'WhiteMatter',
              2: 'GreyMatter',
              3: 'Hippocampus',
              4: 'Amygdala',
              5: 'Thalamus'
              }

    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    # print('*** init_evaluator')

    return evaluator


def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict = None, multi_process: bool = True) -> t.List[structure.BrainImage]:
    """Loads and pre-processes a batch of images.

    The pre-processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        data_batch (Dict[structure.BrainImageTypes, structure.BrainImage]): Batch of images to be processed.
        pre_process_params (dict): Pre-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[structure.BrainImage]: A list of images.
    """
    if pre_process_params is None:
        pre_process_params = {}

    params_list = list(data_batch.items())
    if multi_process:
        images = mproc.MultiProcessor.run(pre_process, params_list, pre_process_params, mproc.PreProcessingPickleHelper)
    else:
        images = [pre_process(id_, path, **pre_process_params) for id_, path in params_list]

    # print('*** pre_process_batch')

    return images


def post_process_batch(brain_images: t.List[structure.BrainImage], segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image], post_process_params: dict = None,
                       multi_process: bool = True) -> t.List[sitk.Image]:
    """ Post-processes a batch of images.

    Args:
        brain_images (List[structure.BrainImageTypes]): Original images that were used for the prediction.
        segmentations (List[sitk.Image]): The predicted segmentation.
        probabilities (List[sitk.Image]): The prediction probabilities.
        post_process_params (dict): Post-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[sitk.Image]: List of post-processed images
    """
    if post_process_params is None:
        post_process_params = {}

    param_list = zip(brain_images, segmentations, probabilities)
    if multi_process:
        pp_images = mproc.MultiProcessor.run(post_process, param_list, post_process_params,
                                             mproc.PostProcessingPickleHelper)
    else:
        pp_images = [post_process(img, seg, prob, **post_process_params) for img, seg, prob in param_list]

    # print('*** post_process_batch')

    return pp_images
