"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

# import for histogram matching
import numpy as np
import matplotlib.pyplot as plt
from pymia.filtering.filter import FilterParams
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk

# import for histogram matching
from typing import Tuple
import statsmodels.api as sm
from scipy.signal import argrelmax


# *******************************************************************************************************************
# ***************** BEGIN - Normalization ***************************************************************************


class NormalizationParameters(pymia_fltr.FilterParams):

    def __init__(self,
                 reference_image: sitk.Image,
                 image_id: str = '',
                 label='',
                 ) -> None:
        super().__init__()
        self.reference_image = reference_image
        self.image_id = image_id
        self.label = label


class ImageNormalization(pymia_fltr.Filter):
    """Represents the base class for the Histogram Matching Method. Was originaly a Normalisation Method Filter"""

    def __init__(self, normalization='HM2', mask=None):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()
        self.normalization = normalization  # Possible initialisation input: 'None', 'HM2'
        self.mask = mask
        self.ref_img = None
        # print('***** init ImageNormalization')

    # ******************** FIGURE PLOT Image ***************************************************************
    @staticmethod
    def plot_image(image_array, ref_array, matched_array, id_, label) -> None:
        # TODO code for plot 1
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
        fig.suptitle(f'subject_{id_}_{label}')
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()
        ax1.imshow(image_array[98, :, :])
        ax1.set_title('source')
        ax2.imshow(ref_array[98, :, :])
        ax2.set_title('reference')
        ax3.imshow(matched_array[98, :, :])
        ax3.set_title('Matched')
        plt.tight_layout()
        output_path = f'C:/temp/result_pic_{id_}_{label}.png'
        fig.savefig(output_path)
        plt.close()

    # ******************** FIGURE PLOT Histogram ***********************************************************
    @staticmethod
    def plot_histogram(data0, data1, data2, id_, label) -> None:
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f'histogram subject_{id_}_{label}')
        colors = ['b', 'r', 'g']
        y_limits = (0, 5e-5)
        axs[0].hist(data0, 400, color=colors[0], density=True)
        axs[0].title.set_text('image_array')
        axs[0].set_ylim(*y_limits)
        axs[1].hist(data1, 400, color=colors[1], density=True)
        axs[1].title.set_text('ref_array')
        axs[1].set_ylim(*y_limits)
        axs[2].hist(data2, 400, color=colors[2], density=True)
        axs[2].title.set_text('matched_array')
        axs[2].set_ylim(*y_limits)
        output_path = f'C:/temp/result_histo_{id_}_{label}.png'
        fig.savefig(output_path)
        plt.close()

    def execute(self, image: sitk.Image, params: NormalizationParameters = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        if self.normalization == 'None':
            print('WARNING: no normalization method applied!')

        else:
            print('WARNING: unknown normalization method initialisation input!')

        if isinstance(img_arr, sitk.Image):
            img_arr.CopyInformation(image)
            return img_arr

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        # print('***** execute ImageNormalization')

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """

        # print('***** str ImageNormalization')

        return 'ImageNormalization:\n' \
            .format(self=self)

    # print('***** ImageNormalization')


class HistogramMatching(ImageNormalization):

    def execute(self, image: sitk.Image, params: NormalizationParameters = None) -> sitk.Image:
        print('Normalization: Histogram Matching Method')

        img_arr = sitk.GetArrayFromImage(image)

        ref_params = params.reference_image

        # matched = sitk.HistogramMatching(image, ref_params)

        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(1000)
        matcher.ThresholdAtMeanIntensityOn()
        matched = matcher.Execute(image, params.reference_image)

        image_array = sitk.GetArrayFromImage(image)
        ref_array = sitk.GetArrayFromImage(ref_params)
        matched_array = sitk.GetArrayFromImage(matched)

        img_arr = matched
        data0 = image_array.flatten()
        data1 = ref_array.flatten()
        data2 = matched_array.flatten()

        # print(params.image_id)
        self.plot_image(image_array, ref_array, matched_array, params.image_id, params.label)
        self.plot_histogram(data0, data1, data2, params.image_id, params.label)

        if isinstance(img_arr, sitk.Image):
            img_arr.CopyInformation(image)
            return img_arr

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        # print('***** execute ImageNormalization')

        return img_out


class NoNormalization(ImageNormalization):
    """Represents no norm normalization"""

    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print('No Normalization applied')
        return image


class ZScore(ImageNormalization):
    """Represents ZScore"""

    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print('Normalization: Z-Score Method')
        img_arr = sitk.GetArrayFromImage(image)
        mean = img_arr.mean()
        std = img_arr.std()
        img_arr = (img_arr - mean) / std
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)
        return img_out


class WhiteStripesBase(ImageNormalization):
    """
    intensity_normalization.util.histogram_tools
    Process the histograms of MR (brain) images
    Author: Jacob Reinhold (jcreinhold@gmail.com)
    Created on: Jun 01, 2021
    Adapted by Michael and Martin
    """

    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        pass

    def smooth_histogram(self, data) -> Tuple[np.ndarray, np.ndarray]:
        # Smoothing the histogram
        data_vec = data.flatten().astype(np.float64)
        bandwidth = data_vec.max() / 80
        kde = sm.nonparametric.KDEUnivariate(data_vec)
        kde.fit(kernel="gau", bw=bandwidth, gridsize=80, fft=True)
        pdf = 100.0 * kde.density
        grid = kde.support
        return grid, pdf

    def get_last_tissue_mode(self, data):
        # Return the last peak of the histogram
        threshold = np.percentile(data, 99.0)
        valid_mask = data >= threshold
        data = data[valid_mask == 0]
        grid, pdf = self.smooth_histogram(data)
        maxima = argrelmax(pdf)[0]
        last_tissue_mode: float = grid[maxima[-1]]

        # For plotting
        if False:
            fig, ax = plt.subplots()
            ax.hist(data, 100, density=True, label="Original Histogram", alpha=0.5)
            ax.plot(grid, pdf / 100, 'k', linewidth=2, label="KDE of Smoothed Histogram", alpha=0.6)
            ax.set_xlim(0)
            ax.set_ylim(0, 0.00006)
            fig.suptitle(f'T1 Histogram')
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")
            ax.legend()
            rcParams.update({'figure.autolayout': True})
            plt.show()
        return last_tissue_mode

    def get_largest_tissue_mode(self, data) -> float:
        # Return the biggest peak
        threshold = np.percentile(data, 96.0)
        valid_mask = data >= threshold
        data = data[valid_mask == 0]
        grid, pdf = self.smooth_histogram(data)
        maxima = argrelmax(pdf)[0]
        largest_tissue_mode = grid[maxima[1]]
        # For plotting
        if False:
            fig, ax = plt.subplots()
            ax.hist(data, 100, density=True, label="Original Histogram", alpha=0.5)
            ax.plot(grid, pdf / 100, 'k', linewidth=2, label="KDE of Smoothed Histogram", alpha=0.6)
            ax.set_xlim(0)
            ax.set_ylim(0, 0.00006)
            fig.suptitle(f'T2 Histogram')
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")
            ax.legend()
            rcParams.update({'figure.autolayout': True})
            plt.show()
        return largest_tissue_mode

    def PlotHisto(self, image_Array_original, image_array):
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Whitestripe normalization histogram')

        colors = ['b', 'r']

        axs[0].hist(image_Array_original.flatten(), 400, color=colors[0], alpha=0.5, density=True)
        axs[1].hist(image_array.flatten(), 400, color=colors[1], alpha=0.5, density=True)
        axs[0].set_title("Original")
        axs[1].set_title("White-Stripe Normalized")
        axs[0].set_ylim(0, 0.00004)
        axs[1].set_ylim(0, 0.04)
        axs[0].set_xlim(-1, 25000)
        axs[1].set_xlim(-17, 7.5)

        fig.suptitle("T1 Histogram from subject: 100307")
        axs[0].set_xlabel("Intensity")
        axs[0].set_ylabel("Frequency")
        axs[1].set_xlabel("Intensity")
        axs[1].set_ylabel("Frequency")

        # Nametosafe=fig.subtitle+super().img.id_
        # plt.savefig("test_rasterization.pdf", dpi=150)
        # axs[2].hist([data0,data1], 400,color=colors, alpha=0.5)
        plt.show()


class WhiteStripesT1(WhiteStripesBase):
    """Represents WhiteStripesT1 normalization"""

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print("Normalization: WhiteStripesT1 Method")
        # Last-->T1
        width = 0.06
        image_array = sitk.GetArrayFromImage(image)
        image_Array_original = image_array
        wm_mode = super().get_last_tissue_mode(image_array)
        wm_mode_quantile: float = np.mean(image_array < wm_mode).item()
        print("wm_mode_quantile:", wm_mode_quantile)
        lower_bound = max(wm_mode_quantile - width, 0.0)
        upper_bound = min(wm_mode_quantile + width, 1.0)
        ws_l, ws_u = np.quantile(image_array, (lower_bound, upper_bound))
        print("The lower bound is:", ws_l, ",the upper bound:", ws_u)
        whitestripe = (image_array > ws_l) & (image_array < ws_u)
        mean = np.mean(image_array[whitestripe])
        std = np.std(image_array[whitestripe])
        print("mean,std:", mean, std)
        image_array = (image_array - mean) / std

        img_out = sitk.GetImageFromArray(image_array)
        img_out.CopyInformation(image)
        # super().PlotHisto(image_Array_original,image_array)	#For plotting before and after
        return img_out


class WhiteStripesT2(WhiteStripesBase):
    """Represents WhiteStripesT2 normalization"""

    # Largest-->T2
    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print("Normalization: WhiteStripesT2 Method")
        width = 0.06
        image_array = sitk.GetArrayFromImage(image)
        mtest3 = np.min(image_array)
        wm_mode = super().get_largest_tissue_mode(image_array)
        if wm_mode > 0:
            wm_mode_quantile: float = np.mean(image_array < wm_mode).item()
            lower_bound = max(wm_mode_quantile - width, 0.0)
            upper_bound = min(wm_mode_quantile + width, 1.0)
        else:
            lower_bound = 0
            upper_bound = 0.1
        # print("Bounds are:",lower_bound,upper_bound)
        ws_l, ws_u = np.quantile(image_array, (lower_bound, upper_bound))
        # print("For the quantile:",lower_bound," to ",upper_bound,"the values are: ",ws_l,"to ",ws_u)

        whitestripe_ind = (image_array > ws_l) & (image_array < ws_u)
        mean = np.mean(image_array[whitestripe_ind])
        std = np.std(image_array[whitestripe_ind])
        image_array_ws = (image_array - mean) / std

        # super().PlotHisto(image_array,image_array_ws,mean,std)

        img_out = sitk.GetImageFromArray(image_array_ws)
        img_out.CopyInformation(image)
        return img_out


# ***************** END - Normalization ***************************************************************************
# *****************************************************************************************************************

class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask

        # print('***** init SkullStrippingParameters')

    # print('***** SkullStrippingParameters')


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

        # print('***** init SkullStripping')

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        mask = params.img_mask  # the brain mask

        # todo: remove the skull from the image by using the brain mask
        # warnings.warn('No skull-stripping implemented. Returning unprocessed image.')
        image = sitk.Mask(image, mask)

        # print('***** execute SkullStripping')

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        # print('***** str SkullStripping')

        return 'SkullStripping:\n' \
            .format(self=self)

    # print('***** SkullStripping')


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    # also changes in pipeline_utilities for this
    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform,
                 is_ground_truth: bool = False):  # **** changed here <<----------------
        # def __init__(self, atlas_t1: sitk.Image, atlas_t2: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas  # ************* changed here <<----------------------------------------------------------
        # self.atlas_t1 = atlas_t1
        # self.atlas_t2 = atlas_t2
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth

        # print('***** init ImageRegistrationParameters')

    # print('***** ImageRegistrationParameters')


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

        # print('***** init ImageRegistration')

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        # todo: replace this filter by a registration. Registration can be costly, therefore, we provide you the
        # transformation, which you only need to apply to the image!
        # warnings.warn('No registration implemented. Returning unregistered image')

        atlas = params.atlas  # ************* changed here <<----------------------
        # atlas_t1 = params.atlas_t1
        # atlas_t2 = params.atlas_t2
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        if is_ground_truth:
            # apply transformation to ground truth and brain mask using nearest neighbor interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkNearestNeighbor, 0,
                                  image.GetPixelIDValue())  # ************* changed here <<----------------------
            # image = sitk.Resample(image, atlas_t1, atlas_t2, transform, sitk.sitkNearestNeighbor, 0,
            #                       image.GetPixelIDValue())
        else:
            # apply transformation to T1w and T2w images using linear interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkLinear, 0.0,
                                  image.GetPixelIDValue())  # ************* changed here <<----------------------
            # image = sitk.Resample(image, atlas_t1, atlas_t2, transform, sitk.sitkLinear, 0.0,
            #                       image.GetPixelIDValue())

        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)

        # print('***** execute ImageRegistration')

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """

        # print('***** str ImageRegistration')

        return 'ImageRegistration:\n' \
            .format(self=self)

    # print('***** ImageRegistration')
