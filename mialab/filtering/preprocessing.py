"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import matplotlib.pyplot as plt
# import for histogram matching
import numpy as np
import matplotlib.pyplot as plt
from pymia.filtering.filter import FilterParams
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import intensity_normalization as im
from typing import Tuple
import numpy as np
import statsmodels.api as sm
from scipy.signal import argrelmax


# *******************************************************************************************************************
# ***************** BEGIN - Normalization ***************************************************************************
# Adapted: pipeline_utilites.py from line 205
# e.g.     if kwargs.get('normalization_pre', False):
#         if kwargs.get('white_stripes', False):
#             pipeline_t1.add_filter(fltr_prep.WhiteStripesT1())
#         elif kwargs.get('no_normalization', False):
#             pipeline_t1.add_filter(fltr_prep.NoNormalization())
#           ....
#
#Adapted: Main.py
#    pre_process_params = {'skullstrip_pre': True,
#                           'normalization_pre': True,
#                           'registration_pre': True,
#                           'coordinates_feature': True,
#                           'intensity_feature': True,
#                           'gradient_intensity_feature': True,
#                           'white_stripes': False,
#                           'no_normalization': False,
#                           'histogram_matching_1': False,
#                           'histogram_matching_2': False,
#                           'z_score': True}
#
class WhiteStripesBase(pymia_fltr.Filter):
    """
    intensity_normalization.util.histogram_tools
    Process the histograms of MR (brain) images
    Author: Jacob Reinhold (jcreinhold@gmail.com)
    Created on: Jun 01, 2021
    Adapted by Michael and Martin
    """
    def smooth_histogram(self,data) -> Tuple[np.ndarray, np.ndarray]:
        """Use kernel density estimate to get smooth histogram
        Args:
            data: np.ndarray of image data
        Returns:
            grid: domain of the pdf
            pdf: kernel density estimate of the pdf of data
        """
        data_vec = data.flatten().astype(np.float64)
        bandwidth = data_vec.max() / 80  # noqa
        kde = sm.nonparametric.KDEUnivariate(data_vec)
        kde.fit(kernel="gau", bw=bandwidth, gridsize=80, fft=True)
        pdf = 100.0 * kde.density
        grid = kde.support
        return grid, pdf



    def TestKDE(self):
        nobs = 300
        np.random.seed(1234)  # Seed random generator
        dens = sm.nonparametric.KDEUnivariate(np.random.normal(size=nobs))
        dens.fit()
        plt.plot(dens.cdf)
        plt.show()

    def get_last_tissue_mode(self,data):
        """Mode of the highest-intensity tissue class
        Args:
            data: image data
            remove_tail: remove tail from histogram
            tail_percentage: if remove_tail, use the
                histogram below this percentage
        Returns:
            last_tissue_mode: mode of the highest-intensity tissue class
        """
        remove_tail= True
        tail_percentage= 96.0
        if remove_tail:
            threshold = np.percentile(data, tail_percentage)
            valid_mask = data >= threshold
            data = data[valid_mask==0]
        grid, pdf = self.smooth_histogram(data)
        maxima = argrelmax(pdf)[0]
        last_tissue_mode: float = grid[maxima[-1]]
        #print("Argmax",np.argmax(pdf),"Applied to grid",grid[maxima[-1]] )
        #print("T1:",last_tissue_mode)



        return last_tissue_mode

    def get_largest_tissue_mode(self, data) -> float:
            """Mode of the largest tissue class
            Args:
                data: image data
            Returns:
                largest_tissue_mode (float): intensity of the mode
            """
            #threshold = np.percentile(data, 96)
            #valid_mask = data >= threshold
            #data = data[valid_mask == 0]
            """"""
            remove_tail = True
            tail_percentage = 96.0
            if remove_tail:
                threshold = np.percentile(data, tail_percentage)
                valid_mask = data >= threshold
                data = data[valid_mask == 0]
            grid, pdf = self.smooth_histogram(data)
            #largest_tissue_mode = grid[np.argmax(pdf)]
            #maxima = argrelmax(pdf)[0]
            maxima = argrelmax(pdf)[0]
            largest_tissue_mode = grid[maxima[1]]
            #print("Min imagae:",np.min(data))
            #print(np.argmax(pdf))
            #print(grid[np.argmax(pdf)])
            #plt.title("Histogram-Largest Tissue Mode")
            #plt.xlabel("Intensity")
            #plt.plot(grid,pdf)
            #plt.show()
            return largest_tissue_mode

    def get_largest_tissue_modev2(self, data) -> float:
        """Mode of the largest tissue class
        Args:
            data: image data
        Returns:
            largest_tissue_mode (float): intensity of the mode
        """
        # threshold = np.percentile(data, 96)
        # valid_mask = data >= threshold
        # data = data[valid_mask == 0]
        grid, pdf = self.smooth_histogram(data)

        largest_tissue_mode = grid[np.argmax(pdf)]
        # largest_tissue_mode = grid[np.argmax(pdf)]
        #print("Min imagae:", np.min(data))
        #print(np.argmax(pdf))
        #print(grid[np.argmax(pdf)])
        #plt.title("Histogram-Largest Tissue Mode")
        #plt.plot(pdf)
        #plt.show()
        return np.argmax(pdf)

    def PlotBeforeAfter(self,image_Array_original,image_array):
        fig, axs = plt.subplots(1,2)
        fig.suptitle('Whitestripe normalization')
        slice=100
        axs[0].imshow(image_Array_original[slice, :, :])
        axs[0].title.set_text('Raw image')
        axs[1].imshow(image_array[slice, :, :])
        axs[1].title.set_text('Normalized image')
        plt.show()

    def PlotHisto(self, image_Array_original, image_array):
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Whitestripe normalization histogram')
        data0 = image_Array_original[image_Array_original > image_Array_original.mean()].flatten()
        data1 = image_array[image_array > image_array.mean()].flatten()
        colors = ['b', 'r']
        axs[0].hist(data0, 400,color=colors[0])
        axs[1].hist(data1, 400,color=colors[1])
        #axs[2].hist([data0,data1], 400,color=colors, alpha=0.5)
        axs[0].grid(True)
        axs[1].grid(True)
        plt.show()




class WhiteStripesT1(WhiteStripesBase):

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print("T1")
        #super().TestKDE
        #preprocessed_image = super().execute(image, params)
        #Last-->T1
        width = 0.05
        image_array = sitk.GetArrayFromImage(image)
        image_Array_original=image_array
        wm_mode = super().get_last_tissue_mode(image_array)
        test1=(image_array < wm_mode)
        test2=np.mean(image_array < wm_mode)
        wm_mode_quantile: float = np.mean(image_array < wm_mode).item()
        lower_bound = max(wm_mode_quantile - width, 0.0)
        upper_bound = min(wm_mode_quantile + width, 1.0)
        ws_l, ws_u = np.quantile(image_array, (lower_bound, upper_bound))
        whitestripe = (image_array > ws_l) & (image_array < ws_u)
        mean = np.mean(image_array[whitestripe])
        std = np.std(image_array[whitestripe])
        image_array = (image_array - mean) / std
        img_out = sitk.GetImageFromArray(image_array)
        img_out.CopyInformation(image)
        return img_out

class WhiteStripesT2(WhiteStripesBase):
    """Represents WhiteStripesT2 normalization"""
    #Largest-->T2
    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print("WhiteStripesT2-started")
        width=0.05
        image_array = sitk.GetArrayFromImage(image)
        mtest3=np.min(image_array)
        wm_mode = super().get_largest_tissue_mode(image_array)
        if wm_mode>0:
            wm_mode_quantile: float = np.mean(image_array < wm_mode).item()
            lower_bound = max(wm_mode_quantile - width, 0.0)
            upper_bound = min(wm_mode_quantile + width, 1.0)
        else:
            lower_bound=0
            upper_bound=0.1
       # print("Bounds are:",lower_bound,upper_bound)
        ws_l, ws_u = np.quantile(image_array, (lower_bound, upper_bound))
        #print("For the quantile:",lower_bound," to ",upper_bound,"the values are: ",ws_l,"to ",ws_u)

        whitestripe_ind = (image_array > ws_l) & (image_array < ws_u)
        mean = np.mean(image_array[whitestripe_ind])
        std = np.std(image_array[whitestripe_ind])
        image_array = (image_array - mean) / std
        img_out = sitk.GetImageFromArray(image_array)
        img_out.CopyInformation(image)
        return img_out

class NoNormalization(pymia_fltr.Filter):
    """Represents no norm normalization"""
    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        return image

class ZScore(pymia_fltr.Filter):
    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        """Represents ZScore"""
        img_arr = sitk.GetArrayFromImage(image)
        mean = img_arr.mean()
        std = img_arr.std()
        img_arr = (img_arr - mean) / std
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)
        return img_out

class HistogramMatching1(pymia_fltr.Filter):

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print('Normalization: Histogram Matching Method 2')
        img_arr = sitk.GetArrayFromImage(image)
        # read in reference image --> ref_arr
        # referenceH = data.ref_arr()
        # imageH = data.img_arr()
        # matched = match_histograms(imageH, referenceH, channel_axis=-1)
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8,3), sharex=True, sharey=True)
        # for aa in (ax1, ax2, ax3):
        #     aa.set_axis_off()
        # ax1.imshow(imageH[98,:,:])
        # ax1.set_title('source')
        # ax2.imshow(referenceH[98,:,:])
        # ax2.set_title('reference')
        # ax3.imshow(matched[98,:,:])
        # ax3.set_title('Matched')
        # plt.tight_layout()
        # plt.show()
        # img_arr = matched
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)
        return img_out

class HistogramMatching2(pymia_fltr.Filter):

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print('Normalization: Histogram Matching Method 1')
        img_arr = sitk.GetArrayFromImage(image)
        # pdfr_arr = copy.deepcopy(img_arr)
        # n_val = 197*233*189                    # assuming all images are of the same size!!!!
        # for xit in range(198):
        #     for yit in range(234):
        #         for zit in range(190):
        #         r_val = img_arr[xit,yit,zit]
        #         nj_val = None #histogram value of r_val[xit,yit,zit]
        #         pdfr_arr[xit,yit,zit] = nj_val/n_val
        sitk.HistogramMatching()

        # img_arr = pdfr_arr

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)
        return img_out

#----------------------------------------------------------------------
#----------------------------------------------------------------------
    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)

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


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

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

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

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
        #warnings.warn('No registration implemented. Returning unregistered image')

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        if is_ground_truth:
            # apply transformation to ground truth and brain mask using nearest neighbor interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkNearestNeighbor, 0,
                                  image.GetPixelIDValue())
        else:
            # apply transformation to T1w and T2w images using linear interpolation
            image = sitk.Resample(image, atlas, transform, sitk.sitkLinear, 0.0,
                                  image.GetPixelIDValue())

        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
