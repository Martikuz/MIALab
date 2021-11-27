"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk

# import for histogram matching
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

#Martin:
#import intensity_normalization as im
import matplotlib.pyplot as plt

# *******************************************************************************************************************
# ***************** BEGIN - Normalization ***************************************************************************


class NormalizationParameters(pymia_fltr.FilterParams):

    def __init__(self,
                 reference_image: sitk.Image
                 ) -> None:
        super().__init__()
        self.reference_image = reference_image


class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self, normalization='HM2', mask=None):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()
        self.normalization = normalization  # Possible initialisation input: 'None', 'Z', 'HM1', 'HM2'
        self.mask = mask
        self.ref_img = None
        print('***** init ImageNormalization')

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
            #img_out = img_arr # why not needed? and why does it lead to an error?

        elif self.normalization == 'Z':
            print('Normalization: Z-Score')
            # todo: normalize the image using numpy
            # warnings.warn('No normalization implemented. Returning unprocessed image.')
            mean = img_arr.mean()
            std = img_arr.std()
            img_arr = (img_arr - mean) / std

            #img_out = sitk.GetImageFromArray(img_arr)
            #img_out.CopyInformation(image)

        elif self.normalization == 'HM1':
            print('Normalization: Histogram Matching Method 1')
            print('shape: ')
            print(img_arr.shape)
            print('size: ')
            print(img_arr.size)

            # CODE TESTING
            # rng = np.random.RandomState(10)
            # a = np.hstack((rng.normal(size=1000),
            #                rng.normal(loc=5, scale=2, size=1000)))
            # _ = plt.hist(a, bins='auto')
            #
            # mu, sigma = 0, 0.1  # mean and standard deviation
            # s = np.random.normal(mu, sigma, 1000)
            # abs(mu - np.mean(s))
            # abs(sigma - np.std(s, ddof=1))
            # count, bins, ignored = plt.hist(s, 30, density=True)
            # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
            #          np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
            #          linewidth=2, color='r')
            #
            # plt.title("Histogram with 'auto' bins")
            # plt.show()
            #
            # data0 = np.histogram(img_arr, bins=10, range=None, normed=None, weights=None, density=None)
            # print(data0)
            # _ = plt.hist(data0, bins='auto')
            # plt.title("Histogram with 'auto' bins")
            # plt.show()


            # first approach
            # pdfr_arr = copy.deepcopy(img_arr)
            # n_val = 197*233*189                    # assuming all images are of the same size!!!!
            # for xit in range(198):
            #     for yit in range(234):
            #         for zit in range(190):
            #         r_val = img_arr[xit,yit,zit]
            #         nj_val = None #histogram value of r_val[xit,yit,zit]
            #         pdfr_arr[xit,yit,zit] = nj_val/n_val
            # img_arr = pdfr_arr

        elif self.normalization == 'HM2':
            print('Normalization: Histogram Matching Method 2')

            # # Martin:
            # # CODE TESTING
            # fig, axs = plt.subplots(1, 2)
            # fig.suptitle('HM2')
            # slice = 100
            # axs[0].imshow(img_arr[slice, :, :])
            # axs[0].title.set_text('Raw image')
            # axs[1].imshow(img_arr[slice, :, :])
            # axs[1].title.set_text('Raw image')
            # plt.show()
            #
            # fig, axs = plt.subplots(1, 2)
            # fig.suptitle('HM2 histogram')
            # data0 = img_arr[img_arr > img_arr.mean()].flatten()
            # data1 = img_arr[img_arr > img_arr.mean()].flatten()
            # colors = ['b', 'r']
            # axs[0].hist(data0, 400, color=colors[0])
            # axs[0].title.set_text('Raw image')
            # axs[1].hist(data1, 400, color=colors[1])
            # axs[1].title.set_text('Raw image')
            # # axs[2].hist([data0,data1], 400,color=colors, alpha=0.5)
            # axs[0].grid(True)
            # axs[1].grid(True)
            # plt.show()

            # ref_t1 = sitk.ReadImage(os.path.join('E:/01_Programme/Git/00_MyFolder/MIALab/data/train/100307', 'T1native.nii.gz'))
            ref_t1 = sitk.ReadImage('E:/01_Programme/Git/00_MyFolder/MIALab/data/train/100307/T1native.nii.gz')
            # rf1 = sitk.GetArrayFromImage(ref_t1)
            # ref_t2 = sitk.ReadImage(os.path.join('E:/01_Programme/Git/00_MyFolder/MIALab/data/train/100307', 'T2native.nii.gz'))
            ref_t2 = sitk.ReadImage('E:/01_Programme/Git/00_MyFolder/MIALab/data/train/100307/T2native.nii.gz')
            # rf2 = sitk.GetArrayFromImage(ref_t2)
            # read in reference image --> ref_arr
            # referenceH = data.ref_arr()
            ref_t1_np = sitk.GetArrayFromImage(params.reference_image)
            ref_t1 = params.reference_image
            referenceH = ref_t1
            # imageH = data.img_arr()
            imageH = img_arr
            # matched = match_histograms(imageH, referenceH, channel_axis=-1)
            matched=sitk.HistogramMatching(image, referenceH)
            matched_array=sitk.GetArrayFromImage(matched)
            image_array=sitk.GetArrayFromImage(image)
            ref_array=sitk.GetArrayFromImage(referenceH)
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8,3), sharex=True, sharey=True)
            for aa in (ax1, ax2, ax3):
                aa.set_axis_off()
            ax1.imshow(image_array[98,:,:])
            ax1.set_title('source')
            ax2.imshow(ref_array[98,:,:])
            ax2.set_title('reference')
            ax3.imshow(matched_array[98,:,:])
            ax3.set_title('Matched')
            plt.tight_layout()
            plt.show()
            img_arr = matched

            fig, axs = plt.subplots(1, 3)
            fig.suptitle('HM2 histogram')
            # data0 = (image_array[image_array > image_array.mean()].flatten())/np.max(image_array)
            # data1 = (ref_array[ref_array > ref_array.mean()].flatten())/np.max(ref_array)
            # data2 = (matched_array[matched_array > matched_array.mean()].flatten())/np.max(matched_array)
            data0 = image_array.flatten()
            data1 = ref_array.flatten()
            data2 = matched_array.flatten()

            colors = ['b', 'r', 'g']
            y_limits = (0, 5e-5)
            axs[0].hist(data0, 400, color=colors[0],density=True)
            axs[0].title.set_text('image_array')
            axs[0].set_ylim(*y_limits)
            axs[1].hist(data1, 400, color=colors[1],density=True)
            axs[1].title.set_text('ref_array')
            axs[1].set_ylim(*y_limits)
            axs[2].hist(data2, 400, color=colors[2],density=True)
            axs[2].title.set_text('matched_array')
            axs[2].set_ylim(*y_limits)
            plt.show()

        elif self.normalization == 'WS':
            print('Normalization: White Stripe Method')
            #"""
            #execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
            modality="t1"
            width=0.05
            if modality is None:
                modality = "t1"
            #mask = sitk.GetArrayFromImage(self.mask)
            #masked = data * mask
            voi = image
            # wm_mode = im.get_tissue_mode(voi, modality)
            # wm_mode_quantile: float = np.mean(voi < wm_mode).item()
            # lower_bound = max(wm_mode_quantile - width, 0.0)
            # upper_bound = min(wm_mode_quantile + width, 1.0)
            # ws_l, ws_u = np.quantile(voi, (lower_bound, upper_bound))
            # whitestripe = (voi > ws_l) & (voi < ws_u)




        else:
            print('WARNING: unknown normalization method initialisation input!')
            img_out = img_arr # why not needed? and why does it lead to an error?

        if isinstance(img_arr, sitk.Image):
            img_arr.CopyInformation(image)
            return img_arr
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        print('***** execute ImageNormalization')

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """

        print('***** str ImageNormalization')

        return 'ImageNormalization:\n' \
            .format(self=self)

    print('***** ImageNormalization')

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

        print('***** init SkullStrippingParameters')

    print('***** SkullStrippingParameters')


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

        print('***** init SkullStripping')

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

        print('***** execute SkullStripping')

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        print('***** str SkullStripping')

        return 'SkullStripping:\n' \
            .format(self=self)

    print('***** SkullStripping')


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

        print('***** init ImageRegistrationParameters')

    print('***** ImageRegistrationParameters')


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

        print('***** init ImageRegistration')

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

        print('***** execute ImageRegistration')

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """

        print('***** str ImageRegistration')

        return 'ImageRegistration:\n' \
            .format(self=self)

    print('***** ImageRegistration')