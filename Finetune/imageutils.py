import os
import SimpleITK as sitk
import numpy as np
import random
from scipy.ndimage import rotate, zoom, shift, generate_binary_structure, affine_transform
from scipy.ndimage import binary_fill_holes, binary_dilation, grey_dilation, binary_erosion
from scipy import ndimage
from monai.transforms import Compose, RandAffined, RandFlipd, RandScaleIntensityd, RandRicianNoised, RandAdjustContrastd


def read_mhd(f):
    '''
    Read in an mhd file.
    :param f: filepath to be read in
    :return: numpy array containing image, (Z,Y,X) order
    '''
    if f[-4:] != '.mhd':
        f += '.mhd'
    if not os.path.isfile(f):
        return None
    itkimage = sitk.ReadImage(f)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    return numpyImage


def volumn_normalization(orig_img, method=0, seg_img=None, const=None, excludezero=True):
    '''
    Rescale the whole image volume intensity to the range of [0, 1]
    :param orig_img:
    :param method: 0-normalization by constant, 1-percentile (0.05-99.95) normalization
    :param seg_img:
    :param const:
    :param excludezero: exclude zero voxels from normalization
    :return: normalized volume
    '''
    if method == 0:
        norm_img = orig_img.astype(np.float32) / const

    if method == 1:
        # use the whole volume to calculate percentile
        threshold = (0.05, 99.95)
        if excludezero:
            val_l, val_h = np.percentile(orig_img[orig_img > 0], threshold)  # only non-zero voxels
        else:
            val_l, val_h = np.percentile(orig_img, threshold)
        norm_img = np.copy(orig_img).astype(np.float32)
        norm_img = (norm_img - val_l) / (val_h - val_l + 1e-5)
        norm_img[norm_img < 0] = 0
        norm_img[norm_img > 1] = 1

    if method == 2:
        # TODO: Rescale the image intensity to the range of [0, 1],
        norm_img = np.copy(orig_img)

    return norm_img


def create_lesion_mask(lesion_img, radius=5):
    '''
    Create 3D lesion volume based on center point
    '''
    spacing = [3, 0.5, 0.5]
    PIRADS_thres = 3

    new_mask = np.zeros_like(lesion_img, dtype=np.uint8)
    dimz, dimy, dimx = new_mask.shape
    idz, idy, idx = np.where(lesion_img >= PIRADS_thres)
    if len(idz) > 0:
        zz, yy, xx = np.mgrid[:dimz, :dimy, :dimx]
        for i in range(len(idz)):
            dist = ((zz - idz[i]) * spacing[0]) ** 2 + ((yy - idy[i]) * spacing[1]) ** 2 + ((xx - idx[i]) * spacing[
                2]) ** 2
            new_mask[dist < radius ** 2] = 1
    return new_mask


def dilate3D(orig_vol, noiselevel=900, minmask=2000):
    '''
    Dilate mask in 3D. The original mask labels must be integers (categorical).
    :param orig_vol: input mask volume, Z*Y*X
    :param noiselevel: mask smaller than this will be removed
    :param minmask: minimum mask size in 2D slice
    :return: dilated mask volume
    '''

    # remove noise mask slices
    vol_binary = np.array(orig_vol > 0, dtype=np.int32)
    for i in range(len(vol_binary)):
        if np.sum(vol_binary[i]) < noiselevel and np.sum(vol_binary[i]) > 0:
            vol_binary[i] = 0
    vol_int = np.asarray(orig_vol * vol_binary, dtype=np.int32)

    if minmask is None:
        # no dilation
        return vol_int
    else:
        # dilate small masks
        struct3 = generate_binary_structure(3, 1)  # no diagonals
        dilatedvol_binary = np.copy(vol_binary)
        dilatedvol_int = np.copy(vol_int)
        slice_ind = np.where(vol_binary.sum(axis=(1, 2)) > 0)[0]
        slice_newmask = np.zeros(len(vol_binary))
        slice_newmask[slice_ind] = 1
        new_mask = np.zeros_like(orig_vol, dtype=np.int32)

        while slice_newmask.sum() > 0:
            for i in slice_ind:
                if slice_newmask[i] == 0 or np.sum(dilatedvol_binary[i]) < minmask:
                    continue
                new_mask[i] = np.copy(dilatedvol_int[i])
                slice_newmask[i] = 0  # the mask is copied if size>minmask
            tmp = grey_dilation(dilatedvol_int, footprint=struct3, mode='nearest')
            dilatedvol_int[dilatedvol_int == 0] = tmp[dilatedvol_int == 0]
            dilatedvol_binary = np.array(dilatedvol_int > 0, dtype=np.int32)

        return new_mask


def erosion2D(vol):
    """2D erosion on each non-zero slices"""
    vol_binary = np.array(vol > 0, dtype=np.int32)
    for i in range(len(vol_binary)):
        if np.sum(vol_binary[i]) > 0:
            vol_binary[i] = binary_erosion(vol_binary[i])
    return vol * vol_binary


def adjust_component(orig_vol, prob_range=0.5):
    # not used
    struct3 = ndimage.generate_binary_structure(3, 3)
    vol_binary = np.asarray(orig_vol > 0, dtype=np.int)
    labeled_vol, n_components = ndimage.label(vol_binary, structure=struct3)

    for i in range(1, n_components + 1):
        # remove low prob voxels from the component
        maxprob = np.max(orig_vol[labeled_vol == i])
        vol_thres = np.copy(orig_vol)
        vol_thres[labeled_vol != i] = 0
        vol_thres[vol_thres < maxprob - prob_range] = 0

        # keep only one component if multiple components are created after thresholding
        tmp_binary = np.asarray(vol_thres > 0, dtype=np.int)
        tmp_labeled, n = ndimage.label(tmp_binary, structure=struct3)
        if n > 1:
            pos = ndimage.measurements.maximum_position(vol_thres)
            idx = tmp_labeled[pos]
            vol_thres[tmp_labeled != idx] = 0


class Augmentation2D:
    def __init__(self, rot45=0, crop=0, translation=0, rot90=0, rot180=0, flip=0, align=0, scale=0,
                 maskchannel=None, dilatechannel=None):
        '''
        Augmentation of 2D images. It is recommended to use binary mask (mask will be smoother after augmentation).
        :param rot45: rotation angle
        :param crop: random crop size, int or tuple
        :param translation: image translation, ignored if using crop
        :param rot90: 90 degree rotation
        :param rot180: 180 degree rotation
        :param flip: image flip
        :param align: misalignment between channels (translation only)
        :param scale: intensity scaling factor (e.g. 10 means random scale of 10% difference, 0.9-1.1)
        :param maskchannel: list of mask channels (no interpolation on the values)
        :param dilatechannel: list of mask channels that need random dilation (0-5 iterations), only for binary mask
        '''
        self.rot45 = rot45
        self.crop = crop
        self.translation = translation
        self.rot90 = rot90
        self.rot180 = rot180
        self.flip = flip
        self.align = align
        self.scale = scale
        if maskchannel is None:
            self.maskchannel = []
        elif isinstance(maskchannel, int):
            self.maskchannel = [maskchannel]
        else:
            self.maskchannel = list(maskchannel)
        if dilatechannel is None:
            self.dilatechannel = []
        elif isinstance(dilatechannel, int):
            self.dilatechannel = [dilatechannel]
        else:
            self.dilatechannel = list(dilatechannel)

    def __call__(self, data):
        if len(self.dilatechannel) > 0:
            self.struct2 = generate_binary_structure(2, 2)
            data = self._dilate(data)
        if self.rot45 != 0:
            data = self._rot45(data)
        if self.align != 0:
            data = self._align(data)
        if self.crop != 0:
            data = self._crop(data)
            self.translation = 0
        if self.translation != 0:
            data = self._translation(data)
        if self.rot90 != 0:
            data = self._rot90(data)
        if self.rot180 != 0:
            data = self._rot180(data)
        if self.flip != 0:
            data = self._flip(data)
        if self.scale != 0:
            data = self._scale(data)
        return data

    def _rot45(self, data):
        # rotate n degrees
        newdata = np.zeros_like(data)
        tmprnd = random.randint(-1 * self.rot45, self.rot45)
        for i in range(len(data)):
            if i in self.maskchannel:
                if np.max(data[i]) > 1:
                    # categorical data, no interpolation
                    newdata[i] = rotate(data[i], tmprnd, order=0, axes=(1, 0), output=np.float32, reshape=False)
                else:
                    # use linear interpolation to smooth the augmented segmentation (for 0/1 binary image only)
                    newdata[i] = rotate(data[i], tmprnd, order=1, axes=(1, 0), output=np.float32, reshape=False)
            else:
                # fill blank areas with mirrored image
                newdata[i] = rotate(data[i], tmprnd, order=1, axes=(1, 0), output=np.float32, reshape=False,
                                    mode='mirror')
        return newdata

    def _crop(self, data):
        # random crop based on the mask (the whole mask must be included in the cropped image)
        if len(self.maskchannel) < 1:
            mask = np.zeros(data.shape[1:])
        else:
            mask = data[self.maskchannel].sum(axis=0)
            mask[mask > 0] = 1
        dim1, dim2 = mask.shape
        if isinstance(self.crop, int):
            range1, range2 = self.crop, self.crop
        else:
            range1, range2 = self.crop

        r = np.flatnonzero(np.sum(mask, axis=1))
        if len(r) == 0:
            p1 = random.randint(0, dim1 - range1)  # empty mask
        else:
            if r[-1] - r[0] + 1 < range1:
                p1 = random.randint(max(0, r[-1] - range1 + 1), min(dim1 - range1, r[0]))
            else:
                p1 = random.randint(r[0], r[-1] - range1 + 1)

        r = np.flatnonzero(np.sum(mask, axis=0))
        if len(r) == 0:
            p2 = random.randint(0, dim2 - range2)
        else:
            if r[-1] - r[0] + 1 < range2:
                p2 = random.randint(max(0, r[-1] - range2 + 1), min(dim2 - range2, r[0]))
            else:
                p2 = random.randint(r[0], r[-1] - range2 + 1)

        return np.copy(data[:, p1:p1 + range1, p2:p2 + range2])

    def _translation(self, data):
        # shift n pixels
        x_shift = random.randint(-1 * self.translation, self.translation)
        y_shift = random.randint(-1 * self.translation, self.translation)
        nshift = (0, x_shift, y_shift)
        return shift(data, nshift, order=1, mode='constant', cval=0)

    def _rot90(self, data):
        # 90 degree rotation
        k = random.randint(-1, 1)
        return np.copy(np.rot90(data, k, axes=(1, 2)))

    def _rot180(self, data):
        # 180 degree rotation
        k = random.randint(0, 1)
        return np.copy(np.rot90(data, k * 2, axes=(1, 2)))

    def _flip(self, data):
        # flip
        newdata = np.copy(data)
        if random.randint(0, 1) == 1:
            newdata = np.flip(newdata, 1)
        if random.randint(0, 1) == 1:
            newdata = np.flip(newdata, 2)
        return np.copy(newdata)

    def _dilate(self, data):
        # random binary dilation
        newdata = np.copy(data)
        for i in self.dilatechannel:
            tmprnd = random.randint(0, 5)
            if tmprnd > 0:
                new_mask = binary_dilation(data[i], structure=self.struct2, iterations=tmprnd)
                newdata[i] = binary_fill_holes(new_mask, structure=self.struct2)
        return newdata

    def _align(self, data):
        # random shift channels
        newdata = np.copy(data)
        for i in range(1, len(data)):
            if i in self.maskchannel:
                # do not shift mask channels
                continue
            else:
                # shift n pixels
                x_shift = random.randint(-1 * self.align, self.align)
                y_shift = random.randint(-1 * self.align, self.align)
                newdata[i] = shift(data[i], (x_shift, y_shift), order=1, mode='constant', cval=0)
        return newdata

    def _scale(self, data):
        # random scaling factor
        newdata = np.copy(data)
        for i in range(len(data)):
            if i in self.maskchannel:
                continue  # no scale for mask channels
            else:
                tmprnd = 1 + random.randint(-1 * self.scale, self.scale) / 100
                newdata[i] = data[i] * tmprnd
        return newdata


def Augmentation2D_dict(**kwargs):
    ''' Augmentation of 2D images. Use MONAI Dictionary Transforms '''
    flipX = kwargs.get('flipX', None)
    flipY = kwargs.get('flipY', None)
    rotxy = kwargs.get('rotxy', None)
    translation = kwargs.get('translation', None)
    spatialscale = kwargs.get('spatialscale', None)
    intensityscale = kwargs.get('intensityscale', None)
    noise = kwargs.get('noise', None)
    gamma = kwargs.get('gamma', None)
    keys_spatial = kwargs.get('spatial', ['image', 'seg', 'label'])
    mode_spatial = ['bilinear' if k == 'image' else 'nearest' for k in keys_spatial]
    keys_intensity = kwargs.get('intensity', ['image'])
    transforms = []
    # spatial
    if flipX is not None:
        transforms.append(RandFlipd(keys=keys_spatial, prob=flipX, spatial_axis=-1))
    if flipY is not None:
        transforms.append(RandFlipd(keys=keys_spatial, prob=flipY, spatial_axis=-2))
    if (rotxy is not None) or (translation is not None) or (spatialscale is not None):
        rotxy = np.pi / 180 * rotxy if rotxy is not None else None
        transforms.append(RandAffined(keys=keys_spatial, mode=mode_spatial, prob=0.8, rotate_range=rotxy,
                                      translate_range=translation, scale_range=spatialscale, padding_mode='zeros'))
    # intensity
    if intensityscale is not None:
        transforms.append(RandScaleIntensityd(keys=keys_intensity, factors=intensityscale, prob=0.5))
    if noise is not None:
        transforms.append(RandRicianNoised(keys=keys_intensity, prob=0.5, mean=0, std=noise))
    if gamma is not None:
        transforms.append(RandAdjustContrastd(keys=keys_intensity, prob=0.5, gamma=gamma))
    return Compose(transforms)


def Augmentation3D(**kwargs):
    '''
    Augmentation of 3D volumes. Input volumes are ch*Z*Y*X.
    :param flipX: image flip in x direction, default 0
    :param rotxy: max rotation angle within XY plane, default 0
    :param translation: (tuple) max translation voxels in each dimension, default None
    :param spaticalscale: (tuple) spatial scaling range of each dim, default None
    :param intensityscale: intensity scaling factor (e.g. 10 means random scale of 10% difference, 0.9-1.1)
    '''
    flipX = kwargs.get('flipX', None)
    rotxy = kwargs.get('rotxy', None)
    translation = kwargs.get('translation', None)
    spatialscale = kwargs.get('spatialscale', None)
    intensityscale = kwargs.get('intensityscale', None)
    from monai.transforms import Compose, RandAffine, RandFlip, RandScaleIntensity
    transforms = []
    if flipX is not None:
        transforms.append(RandFlip(prob=0.5, spatial_axis=-1))
    if rotxy is not None:
        rotxy = np.pi / 180 * rotxy
    transforms.append(RandAffine(prob=0.8, rotate_range=rotxy,
                                 translate_range=translation, scale_range=spatialscale, padding_mode='zeros'))
    if intensityscale is not None:
        transforms.append(RandScaleIntensity(factors=intensityscale, prob=0.5))

    return Compose(transforms)


def Augmentation3D_dict(**kwargs):
    ''' Augmentation of 3D volumes. Use MONAI Dictionary Transforms '''
    flipX = kwargs.get('flipX', None)
    rotxy = kwargs.get('rotxy', None)
    translation = kwargs.get('translation', None)
    spatialscale = kwargs.get('spatialscale', None)
    intensityscale = kwargs.get('intensityscale', None)
    noise = kwargs.get('noise', None)
    gamma = kwargs.get('gamma', None)
    keys_spatial = kwargs.get('spatial', ['image', 'heatmap'])
    keys_intensity = kwargs.get('intensity', ['image'])
    transforms = []
    # spatial
    if flipX is not None:
        transforms.append(RandFlipd(keys=keys_spatial, prob=flipX, spatial_axis=-1))
    if (rotxy is not None) or (translation is not None) or (spatialscale is not None):
        rotxy = np.pi / 180 * rotxy if rotxy is not None else None
        transforms.append(RandAffined(keys=keys_spatial, prob=0.8, rotate_range=rotxy,
                                      translate_range=translation, scale_range=spatialscale, padding_mode='zeros'))
    # intensity
    if intensityscale is not None:
        transforms.append(RandScaleIntensityd(keys=keys_intensity, factors=intensityscale, prob=0.5))
    if noise is not None:
        transforms.append(RandRicianNoised(keys=keys_intensity, prob=0.5, mean=0, std=noise))
    if gamma is not None:
        transforms.append(RandAdjustContrastd(keys=keys_intensity, prob=0.5, gamma=gamma))
    return Compose(transforms)


def heatmap2LoS(H, T, LoSrange=(1, 3, 5), LoSstep=0.5, intermediateProb=None):
    '''
    Convert heatmap value [0, 1] to LoS (Lmin, Lmid, Lmax) using stepwise linear function:
        0->Lmin, T->Lmid, 1->Lmax
        (default range Lmin=1, Lmid=3, Lmax=5, step 0.5)

    The intermediateProb is optional (T <= intermediateProb <= 1). If given, additional stepwise linear function is
    applied so that intermediateProb is mapped to (Lmid+Lmax)/2. This enables us to choose flexible threshold for
    deciding PIRADS 4/5 lesions.

    Input: H - heatmap value, T- heatmap threshold
    Output: L - LoS value
    '''
    from math import floor
    x = np.around(np.asarray(LoSrange, dtype=np.float32) / LoSstep)
    if H < T:
        L = floor(H / T * (x[1] - x[0])) + x[0]  # round to the largest integer less than the number
    else:
        if intermediateProb is None:
            L = round((H - T) / (1 - T) * (x[2] - x[1])) + x[1]  # round off the number to the nearest integer
        else:
            if intermediateProb < T or intermediateProb > 1:
                raise ('intermediateProb must be between threshold and 1')  # raise error if not in the proper range
            T0 = intermediateProb
            L0 = (x[1] + x[2]) / 2
            # compute new LoS
            if H < T0:
                # round to the largest integer less than the number
                L = floor((H - T) / (T0 - T) * (L0 - x[1])) + x[1]
            else:
                # round off the number to the nearest integer
                L = round((H - T0) / (1 - T0) * (x[2] - L0) + L0)

    L = L * LoSstep
    L = int(L) if isinstance(LoSstep, int) else float(L)
    return L


def MyOtsu(image):
    ''' Calculate Otso for [0,255] image'''
    image_min, image_max = int(min(image)), int(max(image))
    n_bins = image_max - image_min + 1

    bin_centers = np.zeros(n_bins, dtype=int)
    for i in range(0, n_bins):
        bin_centers[i] = image_min + i

    hist = np.zeros(n_bins, dtype=float)
    for x in image:
        hist[x - image_min] += 1

    total_sum = len(image)  # total number of voxels in the patch

    # calculate weight1 and weight2
    weight1 = np.zeros(n_bins, dtype=float)
    weight1[0] = hist[0]
    for i in range(1, n_bins):
        weight1[i] = hist[i] + weight1[i - 1]

    weight2 = np.zeros(n_bins, dtype=float)
    weight2[0] = total_sum
    for i in range(1, n_bins):
        weight2[i] = weight2[i - 1] - hist[i - 1]

    hist_times_bin = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        hist_times_bin[i] = hist[i] * bin_centers[i]

    # calculate mean1
    cumsum_mean1 = np.zeros(n_bins, dtype=float)
    mean1 = np.zeros(n_bins, dtype=float)
    cumsum_mean1[0] = hist_times_bin[0]
    for i in range(1, n_bins):
        cumsum_mean1[i] = cumsum_mean1[i - 1] + hist_times_bin[i]
        mean1[i] = cumsum_mean1[i] / weight1[i]

    # calculate mean2
    cumsum_mean2 = np.zeros(n_bins, dtype=float)
    mean2 = np.zeros(n_bins, dtype=float)
    cumsum_mean2[0] = hist_times_bin[n_bins - 1]  # last item of hist_times_bin
    for i in range(1, n_bins):
        cumsum_mean2[i] = cumsum_mean2[i - 1] + hist_times_bin[n_bins - 1 - i]
    for i in range(n_bins):
        mean2[n_bins - 1 - i] = cumsum_mean2[i] / weight2[n_bins - 1 - i]

    # calculate inter-class variance
    Inter_class_variance = np.zeros(n_bins - 1, dtype=float)  # only n_bins-1 values
    dnum = 10000000000
    for i in range(n_bins - 1):
        Inter_class_variance[i] = ((weight1[i] * weight2[i + 1] * (mean1[i] - mean2[i + 1])) / dnum) * (
                mean1[i] - mean2[
            i + 1])  # there's a bug in the web-page, the second item should be weight2[i+1], not weight2[i]

    maxi = 0
    for i in range(n_bins - 1):
        if maxi < Inter_class_variance[i]:
            maxi = Inter_class_variance[i]
            getmax = i

    MyOtsuResult = getmax
    from skimage.filters import threshold_otsu
    SKOtsu = threshold_otsu(image)
    print('MyOtsu {}, SKimage Otsu {}'.format(MyOtsuResult, SKOtsu))

    return MyOtsuResult


def get_bval(patientfolder):
    '''
    get DWI b-values of mhd header
    :return: list of b values: low-b, high-b, calculated-b
    '''
    imagelist = ['DWI_LO_original.mhd', 'DWI_HI_original.mhd', 'Calculated_DWI_HI.mhd']
    bvals = []
    for s in imagelist:
        b = 'N/A'
        with open(os.path.join(patientfolder, s), 'r') as f:
            for line in f:
                if 'bValue' in line or 'b_value' in line:
                    b = line.split('= ')[1].rstrip()
            bvals.append(int(b))
    return bvals


def get_DWIadj(highB):
    '''get scaling factor for ADC and HiB'''
    adc_scale, highb_scale = 1, 1
    if highB >= 800 and highB <= 1200:
        pass
    elif highB < 800:
        adc_scale, highb_scale = 0.9, 1.2
    elif highB > 1200 and highB <= 1500:
        adc_scale, highb_scale = 1.2, 0.9
    elif highB > 1500:
        adc_scale, highb_scale = 2, 0.3
    return adc_scale, highb_scale


def exclude_empty_slice(mask, image, threshold=0.1):
    '''Exclude the slice where empty voxels are more than threshold'''
    newmask = np.copy(mask)
    if mask.shape != image.shape:
        print('Warning: Mask and Image are in different sizes.')
        return newmask
    for i in range(len(mask)):
        voxels = image[i][mask[i] > 0]
        if len(voxels) > 0 and np.sum(voxels <= 1e-5) / len(voxels) > threshold:
            newmask[i] = 0
    sliceidx = np.flatnonzero(np.sum(newmask, axis=(1, 2)))
    if len(sliceidx)>0 and sliceidx[-1] - sliceidx[0] + 1 != len(sliceidx):
        # do not exclude the slice in the middle
        sliceidx = np.arange(sliceidx[0], sliceidx[-1] + 1)
    if len(sliceidx)>0:
        newmask[sliceidx] = np.copy(mask)[sliceidx]
    return newmask


def getaffinematrix(imagefile, mat='V2W'):
    ''' Get affine matrix of the volume, using 4x4 matrix '''
    if isinstance(imagefile, str):
        img = sitk.ReadImage(imagefile)
    else:
        img = imagefile
    V2W = np.eye(4, dtype=np.float64)
    V2W[:3, :3] = np.array(img.GetDirection()).reshape((3, 3)).dot(np.diag(img.GetSpacing()))
    V2W[:3, 3:] = np.array(img.GetOrigin()).reshape((3, 1))
    W2V = np.linalg.inv(V2W)
    if mat=='V2W':
        return V2W
    elif mat=='W2V':
        return W2V
    else:
        return None

def image_reg(srcimg, dstimg, regmat=np.eye(4), src_image_array=None):
    '''
    apply registration matrix (regmat 4x4) and resample src image in dst image space
    Image is loaded from srcimg if the array (X*Y*Z) is not given.
    '''
    if isinstance(srcimg, str) and srcimg[-4:] == '.mhd':
        srcimg = sitk.ReadImage(srcimg)
    if isinstance(dstimg, str) and dstimg[-4:] == '.mhd':
        dstimg = sitk.ReadImage(dstimg)

    # load header info
    V2W_src = np.eye(4, dtype=np.float64)
    V2W_src[:3, :3] = np.array(srcimg.GetDirection()).reshape((3, 3)).dot(np.diag(srcimg.GetSpacing()))
    V2W_src[:3, 3:] = np.array(srcimg.GetOrigin()).reshape((3, 1))
    W2V_src = np.linalg.inv(V2W_src)
    V2W_dst = np.eye(4, dtype=np.float64)
    V2W_dst[:3, :3] = np.array(dstimg.GetDirection()).reshape((3, 3)).dot(np.diag(dstimg.GetSpacing()))
    V2W_dst[:3, 3:] = np.array(dstimg.GetOrigin()).reshape((3, 1))

    # apply reg matrix
    affineMatrix = np.linalg.multi_dot([W2V_src, regmat, V2W_dst])
    rotaionMatrix = affineMatrix[:3, :3]
    offsetMatrix = affineMatrix[:3, 3]
    if src_image_array is None:
        src_image_array = sitk.GetArrayFromImage(srcimg).transpose()

    src_resampled = affine_transform(src_image_array, rotaionMatrix, order=1, offset=offsetMatrix,
                                     output_shape=dstimg.GetSize(), output=src_image_array.dtype,
                                     cval=0, prefilter=False)
    return src_resampled


def dwi_compute(lowBVol, HighBVol, LowB, HighB, computeB=2000, dtype=np.float32):
    '''  Compute new DWI image at given b-value. '''
    # const definition
    NoiseMinimum = 10
    ADCCut = 300
    MaxADCOFFSET = 1000
    MinADC = 0
    DWIScalingFactor = 10

    # Step 1: compute slope and intercept
    orig_lo = np.array(lowBVol, dtype=np.float32)
    orig_hi = np.array(HighBVol, dtype=np.float32)
    log_lo, log_hi = np.log(orig_lo + np.finfo(float).eps), np.log(orig_hi + np.finfo(float).eps)
    log_lo[orig_lo == 0] = 0  # set log to zero for zero-intensity voxels
    log_hi[orig_hi == 0] = 0
    Slope = (log_hi - log_lo) / (HighB - LowB)
    Intercept = ((log_hi + log_lo) - (HighB + LowB) * Slope) / 2

    # Step 2: compute ADC
    ADC = np.maximum(MinADC, Slope * -1e6)  # unit 1e-6 mm2/s

    # Step 3: compute B0
    B0 = np.exp(Intercept)

    # Step 4: noise estimation
    noisevoxels = B0[np.logical_and(ADC < ADCCut, B0 > NoiseMinimum)]
    noisevoxels = np.sort(noisevoxels)
    if len(noisevoxels) > 0.01 * B0.size:
        NoiseLevel = noisevoxels[int(len(noisevoxels) / 2)]
    else:
        NoiseLevel = NoiseMinimum
    NoiseLevel *= 3

    # Step 5: compute B2000 using modified calculation formula
    Fraction = np.minimum(1, B0 / NoiseLevel)
    NoiseEstimation_ADCOffset = MaxADCOFFSET * 1e-6
    ADCOffset = NoiseEstimation_ADCOffset * np.sqrt(1 - Fraction * Fraction)
    if computeB <= HighB:
        TempExponent = computeB * -Slope  # interpolation, ADC*b
    else:
        # extrapolation: apply minimum plausible ADC
        TempExponent = HighB * -Slope + (computeB - HighB) * np.maximum(-Slope, ADCOffset)  # ADC*b=ADC*hB+ADC*(b-hB)
    sig_extrapolated_b = B0 * np.exp(-TempExponent)  # S0*exp(ADC*b)
    DWI_newB = DWIScalingFactor * sig_extrapolated_b

    return ADC.astype(dtype), B0.astype(dtype), DWI_newB.astype(dtype)


def save2mhd(imgarray, output_file=None, ref_file=None):
    """ Save array as an mhd file for quick visualization """
    # save as a temp file if not specified
    if output_file is None:
        output_file = 'tmptest/TempSavedImage.mhd'

    # always use numpy datatype as output format
    with open(output_file[:-3] + 'img', 'w') as f:
        if imgarray.dtype == np.float64 or imgarray.dtype == np.float32:
            imgarray.flatten().astype(np.float32).tofile(f)
            ElementType = 'MET_FLOAT'
        elif imgarray.dtype == np.int32 or imgarray.dtype == np.int16:
            imgarray.flatten().astype(np.int16).tofile(f)
            ElementType = 'MET_SHORT'
        elif imgarray.dtype == np.int8:
            imgarray.flatten().tofile(f)
            ElementType = 'MET_CHAR'
        elif imgarray.dtype == np.uint32 or imgarray.dtype == np.uint16:
            imgarray.flatten().astype(np.uint16).tofile(f)
            ElementType = 'MET_USHORT'
        elif imgarray.dtype == np.uint8:
            imgarray.flatten().tofile(f)
            ElementType = 'MET_UCHAR'

    # output as a resampled space image if not reference is given
    if ref_file is None:
        with open(output_file, 'w') as f:
            f.write('NDims = 3\n')
            f.write('DimSize = ' + ' '.join([str(x) for x in imgarray.shape[::-1]]) + '\n')
            f.write('ElementSpacing = 0.5 0.5 3\nPosition = 0 0 0\n TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
            f.write('ElementType = ' + ElementType + '\n')
            f.write('ElementByteOrderMSB = False\n')
            f.write(f'ElementDataFile = {os.path.basename(output_file)[:-4]}.img\n')
    else:
        f1 = open(ref_file, 'r')
        f2 = open(output_file, 'w')
        for line in f1:
            if 'ElementDataFile' in line:
                f2.write(f'ElementDataFile = {os.path.basename(output_file)[:-4]}.img\n')
            elif 'ElementType' in line:
                f2.write(f'ElementType = {ElementType}\n')
            else:
                f2.write(line)
        f1.close()
        f2.close()


def RandDilate(mask, iter):
    ''' fix/random dilation of a 2D binary mask '''
    if isinstance(iter, int):
        iternum = iter
    else:
        iternum = random.randint(iter[0], iter[1])
    newmask = np.array(mask > 0, dtype=np.int8)
    if iternum > 0:
        struct2 = generate_binary_structure(2, 2)
        newmask = binary_dilation(newmask, structure=struct2, iterations=iternum)
        newmask = binary_fill_holes(newmask, structure=struct2)
    return newmask


def RandChannelShift(data, voxelshift):
    ''' random shift channels, ch*Y*Z or ch*Z*Y*X, voxelshift in XY plane '''
    if len(data) == 1:
        return data
    newdata = np.zeros_like(data)
    newdata[0] = data[0]
    mx, my = data.shape[-1], data.shape[-2]
    x_t = random.randint(-1 * voxelshift, voxelshift)
    y_t = random.randint(-1 * voxelshift, voxelshift)
    print(x_t,y_t)
    x0, x1 = max(0, x_t), min(mx, mx + x_t)
    x0_o, x1_o = max(0, -x_t), min(mx, mx - x_t)
    y0, y1 = max(0, y_t), min(my, my + y_t)
    y0_o, y1_o = max(0, -y_t), min(my, my - y_t)

    if data.ndim == 3:
        newdata[1:, y0:y1, x0:x1] = data[1:, y0_o:y1_o, x0_o:x1_o]
    elif data.ndim == 4:
        newdata[1:, :, y0:y1, x0:x1] = data[1:, :, y0_o:y1_o, x0_o:x1_o]
    else:
        return None

    return newdata


if __name__ == '__main__':
    print('')
    mrdata = np.load('/home/z003cx0c/ProstateAI/Data/VolumeSamples_20230612/Changhai/1003100032.npz')
    heatmapfile = '/home/z003cx0c/ProstateAI/TestingResults/V202104/VB80/HeatMap/Changhai/1003100032.mhd'
    HM = sitk.GetArrayFromImage(sitk.ReadImage(heatmapfile))
    print(mrdata['img'].shape)

    x = mrdata['img']
    y = RandChannelShift(x, 25)


    import matplotlib.pyplot as plt

    plt.subplot(2, 2, 1)
    plt.imshow(x[0,15])
    plt.subplot(2, 2, 2)
    plt.imshow(x[2,15])
    plt.subplot(2, 2, 3)
    plt.imshow(y[0,15])
    plt.subplot(2, 2, 4)
    plt.imshow(y[2,15])
    plt.show()
