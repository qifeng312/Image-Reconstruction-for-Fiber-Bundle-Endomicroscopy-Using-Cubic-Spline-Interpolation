# Cubic spline interpolation
import numpy as np
import pybundle
from pybundle.bundle_calibration import BundleCalibration
from scipy.interpolate import griddata


def calib_spline_interp(img, coreSize, gridSize, **kwargs):
    """
    Performs calibration for fiber bundle image reconstruction using cubic spline interpolation.
    Returns a BundleCalibration instance to be used in recon_spline_interp.

    Parameters:
        img : ndarray
            Calibration image of the fiber bundle, either grayscale (2D) or color (3D).
        coreSize : float
            Estimated average spacing between fiber cores, used for core detection.
        gridSize : int
            Size of the output reconstructed image (image will be square).

    Optional Keyword Arguments:
        centreX, centreY : int
            Manually specify the bundle center coordinates; automatically detected if not provided.
        radius : int
            Radius of the bundle; automatically detected if not provided.
        filterSize : float
            Sigma value for Gaussian filtering during core extraction (default is 0, meaning no filtering).
        background : ndarray
            Background image for background subtraction during reconstruction.
        normalise : ndarray
            Normalization reference image; if provided, core intensities are normalized.
        autoMask : bool
            Whether to automatically mask areas outside the fiber bundle (default True).
        mask : bool
            Whether to apply a circular mask to the final reconstructed image (default True).
        whiteBalance : bool
            If True, each color channel will be normalized individually (default False).
    """

    # Read keyword arguments
    centreX = kwargs.get('centreX', None)
    centreY = kwargs.get('centreY', None)
    radius = kwargs.get('radius', None)
    filterSize = kwargs.get('filterSize', 0)
    normalise = kwargs.get('normalise', None)
    autoMask = kwargs.get('autoMask', True)
    mask = kwargs.get('mask', True)
    background = kwargs.get('background', None)
    whiteBalance = kwargs.get('whiteBalance', False)

    if autoMask:
        img = pybundle.auto_mask(img, radius=radius)

    # Find core centers in the calibration image
    coreX, coreY = pybundle.find_cores(img, coreSize)
    coreX = np.round(coreX).astype('uint16')
    coreY = np.round(coreY).astype('uint16')

    # Estimate bundle center and radius if not provided
    if centreX is None:
        centreX = np.mean(coreX)
    if centreY is None:
        centreY = np.mean(coreY)
    if radius is None:
        dist = np.sqrt((coreX - centreX) ** 2 + (coreY - centreY) ** 2)
        radius = max(dist)

    calib = init_spline_interp(img, coreX, coreY, centreX, centreY, radius, gridSize,
                               whiteBalance=whiteBalance, filterSize=filterSize,
                               background=background, normalise=normalise, mask=mask)

    calib.nCores = np.shape(coreX)

    return calib


def init_spline_interp(img, coreX, coreY, centreX, centreY, radius, gridSize, **kwargs):
    """
    Initializes the calibration for spline interpolation, including fiber core detection
    and bundle geometry setup.

    Parameters:
        img : ndarray
            Calibration image (2D grayscale or 3D color).
        coreX, coreY : ndarray
            X and Y coordinates of fiber cores.
        centreX, centreY : float
            Center of the fiber bundle.
        radius : float
            Radius of the fiber bundle.
        gridSize : int
            Output image size (square).

    Optional Keyword Arguments:
        filterSize : float
            Sigma value for optional Gaussian filtering (default None).
        background : ndarray
            Background image for subtraction (optional).
        normalise : ndarray
            Image used for normalization (optional).
        mask : bool
            Whether to apply a circular mask to the output image (default True).
        whiteBalance : bool
            Whether to normalize each color channel separately (default False).
    """

    filterSize = kwargs.get('filterSize', None)
    normalise = kwargs.get('normalise', None)
    background = kwargs.get('background', None)
    mask = kwargs.get('mask', True)
    whiteBalance = kwargs.get('whiteBalance', False)

    # Detect if the image is color
    col = img.ndim > 2

    # Extract core intensity values for normalization and background subtraction
    if normalise is not None:
        normaliseVals = pybundle.core_values(normalise, coreX, coreY, filterSize).astype('double')
        if col and not whiteBalance:
            normaliseVals = np.mean(normaliseVals, 1)
            normaliseVals = np.expand_dims(normaliseVals, 1)
    else:
        normaliseVals = 0

    if background is not None:
        backgroundVals = pybundle.core_values(background, coreX, coreY, filterSize).astype('double')
    else:
        backgroundVals = 0

    # Initialize BundleCalibration object
    calib = BundleCalibration()
    calib.col = col
    if calib.col:
        calib.nChannels = np.shape(img)[2]

    calib.radius = radius
    calib.coreX = coreX
    calib.coreY = coreY
    calib.gridSize = gridSize
    calib.filterSize = filterSize
    calib.normalise = normalise
    calib.normaliseVals = normaliseVals
    calib.background = background
    calib.backgroundVals = backgroundVals
    calib.nCores = np.shape(coreX)[0]

    if mask:
        calib.mask = pybundle.get_mask(np.zeros((gridSize, gridSize)),
                                       (gridSize / 2, gridSize / 2, gridSize / 2))
    else:
        calib.mask = None

    return calib


def recon_spline_interp(img, calib, **kwargs):
    """
    Performs fiber bundle image reconstruction using cubic spline interpolation,
    based on prior calibration.

    Parameters:
        img : ndarray
            Raw fiber bundle image (grayscale 2D or color 3D).
        calib : BundleCalibration
            Calibration data generated by calib_spline_interp().

    Optional Keyword Arguments:
        coreSize : float
            Estimated core size, default is 3 (only used if filtering is applied).

    Returns:
        Reconstructed image as ndarray (2D or 3D).
    """

    coreSize = kwargs.get('coreSize', 3)

    # Extract core intensities from the raw image
    cVals = pybundle.core_values(img, calib.coreX, calib.coreY, calib.filterSize, **kwargs).astype('float64')

    # Apply background subtraction if provided
    if calib.background is not None:
        cVals = cVals - calib.backgroundVals

    # Apply normalization if provided
    if calib.normalise is not None:
        cVals = (cVals / calib.normaliseVals * 255)

    coreX, coreY = calib.coreX, calib.coreY
    centreX = np.mean(coreX)
    centreY = np.mean(coreY)
    radius = calib.radius
    gridSize = calib.gridSize

    # Generate the interpolation grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(centreX - radius, centreX + radius, gridSize),
        np.linspace(centreY - radius, centreY + radius, gridSize)
    )

    # Perform cubic spline interpolation
    interpolated_img = griddata(
        points=(coreX, coreY),
        values=cVals,
        xi=(grid_x, grid_y),
        method='cubic'
    )

    pixelVal = interpolated_img

    # Reshape the interpolated values into an image
    if calib.col:
        pixelVal = np.reshape(pixelVal, (calib.gridSize, calib.gridSize, calib.nChannels))
    else:
        pixelVal = np.reshape(pixelVal, (calib.gridSize, calib.gridSize))

    return pixelVal
