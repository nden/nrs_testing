import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import wcs
from test_nrs import test_pipeline_esa
from jwst import datamodels
from jwst.assign_wcs import nirspec


def test_slits_esa_coords(pipeline_file, esa_trace_list):
    """
    Get detector coordinates from the ESA trace files and
    evaluate the pipeline WCS on them. Compare wavelength results.

    Parameters
    ----------
    pipeline_file : str
        Output of assign_wcs.
    esa_trace_list : list
        List of Trace* file names
    """
    for f in esa_trace_list:
        s = fits.getval(f, 'SLICEID')
        print('\nRunning test for slice {0}'.format(s))
        # Open the file as a data model.
        im = datamodels.ImageModel(pipeline_file)
        # get the WCS object of this slice
        w = nirspec.nrs_wcs_set_input(im, s)
        # test_pipeline_esa returns the difference
        # between the wavelengths from the pipeline and the ESA trace file.
        diff = test_pipeline_esa(w, f)


def test_wavelength(slice_wcs, esa_trace):
    """
    Get detector coordinates from the ESA trace files and
    evaluate the pipeline WCS on them. Compare wavelength results.

    Parameters
    ----------
    slice_wcs : `gwcs.wcs.WCS`
        The pipeline WCS object for this slice
    esa_trace : str
        The ESA Trace file for this slice
    """

    # Open the Trace file, read in the wavelength extension
    # and its FITS WCS.
    tr = fits.open(esa_trace)
    s = tr[0].header['SLICEID']
    lam_esa = tr[4].data
    esa_w = wcs.WCS(tr[4].header)
    tr.close()

    # Compute the (x, y) pixel positions corresponding to the wavelength array
    y, x = np.mgrid[:lam_esa.shape[0], :lam_esa.shape[1]]
    x1, y1 = esa_w.all_pix2world(x + 1, y + 1, 1) # add +1 to convert to FITS coordinates

    # Compute RA, DEC, lam using the pipeline WCS object
    r, d, l = slice_wcs(x1, y1)

    # Plot the difference of the two wavelength arrays
    diff = l * 10**-6 - lam_esa
    return diff
