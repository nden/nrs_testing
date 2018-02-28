import numpy as np
from astropy.io import fits
from gwcs.wcstools import grid_from_bounding_box
from jwst.assign_wcs import nirspec
from jwst import datamodels
from numpy.testing import assert_allclose


def test_pipeline_esa(slit_wcs, trace_file):
    """
    Compare the pipeline WCS against ESA reg test results.

    Parameters
    ----------
    slit_wcs : `gwcs.wcs.WCS`
        The WCS object of a slice.
    trace_file : str
        The name of the Trace file which corresponds to the slice.
    """
    # read the trace file and its WCS
    tr = fits.open(trace_file)
    slit1 = tr[5].data
    lam1 = tr[4].data
    msax = tr[6].data
    msay = tr[7].data

    crpix1 = tr[5].header['crpix1']
    crpix2 = tr[5].header['crpix2']
    crval1 = tr[5].header['crval1']
    crval2 = tr[5].header['crval2']
    slit_id = tr[0].header['SLITID']
    tr.close()
    # Get x, y indices within the slit.
    # The slit is taken between -0.5 and 0.5 with 0.0 in the center of the slit.
    cond = np.logical_and(slit1 < .5, slit1 > -.5)
    y, x = cond.nonzero()


    # compute the input indices using the WCS
    # x,y are 0-based, add +1 to go to FITS coords.
    x = x - crpix1 + crval1 + 1 # +1 to 1-based FITS coords
    y = y - crpix2 + crval2 + 1

    # call the pipeline WCS
    r, d, l = slit_wcs(x, y)

    # get the difference
    diff = lam1[cond] - l * 10**-6

    # get the transform from detector to slit_frame
    det2slit = slit_wcs.get_transform('detector', 'slit_frame')
    det2msa = slit_wcs.get_transform('detector', 'msa_frame')

    slitx, slity, _ = det2slit(x, y)
    slit_diff = slit1[cond] - slity
    pmsa_x, pmsa_y, _ = det2msa(x, y)
    msax_diff = msax[cond] - pmsa_x
    msay_diff = msay[cond] - pmsa_y

    print('Slit {0}'.format(slit_id))
    try:
        assert_allclose(lam1[cond], l*10**-6, atol=10**-13)
        print('\t Max diff in wavelength: ', np.abs(diff).max())
        print('\t Mean diff in wavelength: ', np.abs(diff).mean())
    except AssertionError:
        not_close = np.isnan(diff).nonzero()
        close = (~np.isnan(diff)).nonzero()
        print('\t Number of pixels with NaN values in the diff: ', len(not_close[0]))
        print('\t Number of pixels with values of <= 10**-13 in the diff: ', len(l[close]))
        print('\t Max diff in wavelength: ', np.abs(diff[close]).max())
        print('\t Mean diff in wavelength: ', np.abs(diff[close]).mean())
    try:
        assert_allclose(slit1[cond], slity, atol=10**-13)
        print('\t Max diff along slit y-axis: ', np.abs(slit_diff).max())
    except AssertionError:
        not_close = np.isnan(slit_diff).nonzero()
        close = (~np.isnan(slit_diff)).nonzero()
        #print('Number of pixels with NaN values in the diff: ', len(not_close[0]))
        #print('Number of pixels with values of <= 10**-13 in the diff: ', len(l[close]))
        print('\t Max diff along slit y-axis: ', np.abs(slit_diff[close]).max())

    try:
        assert_allclose(msax[cond], pmsa_x, atol=10**-13)
        print('\t Max diff in MSA x coordinate: ', np.abs(msax_diff).max())
        print('\t Mean diff in MSA x coordinate: ', np.abs(msax_diff).mean())
    except AssertionError:
        not_close = np.isnan(msax_diff).nonzero()
        close = (~np.isnan(msax_diff)).nonzero()
        #print('Number of pixels with NaN values in the diff: ', len(not_close[0]))
        #print('Number of pixels with values of <= 10**-13 in the diff: ', len(l[close]))
        print('\t Max diff in MSA x coordinate: ', np.abs(msax_diff[close]).max())
        print('\t Mean diff in MSA x coordinate: ', np.abs(msax_diff[close]).mean())
    try:
        assert_allclose(msay[cond], pmsa_y, atol=10**-13)
        print('\t Max diff in MSA x coordinate: ', np.abs(msay_diff).max())
        print('\t Mean diff in MSA x coordinate: ', np.abs(msay_diff).mean())
    except AssertionError:
        not_close = np.isnan(msay_diff).nonzero()
        close = (~np.isnan(msay_diff)).nonzero()
        #print('Number of pixels with NaN values in the diff: ', len(not_close[0]))
        #print('Number of pixels with values of <= 10**-13 in the diff: ', len(l[close]))
        print('\t Max diff in MSA y coordinate: '.format(slit_id), np.abs(msay_diff[close]).max())
        print('\t Mean diff in MSA y coordinate: '.format(slit_id), np.abs(msay_diff[close]).mean())
    return diff
