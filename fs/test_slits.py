from astropy.io import fits
from jwst.assign_wcs import nirspec
from jwst import datamodels
from numpy.testing import assert_allclose
from test_nrs import test_pipeline_esa


map_slit_names = {'SLIT_A_1600' : 'S1600A1',
                  'SLIT_A_200_1': 'S200A1',
                  'SLIT_A_200_2': 'S200A2',
                  'SLIT_A_400':   'S400A1',
                  #'SLIT_B_200':   'S200B1'
                  }


def test_fixed_slit(pipeline_file, esa_trace_list):
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
        s = fits.getval(f, 'SLITID')
        slit = map_slit_names[s]
        print('\nRunning test for slit {0}'.format(slit))
        im = datamodels.ImageModel(pipeline_file)
        w = nirspec.nrs_wcs_set_input(im, slit)
        test_pipeline_esa(w, f)
