#!/usr/bin/env bash
# python BraTS_2014_data_convert_to_nifti.py
# python BraTS_2014_data_do_bc.py
python BraTS_2014_data_do_ws_no_bc.py
python BraTS_2014_data_convert_bc_ws_to_npy_no_bc.py
python BraTS_2014_data_extract_brain_region_no_bc.py
python BraTS_2014_data_adapt_value_range_no_bc.py
