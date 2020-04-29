# Generating random curves

Use the `CurveGenerator_Matt.ipynb` file to generate and test curves.

1. Use `genNCurves()` to write out `n` curves, defined by their coefficients in the directory `curves`. Each curve has an ID number between `0` and `n-1` to refer to it.
2. Use `captureTestFrCurves()` and `captureTestResCurves()` to capture each curve with varying frame rates and resolutions respectively. These functions will write out the following:
  - In the directories `frame_rate_tests` and `resolution_tests`, the `.json` files needed by the locomotion package will be generated. Each file with the name `CRV_XX.json` contains a list of the required information for the curve with ID number `XX`.
  - In the directory `curve_testing`, files with the naming convention `PREFIX_CRV_XX_FR_YY_RES_ZZ.dat` will be stored. These files will only have an `X` and `Y` column. The prefix will be either `FR_TEST_` or `RES_TEST`, indicating which variable we are testing for. `XX`, `YY` and `ZZ` are the curve ID number, frame rate and resolution the curve is captured in respectively.
    - The directory `summary_stats` contains summary statistics for each curve.
    - The directories `frame_rate_tests_data` and `resolution_tests_data` contain the data files for the respective variable being tested.

# Testing for Robustness

Use the `robustness_testing.ipynb` in the `locomotion` directory above this to execute the robustness tests. They have been written in a way similar to the installation checks.

The results for frame rate and resolution tests can be seen in `frame_rate_results` and `resolution_results` respectively, where the heatmaps are in both csv and html formats for each curve.
