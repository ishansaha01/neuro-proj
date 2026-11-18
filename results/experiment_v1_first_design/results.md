# Reconstruction experiment

Dataset `outputs/demo_sweep`, 8 scenes, seed 0; mean ± standard deviation over scenes. Run time 19.5 min.

| method | decoding_accuracy | activation_dice | cbv_correlation | activation_cnr | coverage_0.5 | coverage_0.8 | coverage_0.9 | seconds |
|---|---|---|---|---|---|---|---|---|
| pd_skull_das | 0.628 ± 0.098 | 0.017 ± 0.032 | 0.045 ± 0.174 | 0.466 ± 0.389 |  |  |  | 0.000 ± 0.000 |
| pd_skull_ct | 0.587 ± 0.094 | 0.147 ± 0.108 | 0.316 ± 0.191 | 0.965 ± 0.404 |  |  |  | 0.000 ± 0.000 |
| pd_skull_true | 0.597 ± 0.089 | 0.247 ± 0.120 | 0.595 ± 0.408 | 0.967 ± 0.420 |  |  |  | 0.000 ± 0.000 |
| pd_clean_das | 0.666 ± 0.038 | 0.287 ± 0.115 | 0.912 ± 0.034 | 1.062 ± 0.080 |  |  |  | 0.000 ± 0.000 |
| deconv_true_h | 0.622 ± 0.082 | 0.111 ± 0.089 | 0.713 ± 0.287 | 0.854 ± 0.305 |  |  |  | 0.032 ± 0.011 |
| deconv_ct | 0.659 ± 0.104 | 0.142 ± 0.110 | 0.218 ± 0.129 | 0.878 ± 0.289 |  |  |  | 10.261 ± 2.856 |
| deconv_calibrated | 0.650 ± 0.102 | 0.139 ± 0.117 | 0.217 ± 0.127 | 0.873 ± 0.287 |  |  |  | 66.021 ± 15.753 |
| dps_true_h | 0.544 ± 0.105 | 0.128 ± 0.152 | 0.689 ± 0.264 | 0.701 ± 0.326 | 0.397 ± 0.053 | 0.704 ± 0.049 | 0.832 ± 0.034 | 18.797 ± 3.764 |
| dps_ct | 0.425 ± 0.106 | 0.000 ± 0.000 | 0.073 ± 0.050 | 0.054 ± 0.414 | 0.380 ± 0.146 | 0.710 ± 0.132 | 0.841 ± 0.074 | 28.408 ± 6.023 |
| dps_calibrated | 0.656 ± 0.080 | 0.106 ± 0.104 | 0.214 ± 0.125 | 0.750 ± 0.519 | 0.188 ± 0.184 | 0.348 ± 0.269 | 0.453 ± 0.272 | 84.476 ± 18.997 |
| prior_only | 0.528 ± 0.069 | 0.025 ± 0.071 | 0.039 ± 0.048 | 0.240 ± 0.340 |  |  |  | 3.055 ± 0.816 |

| screen | delay_error_ct_ns | log_amplitude_error_ct_mean | log_amplitude_error_ct_rms | delay_error_calibrated_ns | log_amplitude_error_calibrated_mean | log_amplitude_error_calibrated_rms |
|---|---|---|---|---|---|---|
| mean ± std | 68.688 ± 37.078 | -2.033 ± 0.127 | 2.035 ± 0.127 | 70.388 ± 37.125 | -0.230 ± 1.172 | 0.845 ± 0.794 |
