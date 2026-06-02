# arousal_score_audio

rows: 40
target_mean: 0.468
split: holdout

## Ridge Regression

mae: 0.292
r2: -1.665

## Random Forest Regressor

mae: 0.164
r2: -0.247

## Extra Trees Regressor

mae: 0.160
r2: -0.215

## Gradient Boosting Regressor

mae: 0.153
r2: -0.026

## Svr Rbf

mae: 0.154
r2: -0.088

## Knn Regressor

mae: 0.158
r2: -0.159

## Mlp Regressor

mae: 0.374
r2: -3.842

## Random forest feature importance

                feature  importance
          chroma_mean_1    0.139415
            mfcc_std_10    0.038173
         chroma_mean_12    0.036982
          chroma_mean_3    0.033336
      loudness_db_range    0.030743
        loudness_db_std    0.030669
 spectral_bandwidth_std    0.028378
          chroma_mean_6    0.027505
           mfcc_mean_12    0.024832
        mode_confidence    0.024452
spectral_bandwidth_mean    0.023628
            mfcc_mean_8    0.019780
  major_key_correlation    0.019123
       rms_energy_slope    0.018715
          chroma_mean_2    0.017278

## Extra trees feature importance

               feature  importance
         chroma_mean_1    0.041887
        chroma_mean_12    0.029628
     loudness_db_range    0.026880
       loudness_db_std    0.025637
       key_pitch_class    0.023220
         chroma_mean_3    0.022068
spectral_bandwidth_std    0.020361
        rms_energy_std    0.019752
           mfcc_mean_5    0.019303
           mfcc_std_11    0.018258
      rms_energy_range    0.018245
       loudness_db_p90    0.017893
       mode_confidence    0.017800
         chroma_mean_2    0.017782
          mfcc_mean_12    0.016710
