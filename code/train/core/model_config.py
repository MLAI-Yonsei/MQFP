model_configuration = {"bcg": {"data_dim": 625, "last_dim": 256, "wavelet_dim":629, "spectro_dim": 297},
                       "ppgbp": {"data_dim": 262, "last_dim": 64, "wavelet_dim":265, "spectro_dim": 132},
                       "sensors": {"data_dim": 625, "last_dim": 512, "wavelet_dim":629, "spectro_dim": 351},
                       "vital_ecg": {"data_dim": 1250, "last_dim": 512, "wavelet_dim":1253, "spectro_dim": 627},
                       "mimic_ecg": {"data_dim": 1250, "last_dim": 512, "wavelet_dim":1253, "spectro_dim": 627},
                       "uci2": {"data_dim": 625, "last_dim": 256, "wavelet_dim":629}}

spectro_dim = {"bcg": {"ppgbp": 132, "sensors": 297},
               "ppgbp": {"bcg": 351, "sensors": 351},
               "sensors": {"bcg": 297, "ppgbp": 132},
               "vital_ecg": {"mimic_ecg": 627},
               "mimic_ecg": {"vital_ecg": 627}}