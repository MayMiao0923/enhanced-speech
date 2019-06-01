'''
SUMMARY:  config file
AUTHOR:   YONG XU
Created:  2017.12.20
Modified: 
--------------------------------------
'''
ori_dev_root ='/vol/vssp/msos/yx/source_separation/enhSpec2wav' ### running folder 运行文件夹

dev_wav_fd = '/vol/vssp/AP_datasets/audio/dcase2016/timit_enh/wav'  ### fold where all wavs are there 折叠所有wavs都在那里

# temporary data folder
scrap_fd = "/vol/vssp/msos/yx/source_separation/enhSpec2wav/fea"    ### generated log-magnitude-fea parent folder 生成log-magnitude-fea父文件夹
dev_fe_mel_fd = scrap_fd + '/fe/log_mag_spec'    ### generated log-magnitude-fea folder 生成log-magnitude-fea文件夹

enh_fea_wav_fd = "/vol/vssp/msos/yx/source_separation/enhSpec2wav/enh_wavs" ### enhanced fea and wav 增强了fea和wav

model_fd = "/vol/vssp/msos/yx/source_separation/enhSpec2wav/fea/md" ### trained models folder训练好的模型文件夹

dev_cv_csv_path = ori_dev_root + '/file_list.csv'  ### file list "number id,file name,train (0) or test (1) flag", including train and test files
                                                    ## 文件列表“数字ID，文件名，训练（0）或测试（1）标志”，包括训练和测试文件
fs = 16000.  # sample rate 采样率
win = 512. #32ms ###win size 窗的大小
