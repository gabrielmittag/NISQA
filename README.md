
# NISQA - Non-Intrusive Speech Quality Assessment
NISQA is a speech quality prediction model for super-wideband communication networks. It is a *non-intrusive* or *single-ended model*, which means that it relies on the degraded output speech signal only, without the need of a clean reference. 


<img src="https://github.com/gabrielmittag/NISQA/blob/master/model.png" width="500">


More information about the model can be found here:

 - G. Mittag and S. Möller, "Non-intrusive Speech Quality Assessment for Super-wideband Speech Communication Networks," *ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Brighton, United Kingdom, 2019, pp. 7125-7129.

## NISQA v0.4
NISQA can be run as a command line tool in Windows 10 for research purposes (Win 10 needs to be updated to at least ver. 1809). The neural networks "nisqa_cnn.onnx" and "nisqa_lstm.onnx" must be placed in the folder from which you are running "nisqa.exe" from.

NISQA accepts .wav files with a duration of 5 to 12 seconds. The sample rate of the degraded speech files should be 48 kHz. NISQA can also upsample .wav files with sample rates of 8, 16, 32, and 44.1 kHz, however, the internal upsamling filter will affect the results.

The resulting MOS values represent the predicted quality in a super-wideband context. This means the bandwidth limitation of a clean narrowband or wideband signal will be considered as quality degradation by the model.

Please note that the model is still work in progress, if you notice unexpected behaviour for certain degradation conditions feel free to contact me: gabriel.mittag@tu-berlin.de

### Download
[NISQA v0.4.1](https://github.com/gabrielmittag/NISQA/releases/download/v0.4.1/nisqa.zip)

### Usage
To output the quality of a degraded speech file use: `nisqa.exe "path_to_speech_file.wav"`

To save the results in a .csv file use: `nisqa.exe "path_to_speech_file.wav" "path_to_csv_file.csv"`

## Licence
NISQA is licensed under [GNU General Public License](https://github.com/gabrielmittag/NISQA/blob/master/LICENSE)


Copyright © 2019 Quality and Usability Lab, TU Berlin
www.qu.tu-berlin.de

