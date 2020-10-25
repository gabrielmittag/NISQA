
# NISQA - Non-Intrusive Speech Quality Assessment

We are currently working on updating the repository to make training and evaluating of the NISQA speech quality model easy. We will also supply a pretrained model and a speech quality dataset. For now, the TTS Naturalness prediction model and the old NISQA v0.4 model are available. The TTS Naturalness model uses the same neural network as the NISQA speech quality model that predicts quality of degraded speech from communication systems. However, the TTS model is trained to predict the Naturalness of synthesised speech. NISQA v0.4 predicts the quality of natural speech that was sent through a communication channel (e.g. VoIP call).

## G. Mittag and S. Möller, “Deep Learning Based Assessment of Synthetic Speech Naturalness,” in Interspeech 2020, 2020, pp. 1–5.

To use NISQA to estimate the Naturalness of Text-To-Speech synthesised speech, the provided weights can be used as follows (note that the weights cannot be used for commercial purposes, see license below):

### Requirements

To install requirements install Anaconda and then use:

```setup
conda env create -f env.yml
```

This will create a new enviroment with the name "nisqa". Activate this enviroment to go on:

```setup2
conda activate nisqa
```

### Naturalness prediction
There are three modes available to predict the speech of synthesised speech:
* Predict a single file
* Predict all files in a folder
* Predict all files in a CSV table

To predict the Naturalness of a single .wav file use:
```single file
python run_nisqa.py --mode predict_file --pretrained_model nisqa_tts.tar --deg /path/to/wav/file.wav --output_dir /path/to/dir/with/results
```
To predict the Naturalness of all .wav files in a folder use:
```single file
python run_nisqa.py --mode predict_dir --pretrained_model nisqa_tts.tar --deg_dir /path/to/folder/with/wavs --num_workers 0 --bs 10 --output_dir /path/to/dir/with/results
```

To predict the Naturalness of all .wav files listed in a csv table use:
```single file
python run_nisqa.py --mode predict_csv --pretrained_model nisqa_tts.tar --csv_file files.csv --csv_deg column_with_filename --csv_dir column_with_folder --num_workers 0 --bs 10 --output_dir /path/to/dir/with/results
```

The results will be printed to the console and saved to a csv file in a given folder (optional with --output_dir). To speed up the prediction, the number of workers and batch size of the Pytorch Dataloader can be increased (optional with --num_workers and --bs).

## NISQA v0.4 (to be updated soon)
NISQA is a speech quality prediction model for super-wideband communication networks. It is a *non-intrusive* or *single-ended model*, which means that it relies on the degraded output speech signal only, without the need of a clean reference. 


<img src="https://github.com/gabrielmittag/NISQA/blob/master/model.png" width="500">


More information about the model can be found here:

 - G. Mittag and S. Möller, "Non-intrusive Speech Quality Assessment for Super-wideband Speech Communication Networks," *ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Brighton, United Kingdom, 2019, pp. 7125-7129.
 
#### Windows command line tool
NISQA can be run as a command line tool in Windows 10 for research purposes (Win 10 needs to be updated to at least ver. 1809). The neural networks "nisqa_cnn.onnx" and "nisqa_lstm.onnx" must be placed in the folder from which you are running "nisqa.exe" from.

NISQA accepts .wav files with a duration of 5 to 12 seconds. The sample rate of the degraded speech files should be 48 kHz. NISQA can also upsample .wav files with sample rates of 8, 16, 32, and 44.1 kHz, however, the internal upsamling filter will affect the results.

The resulting MOS values represent the predicted quality in a super-wideband context. This means the bandwidth limitation of a clean narrowband or wideband signal will be considered as quality degradation by the model.

Please note that the model is still work in progress, if you notice unexpected behaviour for certain degradation conditions feel free to contact me: gabriel.mittag@tu-berlin.de

### Download
[NISQA v0.4.2](https://github.com/gabrielmittag/NISQA/releases/download/v0.4.2/nisqa.zip)

### Usage
To output the quality of a degraded speech file use: `nisqa.exe "path_to_speech_file.wav"`

To save the results in a .csv file use: `nisqa.exe "path_to_speech_file.wav" "path_to_csv_file.csv"`

## Licence
NISQA code is licensed under [MIT License](https://github.com/gabrielmittag/NISQA/blob/master/LICENSE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The model weights (nisqa_tts.tar) are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />

Copyright © 2019/2020 Quality and Usability Lab, TU Berlin  
www.qu.tu-berlin.de

