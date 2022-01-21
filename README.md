# Vietnamese Spelling Correction
## Introduction
This repo is the final project in my third year course: Natural Language Processing. Our goal is to build a wep app that can detect and correct incorrect Vietnamese phrase or sentences. Wrong phrases/sentences can be divided into 3 categories: typos, spelling errors or non-diacritic. 

## Enviroment
This project was trained out on a machine has:
- NVIDIA Tesla P100 16GB
- torch=1.8.2
- cudnn=8.2
- cudatoolkit=10.2
- ntlk=3.6.5
- numpy=1.20
- matplotlib=3.4

## Project structure

The project is generally structured as follows:
```
root/
├─ data/
│  ├─ corpus-title.txt
│  ├─ train_normal_captions.npy
│  ├─ valid_normal_captions.npy
│  ├─ list_5gram_nonum_valid.npy
│  ├─ list_5gram_nonum_train.npy
│  ├─ train_onehot_labels.npy
│  ├─ valid_onehot_labels.npy
├─ notebooks/
│  ├─ detector-hybrid.ipynb
│  ├─ paper-implemented.ipynb
│  ├─ vietnamese-spelling-correction-clean-data.ipynb
│  ├─ vietnamese-spelling-correction-seq2seq-lstm-attention.ipynb
│  ├─ vietnamese-spelling-correction-seq2seq-transformer.ipynb
│  ├─ vietnamese-spelling-correction-detector-model-lstm.ipynb
│  ├─ vietnamese-spelling-correction-detector-model-transformer-encoder.ipynb
├─ model/
│  ├─ model_transformer_40.pth
│  ├─ model_attention_40.pth
│  ├─ model_detector_lstm_1.pth
│  ├─ model_detector_transformer_1.pth
├─ py-scripts/
│  ├─ create_npy.py
│  ├─ create_dataset.py
│  ├─ file_processing.py
│  ├─ config.py
│  ├─ vocab.py
│  ├─ lstm_corrector.py
│  ├─ transformer_corrector.py
│  ├─ lstm_detector.py
│  ├─ transformer_detector.py
│  ├─ inference.py
│  ├─ load_dataset.py
│  ├─ synthesize_data.py
```

`root/` is where YOU clone this repo into. Therefore, you might want to rename `root/` to whatever you want. Here I use `root/` for convenience. There are 3 folders ignored by git because they contains large files. 
- `data/` - dataset used for training and testing
- `notebooks/` - all the notebooks used for training models
- `model/` - all the models saved after training
- `py_scripts/` -all the python scripts that I have written

## Dataset preparation
You can download directly from this link: https://drive.google.com/open?id=1ypvEoGRNWrNLmW246RtBm9iMyKXm_2BP if you want to create your own dataset
If you just want the data I use in this project, then download from this link: https://drive.google.com/drive/folders/1PafUx7PhTJwTEXgHPk9jp5iLrAl560gI?usp=sharing

Here is how to get the same data as mine:

	python run py_scripts/file_processing.py
	python run py_scripts/create_npy.py
Then run this notebook: vietnamese-spelling-correction-clean-data.ipynb(check carefully the directory and change if necessary)

## Training 
If you setup everything correctly, then you can press run all on Jupyter Notebook and no error will occur. Just remember to change the directory of the dataset to match with your local machine

## Inference
	streamlit run py_scripts/inference.py

## Result
| Model   | Dectection Precision | Dectection Recall | Dectection F1 | Correction Accuracy |
|----------|------------------------|--------------------|-----------------|-----------------------|
| Seq2Seq LSTM with attention(40 epochs) | 88.44% | 76.81% | 82.21% | 82.8%
|Seq2Seq-Attention Transformer(40 epochs) | 91.67% | 78.38% | 84.51% | 84.4%
|Hierarchical Transformer(40 epochs) |  | |  | 70%

The training time for correction model is nearly the same, around 16-17 minutes/epoch
The training time for lstm detection model is 14 minutes/epoch, whereas for transformer model is 12 minutes/epoch. 
I only train detection model for one epoch.
## Demo
![LSTM](https://github.com/ssjinkaido/Final-NLP-Project/blob/master/demo/demo_lstm.PNG)
![Transformer](https://github.com/ssjinkaido/Final-NLP-Project/blob/master/demo/demo_transformer.PNG)

## Reference
Most of my ideas come from [hisister97/Spelling_Correction_Vietnamese](https://github.com/hisiter97/Spelling_Correction_Vietnamese) and [huynhnhathao/vietnamese_spelling_error_correction](https://github.com/huynhnhathao/vietnamese_spelling_error_correction). Kudos to them!! 
Also here are the list of papers that I have read:
* [Do DT, Nguyen HT, Bui TN, Vo HD. PRICAI'21. VSEC: Transformer-based Model for Vietnamese Spelling Correction](https://arxiv.org/pdf/2111.00640.pdf)
* [Tran, Hieu, et al. IEA/AIE'21. Hierarchical Transformer Encoders for Vietnamese Spelling Correction](https://arxiv.org/pdf/2105.13578.pdf)
* [Dang TDA, Nguyen TTT. PACLIC'20. TDP – A Hybrid Diacritic Restoration with Transformer Decoder](https://aclanthology.org/2020.paclic-1.9.pdf)
* [Nguyen HT, Dang TB, Nguyen LM. PACLING'19. Deep learning approach for vietnamese consonant misspell correction](https://www.researchgate.net/publication/342620453_Deep_Learning_Approach_for_Vietnamese_Consonant_Misspell_Correction)
* [Hung BT. KSE'18. Vietnamese Diacritics Restoration Using Deep Learning Approach](https://www.researchgate.net/publication/329650270_Vietnamese_Diacritics_Restoration_Using_Deep_Learning_Approach)
* [Jakub et al. LREC'18. Diacritics Restoration Using Neural Networks](https://www.aclweb.org/anthology/L18-1247)
* [Pham et al. IALP'17. On the Use of Machine Translation-Based Approaches for Vietnamese Diacritic Restoration](https://arxiv.org/pdf/1709.07104.pdf)

## License & Copyright
You can do anything with the code, just cite this repo if you use it!



