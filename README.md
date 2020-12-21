# RE-BERT: Automatic Extraction of Software Requirements from App Reviews using BERT Language Model

RE-BERT is a software requirements extractor from app reviews. The extraction of software requirements from app reviews can be defined as a token classification problem. We used the BIO format (short for beginning, inside, outside) to structure the training set reviews, as shown in the figure below. Note that the tokens for each sentence in the training set are labeled with class B, which indicates that the token represents the beginning of a software requirement; class I, which indicates that the token is inside a software requirement; and class O, which indicates that the token is outside a software requirement in the sentence.

<img src="https://raw.githubusercontent.com/adailtonaraujo/RE-BERT/main/img.PNG" alt="drawing" width="400"/>


### Using a pre-trained RE-BERT model

If you want to use our pre-trained RE-BERT to extract software requirements from app reviews, then start with the RE-BERT API demo code. We trained RE-BERT with eight app review datasets ([available here](https://github.com/jsdabrowski/CAiSE-20)) from different domains. BERT-based models usually have satisfactory generalizability, so you can use RE-BERT to extract software requirements for other apps. If the quality is not adequate for your tasks, see the section on training your own RE-BERT model.

* [Pre-trained RE-BERT Model (Notebook)](Pre_trained_RE_BERT_Model_API_Demo.ipynb)

### Training your own RE-BERT model

The first step in training your own model is to generate a training set in the BIO format. ([Read more details about the training file format](datasets_iob)). We provide a notebook with a step-by-step guide to conduct RE-BERT training. This notebook can be used to reproduce the experiments in the paper published in ACM SAC-RE 2021.

* [Training your own RE-BERT Model (Notebook)](Training_a_RE_BERT_model.ipynb)


### Contributions and Bug reports

RE-BERT is under development. There may be unknown problems in the code and we hope to get your help to make it easier to use.

### Related Repositories

* [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch)
* [LC-ABSA](https://github.com/yangheng95/LC-ABSA/)
* [CAiSE-20](https://github.com/jsdabrowski/CAiSE-20)
