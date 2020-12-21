import numpy as np
import pandas as pd

def feature_matching(f_pred, f_true): # f_* are arrays of tokens


  if len(f_pred) == 0: return (False,0)
  if len(f_true) == 0: return (False,0)


  n = np.abs(len(f_pred)-len(f_true))

  test_list = f_true
  sub_list = f_pred
  if len(f_pred) > len(f_true):
    test_list = f_pred
    sub_list = f_true

  subset = False
  if(set(sub_list).issubset(set(test_list))): 
      subset = True

  return subset,n


def metrics(review_f_pred,review_f_true,n): # list of predicted features

  TP = 0
  FP = 0
  FN = 0


  # case 1 (1 predicted and 1 labeled)
  if len(review_f_pred) == len(review_f_true) and len(review_f_true)==1:
    f_pred = review_f_pred[0]
    f_true = review_f_true[0]
    matching = feature_matching(f_pred,f_true)
    if matching[0] == False or matching[1] > n:
      FP+=1
      FN+=1
    if matching[0] == True and matching[1] <= n: TP+=1
    return (TP,FP,FN)

  # case 2 (x predicted and y labeled, with x > 1, and y > 1)
  ## computing TP
  for f_pred in review_f_pred:
    for f_true in review_f_true:
      matching = feature_matching(f_pred,f_true)
      if matching[0] == True and matching[1] <= n: TP+=1

  ## computing FP
  for f_pred in review_f_pred:
    FP_flag = 1
    for f_true in review_f_true:
      matching = feature_matching(f_pred,f_true)
      if matching[0] == True and matching[1] <= n:
        FP_flag = 0
        break # try next
    if FP_flag == 1: FP+=1

  ## computing FN
  for f_true in review_f_true:
    FN_flag = 1
    for f_pred in review_f_pred:
      matching = feature_matching(f_pred,f_true)
      if matching[0] == True and matching[1] <= n:
        FN_flag = 0
        break # try next
    if FN_flag == 1: FN+=1

  return (TP,FP,FN)


def f1_measure(features_extracted, features_labeled, n): # list of extracted features from reviews datadaset
  counter = 0

  if len(features_extracted)!=len(features_labeled):
    return None

  eval_statistics = []
  for counter in range(0,len(features_extracted)):
    txt_features_extracted = features_extracted[counter].split(';')
    txt_features_labeled = features_labeled[counter].split(';')

    review_f_pred = []
    review_f_true = []
    for item in txt_features_extracted: review_f_pred.append( item.split(' ') )
    for item in txt_features_labeled: review_f_true.append( item.split(' ') )


    metric = metrics(review_f_pred, review_f_true, n)
    eval_statistics.append(metric)

    counter += 1

  metrics_sum = pd.DataFrame(eval_statistics).rename(columns={0:'TP',1:'FP',2:'FN'}).sum()
  TP = metrics_sum['TP']
  FP = metrics_sum['FP']
  FN = metrics_sum['FN']

  precision = TP / (TP+FP)
  recall = TP / (TP+FN)
  f1 = (2*precision*recall) / (precision+recall)

  return precision,recall,f1
