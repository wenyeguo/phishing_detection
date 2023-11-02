from collections import Counter
from urllib.parse import urlparse
import os
import glob
import numpy as np
import pandas as pd

def read_data():
	df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', '*.csv'))))
	X = df["URL"].values
	Y = df["Productivity"].values
	Y = Y.astype('int')
	print("read data",Y.shape)
	return X, Y

# include http
def IP_address(X):
	res = []
	for i in range(len(X)):
		domain = urlparse(X[i]).netloc
		status = not any(c.isalpha() for c in domain)
		if status:
			status = 1
		else:
			status = 0
		res.append(status)
	return res

def ratio_of_domain_length(X):
	res = []
	for i in range(len(X)):
		item = X[i]
		domain = urlparse(item).netloc
		# print(item, domain,  len(domain)/len(item))
		res.append(float("%0.3f"% (len(domain)/len(item))))
	return res

# exclude http
def specific_char_occurence(X):
	# @ and - occurence
	res = []
	chars = set('@-')
	for item in X:
		char_in_URL = Counter(item)
		count = 0
		for c in chars:
			count += char_in_URL[c]
		res.append(count)
	return res

def num_of_punctuation(X):
	#  . ! # $ % &*,;:’
	res = []
	chars = set('.!#$%&*,;:’')
	for i in range(len(X)):
		item = X[i]
		char_in_URL = Counter(item)
		count = 0
		for c in chars:
			count += char_in_URL[c]
		res.append(count)
	# print(len(res))
	# print(count)
	
	return res


def num_of_suspicious_words(X):
	suspicious = ["confirm","account","secure","ebayisapi","webscr","login","signin","submit","update"]
	res = []
	for i in range(len(X)):
		count = 0
		for s in suspicious:
			if s in X[i]:
				count += 1
		res.append(count)	
	return res


# char_freq_of_url each url
def char_freq_of_url(url):
	letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
	res = {}
	char_in_URL = Counter(url)
	sum = 0
	for letter in letters:
		sum += char_in_URL[letter]
		if letter in char_in_URL:
				res[letter] = char_in_URL[letter]
		else:
			res[letter] = 0
	for letter in letters:	
		if sum == 0:
			res[letter] = 0
		else:
			res[letter] = float("%0.5f"% (res[letter]/sum))
	res_list = list(res.values())
	return res_list

def standard_english_freq():
	freq = [8.167, 1.492,2.782,4.253,12.702,2.228,2.015,6.094,6.966,0.153,0.772,4.025,2.406,6.749,7.507,1.929,0.095,5.987,6.327,9.056,2.758,0.978,2.360,0.150,1.974,0.074]
	for i in range(len(freq)):
		freq[i] = float("%0.5f"% (freq[i]/100))
	return freq

from scipy import stats
def KS_distance(X):
	res = []
	for item in X:
		pdf = char_freq_of_url(item)
		cdf = np.cumsum(pdf)
		standard_cdf = np.cumsum(standard_english_freq())
		s = stats.mstats.ks_twosamp(cdf,standard_cdf).statistic
		res.append(float("%0.3f"% s))
	# print(res)
	return res

# P - standard, Q - URL
def KL(P, Q):
	P = np.array(P)
	Q = np.array(Q)
	idx = []
	for i in range(len(P)):
		if Q[i] == 0:
			idx.append(i)
	# print(idx)
	new_P = np.delete(P, idx)
	new_Q = np.delete(Q, idx)	
	# print(new_P)
	# print(new_Q)	
	new_P = new_P/np.sum(new_P) 
	new_Q = new_Q/np.sum(new_Q)
	d = np.sum(new_P * np.log(new_P/new_Q))
	# print(d)
	# sum = np.sum(d)
	return d	

def KL_Divergence(X):
	standard = standard_english_freq()
	res = []
	for i in range(len(X)):
		url = char_freq_of_url(X[i])
		d =  KL(standard, url)
		# print(i, d)
		res.append(d)
	return res

def Euclidean_Distance(X):
	standard = standard_english_freq()
	# standard = np.array(standard)
	res = []
	for i in range(len(X)):
		url = char_freq_of_url(X[i])
		# url = np.array(url)
		dist = math.dist(standard - url)
		res.append(float("%0.3f"% dist))
	# print("Euclidean_Distance", res)
	return res	


def char_freq(X):
	freq = []
	letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
	for item in X:
		res = {}
		char_in_URL = Counter(item)
		sum = len(item)
		for letter in letters:
			if letter in char_in_URL:
				res[letter] = char_in_URL[letter]
			else:
				res[letter] = 0
			res[letter] = float("%0.3f"% (res[letter]/sum))
		res_list = list(res.values()) 
		# print(res_list)
		freq.append(res_list)
		# print(freq)

	return freq

# remove http part
def remove_prefix(X):
	for i in range(len(X)):
		X[i] = X[i].removeprefix("http://")
		X[i] = X[i].removeprefix("https://")
	return X

# normalize single feature
def normalization(res):
	max_value = max(res)
	min_value = min(res)
	for i in range(len(res)):
		res[i] = float("%0.3f"% ((res[i] - min_value)/(max_value - min_value)))
	return res

# write features to file
import csv
def write_to_file(X, file_name):
	# print(X)
	with open(file_name, "w", newline="") as file:
		writer = csv.writer(file)
		writer.writerows(map(lambda x: [x], X))

def write_to_file_vector(X, file_name):
	with open(file_name, "w", newline="") as file:
		writer = csv.writer(file)
		writer.writerows(X)

# normalize vector
def normalization_vector():
	res = []
	df = read_features("data/1/charFreq.csv")
	letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
	for letter in letters:
		l = df[letter].values
		l = l.tolist()
		new_l = normalization(l)	
		res.append(new_l)
	return res

def read_features(file):
	df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', file))))
	df = df.dropna() 
	X = df.drop(labels = ['f_KL', 'f_KS', 'f_ED'], axis=1)
	# X = df
	return X

def transport(s):
	return np.array(s).T.tolist()

# # write normalized features to csv
# X, Y = read_data()
# print(X.shape)
# f_KL = normalization(KL_Divergence(remove_prefix(X)))
# f_KS = normalization(KS_distance(remove_prefix(X)))
# f_ED = normalization(Euclidean_Distance(remove_prefix(X)))
# f_specific = normalization(specific_char_occurence(remove_prefix(X)))
# f_punc = normalization(num_of_punctuation(remove_prefix(X)))
# f_suspicious = normalization(num_of_suspicious_words(remove_prefix(X)))
# f_IP = normalization(IP_address(X))
# f_domain = normalization(ratio_of_domain_length(X))
# list = normalization_vector()
# list.append(f_KL)
# list.append(f_KS)
# list.append(f_ED)
# list.append(f_specific)
# list.append(f_punc)
# list.append(f_suspicious)
# list.append(f_IP)
# list.append(f_domain)
# list = transport(list)
# print('row',len(list))
# print('col', len(list[0]))
# # print(list)
# with open('data/features_total.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","f_KL","f_KS", "f_ED", "f_specific", "f_punc","f_suspicious","f_IP","f_domain"])
#     writer.writerows(list)


# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import cross_val_score,cross_val_predict
# from sklearn import metrics, svm
# # # # main train clf
# URL, Y = read_data()
# # print(X.shape)
# # print(Y.shape)
# X = read_features("data/features_total.csv")
# print(X.shape)



# # x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state =0)
# # clf = GaussianNB()
# # clf = LogisticRegression(solver='lbfgs', max_iter=1000)
# # clf = RandomForestClassifier()

# scores = cross_val_score(clf, X, Y,cv=5)
# # print("GaussianNB clf, 5-fold cross validation")
# # print("LogisticRegression clf, 5-fold cross validation")
# print("RandomForest clf, 5-fold cross validation")


# print("Accuracy", scores)
# print(f"Average accuracy = {np.mean(scores):.3f}")
# print(f"Average Precision = {np.mean(cross_val_score(clf,X,Y,cv=5,scoring='precision')):.3f}")
# print(f"Average Recall = {np.mean(cross_val_score(clf,X,Y,cv=5,scoring='recall')):.3f}")
# print(f"Average F1 = {np.mean(cross_val_score(clf,X,Y,cv=5,scoring='f1')):.3f}")

# predicted = cross_val_predict(LogisticRegression(solver='lbfgs', max_iter=1000), X, Y, cv=5)
import math
X, Y = read_data()
res = normalization(Euclidean_Distance(remove_prefix(X)))
write_to_file(res, "data/EU.csv")
# print(metrics.accuracy_score(Y, predicted))
