"""
Preliminary tests for a time-series classifier, that can be trained with sample data according to [1].
We train by building an as small as possible data dictionary which describes all the characteristic features of the data classes.

Convert the module into a class structure later.

[1] B. Hu, Y. Chen, and E. Keogh, SIAM 578 (2013)
"""

import numpy as n
import scipy.linalg as lin

"""
Calculate the square of the Euclidian distance between two vectors or between one vector and an array of vectors.
Pass two same-length vectors or a vector of length n and an array of m vectors of length n.
"""
def euclidian_dist_square(a, b): # a, b: either 1D-arrays of same length or one 1D-array of length n and one 2D-array with shape (m, n), where m is arbitrary
	diff = a - b
	return n.sum((diff*diff).T, axis = 0) # Transpose, so axis=0 can be used for both 1D and 2D diffs

"""
Calculate the distance of two vectors of, in general, different length.
Returns the lowest Euclidian distance of the test array among all the subsequences of the sample query q
"""
def nearest_dist(q, test): # q: Query vector, test: test vector to match
	bsf_dist = n.inf # best-so-far distance, initialize at infinity
	lq = len(q)
	lt = len(test)
	dists = n.zeros(abs(lq - lt) + 1)
	# dists = n.zeros(lq + lt - 1)
	# for i in range(1, lq + lt):
		# print(len(q[max(0,i-lt):min(i,lq)]), len(test[max(0,lt-i):min(lt,lt-i+lq)]))
		# new_dist = n.sqrt(euclidian_dist_square(q[max(0,i-lt):min(i,lq)], test[max(0,lt-i):min(lt,lt-i+lq)])) / (min(i,lq) - max(0,i-lt)) 
	if lt < lq:
		for i in range(lt, lq + 1): # if we know that lt < lq
			new_dist = n.sqrt(euclidian_dist_square(q[i-lt:i], test)) / lt
			dists[i-lt] = new_dist
			if new_dist < bsf_dist:
				bsf_dist = new_dist
	else:
		for i in range(lq, lt + 1): # if we know that lt >= lq
			new_dist = n.sqrt(euclidian_dist_square(q, test[lt-i:lt-i+lq])) / lq
			dists[i-lq] = new_dist
			if new_dist < bsf_dist:
				bsf_dist = new_dist
	return bsf_dist, dists

"""
Classify query q using the data dictionary D.
"""
def classify(q, D, r):
	# Calculate nearest neighbor within D
	bsf_nn_dist = n.inf # best-so-far distance, initialize at infinity
	class_prediction = -1 # current best estimate for the class of q, initialize at 'other', labeled by -1
	for c_ind, cl in enumerate(D): # iterate over classes in D
		for vec in cl: # iterate over vectors that belong to the current class
			new_dist, dists = nearest_dist(q, vec)
			if new_dist < bsf_nn_dist:
				bsf_nn_dist = new_dist
				class_prediction = c_ind # assign q the class of the (current) nearest neighbor. QUESTION: should it not be the nearest neighbor after summing all distances within each class?
	# if distance to nearest neighbor is below the rejection threshold r, return predicted class
	if bsf_nn_dist < r:
		return class_prediction
	# if the distance is larger than r, assign class 'other', labeled by -1
	else:
		return -1

def score(c, C, S): # c: tuple class index (class, position) of q, S: current scores
	# friends = C[c[0]][:c[1]:] # omit q when constructing the friends
	# enemy_classes = C[:c[0]:]
	print c
	friend_indices = [[c[0], j] for j in n.arange(len(C[c[0]]))[:c[1]:]]
	enemy_indices = []
	for i in n.arange(len(C))[:c[0]:]:
		enemy_indices += [[i, j] for j in xrange(len(C[i]))]

	bsf_nn_dist = n.inf
	friend_dists = n.zeros(len(friend_indices))
	for i, subseq_ind in enumerate(friend_indices):
		new_dist = n.sqrt(euclidian_dist_square(C[c[0]][c[1]], C[subseq_ind[0]][subseq_ind[1]]))
		friend_dists[i] = new_dist
		if new_dist < bsf_nn_dist:
			bsf_nn_dist = new_dist
	friend_nn_dist = bsf_nn_dist

	bsf_nn_dist = n.inf
	enemy_dists = n.zeros(len(enemy_indices))
	for i, subseq_ind in enumerate(enemy_indices):
		new_dist = n.sqrt(euclidian_dist_square(C[c[0]][c[1]], C[subseq_ind[0]][subseq_ind[1]]))
		enemy_dists[i] = new_dist
		if new_dist < bsf_nn_dist:
			bsf_nn_dist = new_dist
	enemy_nn_dist = bsf_nn_dist

	if friend_nn_dist < enemy_nn_dist:
		likely_true_positives = n.where(friend_dists < enemy_nn_dist)[0]
		for ltp in likely_true_positives:
			S[friend_indices[ltp][0]][friend_indices[ltp][1]] += 1. / friend_dists[ltp]
	else:
		likely_false_positives = n.where(enemy_dists < friend_nn_dist)[0]
		for lfp in likely_false_positives:
			S[enemy_indices[lfp][0]][enemy_indices[lfp][1]] -= 2./(len(C) - 1.) / enemy_dists[lfp]


"""
Train by creating a data dictionary.training_data: sequence of time series, one for each class. Indices in this sequence are the class labels. x: fraction of the data that should be in the dictionary. l: query length. Data dictionary entries will have length 2l because of padding. 2l should probably be well below the CHUNK of the stream. Play around with this.
"""
def train(training_data, x, l, maxqueries):
	
	N_train = len(training_data)
	N_subseq = sum([len(dat) for dat in training_data])
	maxDlength = N_subseq/float(l) * x
	print("maxDlength = {0}".format(maxDlength))
	# C: all subsequences of the training data, one array per class. Q: the subset of C that is used for drawing queries for scoring C, D: the data dictionary, maxqueries: maximum number of queries for scoring C in each step
	# start with all the training data C as Q. D is empty.
	C = []
	doubleC = []
	for sample_class in training_data:
		# TODO: add loop over various time series per class
		C.append(lin.hankel(sample_class[:l], sample_class[l-1:]).T) # create all l-length subsequences of the training data from the current class (assuming one time series per class)
	D = [[] for cl_num in xrange(N_train)] # D will contain actual sequences because we want to pass them to the classifier
	# Q0 = []
	# for sample_class in training_data:
	# 	# TODO: add loop over various time series per class
	# 	Q0.append(lin.hankel(sample_class[:l], sample_class[l-1:]).T) # create list all l-length subsequences of the training data
	# Q0 = n.array(Q0) # cast it into an array
	Q = []
	for i in xrange(len(C)):
		Q += [[i, j] for j in xrange(len(C[i]))] # for start, Q points to all of Q0
	Q = n.array(Q)

	Dcount = 0 # initialize counter for length of D
	while (Dcount < maxDlength) and (len(Q) > 0):
		# 1: score C by drawing random queries from Q and calculating likely false/true positives.
		S = [n.zeros(len(cl)) for cl in C] # initialize scores with zeros
		count = 0 # counter for scoring queries
		# chosenQs = n.random.choice(Q, maxqueries, replace = False)
		print("Length of Q: {0}".format(len(Q)))
		chosenQs = Q[list(n.random.permutation(n.arange(len(Q)))[:maxqueries])]
		for q in chosenQs:
			score(q, C, S)
		# 2: take the subsequence in C with the highest score, place it in D and remove it from C.
		if Dcount == 0: # in first iteration choose best subsequence from each class for placement in D
			for cl in range(len(C)):
				max_ind = S[cl].argmax()
				D[cl].append(C[cl][max_ind])
				C[cl] = n.delete(C[cl], max_ind, 0)
			Dcount = N_train
		else:
			max_index = [0,0]
			max_val = -n.inf
			for i, sc in enumerate(S):
				new_ind = sc.argmax()
				new_val = sc[new_ind]
				if new_val > max_val:
					max_val = new_val
					max_index = [i, new_ind]
			print(max_index)
			D[max_index[0]].append(C[max_index[0]][max_index[1]]) # add the highest scoring subsequence to the respective class in D and ...
			Dcount += 1
			C[max_index[0]] = n.delete(C[max_index[0]], max_index[1], 0) # ... remove it from C
		# 3: classify all subsequences in C with the new D. Make Q equal to the set of subsequences that could not be correctly classified.
		Q = []
		for class_label, seqs in enumerate(C):
			for k, seq in enumerate(seqs):
				# TODO: Check classification, something may be wrong
				class_prediction = classify(seq, D, n.inf) # for building D, use infinite threshold, so each query gets assigned
				if class_prediction != class_label:
					Q.append([class_label, k])
		Q = n.array(Q)
		print("Length of current D: {0}".format([len(d) for d in D]))
		# if length of D exceeds x: break, else: goto 1
	print("Fraction of wrongly classified queries after learning: {0}".format(float(len(Q))/float(N_subseq)))
	# learn the rejection threshold r, passing a set of valid and a set of invalid queries and D.
	# if we don't want to implement this, just plot a histogram of NN distances or guess threshold
	r = n.inf
	
	return D, r # return the data dictionary D and the rejection threshold r. D is an array of arrays, which in turn contain the base vectors for each class

"""
Return array of as many slice_size sized 1D arrays as can be sliced from sample file identified by ton_name.
"""
def get_sliced_up_sample(ton_name, slice_size):
	data = n.loadtxt('samples/' + ton_name + '.ton')
	slice_number = int(len(data)/slice_size)
	data = n.split(data, slice_size*n.arange(1,slice_number))
	return n.array(data[:-1]) # throw away last slice because it might be to short

"""
Return whole sample file identified by ton_name as 1D array.
"""
def get_sample(ton_name):
	data = n.loadtxt('samples/' + ton_name + '.ton')
	return data

learn_tones = ['E2', 'G2', 'D#3'] # position indices in this list determine class labels
sample_data = [get_sample(tn) for tn in learn_tones]
for i, s in enumerate(sample_data):
	sample_data[i] = s[:len(s)//4]

D, r = train(sample_data, 1., 512, 10)

cl = n.random.randint(len(learn_tones))
seq_ind = n.random.randint(len(sample_data[cl]) - 130)
testpred = classify(sample_data[cl][seq_ind:seq_ind+128], D, r)
print("Random query from class {0} classified as {1}".format(cl, testpred))