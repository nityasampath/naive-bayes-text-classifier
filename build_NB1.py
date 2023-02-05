#!/usr/bin/env python3

import sys
import math

training_file = sys.argv[1]
test_file = sys.argv[2]
class_prior_delta = float(sys.argv[3])
cond_prob_delta = float(sys.argv[4])
model_file = open(sys.argv[5], 'w')
sys_output = open(sys.argv[6], 'w')

#store training data
feat_counts = {} #{label : {word : # of docs}}
class_counts = {} #{label : # of docs}
class_priors = {} #{label : logprob}
cond_probs = {} #{label : {word : logprob}}
cond_probs_not = {} #{label : {word : logprob}}
unknowns = {} #{label : logprob}
unknowns_not = {} #{label : logprob}

#read in training data
with open(training_file, 'r') as train_data:

	for line in train_data:
		line = line.strip()
		tokens = line.split(' ')
		
		label = tokens[0]
		if label in class_counts:
			class_counts[label] += 1 
		else:
			class_counts[label] = 1

		for token in tokens[1:]:
			pair = token.split(':')
			word = pair[0]
			if label in feat_counts:
				if word in feat_counts[label]:
					feat_counts[label][word] += 1
				else:
					feat_counts[label][word] = 1
			else:
				feat_counts[label] = {word : 1}

class_total = sum(class_counts.values()) #total number of documents
class_num = len(class_counts) ##number of classes

model_file.write('%%%%% prior prob P(c) %%%%%\n')
for c in class_counts: #calculate class priors and store logprobs in dictionary
	class_prior = ((class_counts[c] + class_prior_delta)/(class_total + (class_prior_delta * class_num)))
	class_priors[c] = math.log10(class_prior)
	model_file.write(c + '\t' + str(class_prior) + '\t' + str(math.log10(class_prior)) + '\n')

model_file.write('%%%%% conditional prob P(f|c) %%%%%\n')
for c in feat_counts: #calculate conditional probabilities and unknown values and store logprobs in dictionaries
	model_file.write('%%%%% conditional prob P(f|c) c=' + c +' %%%%%\n')
	cond_probs[c] = {}
	cond_probs_not[c] = {}
	unknown = cond_prob_delta / (class_counts[c] + (cond_prob_delta * 2))
	unknown_not = 1 - unknown
	unknowns[c] = math.log10(unknown)
	unknowns_not[c] = math.log10(unknown_not)
	for word in feat_counts[c]:
		cond_prob = (feat_counts[c][word] + cond_prob_delta)/(class_counts[c] + (cond_prob_delta * 2)) 
		con_prob_not = 1 - cond_prob
		cond_probs[c][word] = math.log10(cond_prob)
		cond_probs_not[c][word] = math.log10(con_prob_not)
		model_file.write(word + '\t' + c + '\t' + str(cond_prob) + '\t' + str(math.log10(cond_prob)) + '\n')

model_file.close()

#fuction to classify datasets
def classify(data, data_label): 
	sys_output.write('%%%%% ' + data_label + ':' + '\n')

	#to store correct and incorrect counts to calculate accuracy
	confusion_matrix = [[0 for i in range(3)] for j in range(3)]
	i = 0
	j = 0

	with open(data, 'r') as data_file:

		linecount = 0

		for line in data_file:
			tokens = line.split(' ')

			class_label = tokens[0]

			logprobs = {} #to store logprobs

			for c in class_counts: #calculate logprobs for this document for each line
				logprob = 0
				for token in tokens[1:]:
					pair = token.split(':')
					word = pair[0]
					if word in cond_probs[c]:
						logprob += cond_probs[c][word] + cond_probs_not[c][word]
					else:
					 	logprob += unknowns[c] + unknowns_not[c] 
				logprob += class_priors[c]
				logprobs[c] = logprob

			#convert logprobs to probs
			maxlogprob = -1 * max(logprobs.values()) #get logprob with smallest absolute value

			probs = {} #to store probabilities

			for label, logprob in logprobs.items(): #divide logprobs by logprob with smallest absolute value
				quotient = logprob/maxlogprob
				prob = pow(10, quotient)
				probs[label] = prob

			sumprobs = sum(probs.values())

			#normalize probs by dividing each by the sum of the probs
			for label, prob in probs.items():
				probs[label] = prob/sumprobs

			system_out = max(probs, key=lambda key: probs[key]) #get highest probability label

			#count incorrect and correct responses
			if class_label == 'talk.politics.guns':
				i = 0
			elif class_label == 'talk.politics.misc':
				i = 1
			elif class_label == 'talk.politics.mideast':
				i = 2

			if system_out == 'talk.politics.guns':
				j = 0
			elif system_out == 'talk.politics.misc':
				j = 1
			elif system_out == 'talk.politics.mideast':
				j = 2

			confusion_matrix[i][j] = confusion_matrix[i][j] + 1

			#sort probabilities dictionary by value to print in order
			sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)

			output = ''
			for item in sorted_probs:
				label = item[0]
				prob = item[1]
				output += ' ' + label + ' ' + str(prob)

			sys_output.write('array:' + str(linecount) + ' ' + class_label + str(output) + '\n')
			linecount += 1

		sys_output.write('\n')

		#calculate accuracy from confusion matrix values
		total = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][2]
		correct = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2] 
		accuracy = correct/total

		print('Confusion matrix for the ' + data_label + ':\nrow is the truth, column is the system output')
		print()
		print('             talk.politics.guns talk.politics.misc talk.politics.mideast')
		print('talk.politics.guns ' + str(confusion_matrix[0][0]) + ' ' + str(confusion_matrix[0][1]) + ' ' + str(confusion_matrix[0][2]))
		print('talk.politics.misc ' + str(confusion_matrix[1][0]) + ' '  + str(confusion_matrix[1][1]) + ' '  + str(confusion_matrix[1][2]))
		print('talk.politics.mideast ' + str(confusion_matrix[2][0]) + ' '  + str(confusion_matrix[2][1]) + ' '  + str(confusion_matrix[2][2]))
		print()
		if data_label == 'training data':
			print(' Training accuracy= ' + str(accuracy))
		if data_label == 'test data':
			print(' Test accuracy= ' + str(accuracy))
		print()
		print()


classify(training_file, 'training data')
classify(test_file, 'test data')

sys_output.close()

