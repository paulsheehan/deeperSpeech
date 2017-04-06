import extract_features as ext
import csv
import glob
import os
import numpy as np


outCSVTrain = open('data/training_features2.csv', 'w')
outCSVTest = open('data/test_features2.csv', 'w')

outputTrain = csv.writer(outCSVTrain, delimiter=',')
outputTest = csv.writer(outCSVTest, delimiter=',')

def load_sound_files(file):
	data = ext.extract_features(file)

	return data

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
	headers = ["id", "freq_min", "freq_max", "freq_std", \
		"mfcc_power", "mfcc_mean", "mfcc_std", "mfcc_delta_mean", \
		"mfcc_delta_std", "min_zero_crossing_rate", "result"]

	#count each file to store ID	
	trainCount = 0
	testCount = 0

	#write headers to the training dataset
	outputTrain.writerow(headers)
	outCSVTrain.flush()

	#write headers to the test dataset
	outputTest.writerow(headers)
	outCSVTest.flush()
	
	for i in sub_dirs:

		if i == "good/":
			label = 1
		else:
			label = 0
		for fn in glob.glob(os.path.join(parent_dir, i, file_ext)):
			rSplit = np.random.randint(1, 100)
			print fn
			try:
				#extract 11 features from the audio file
				freqMin, freqMax, freqStd, mfccsLogPower, mfccsMean, \
				mfccsStd, mfccs_delta_mean, \
				mfccs_delta_std, zxr = ext.extract_features(fn)

				#80% chance the extracted features will be written to train CSV
				if(rSplit < 80):
					#write features to train CSV
					for j in range(len(freqMin)):
						outputTrain.writerow([trainCount, freqMin[j], freqMax[j], freqStd[j], \
							mfccsLogPower[j], mfccsMean[j], mfccsStd[j], mfccs_delta_mean[j], \
							mfccs_delta_std[j], zxr[j], label])
						outCSVTrain.flush()
					trainCount+=1
				else:
					for j in range(len(freqMin)):
						#write features to test CSV
                                                outputTest.writerow([testCount, freqMin[j], freqMax[j], \
						freqStd[j], mfccsLogPower[j], mfccsMean[j], mfccsStd[j], \
						mfccs_delta_mean[j], mfccs_delta_std[j], zxr[j], label])

						outCSVTest.flush()
                                        testCount+=1
            		except Exception as e:
				print "Error encountered while parsing file: ", fn
				continue
	return 0

parent_dir = 'wav'
sub_dirs = ["good/", "bad/"]
parse_audio_files(parent_dir, sub_dirs)

