import csv
import glob
import os
import librosa
import scipy
import numpy as np

def extract_features(file_name):

        data, rate = librosa.load(file_name, mono=True, sr=44100)
	frameSize = 2048
	binWidth = rate/frameSize
	
	freqMinSec = []
        freqMaxSec = []
        freqStdSec = []
        mfccLogPowerSec = []
        mfccMeanSecond = []
        mfccStdSec = []
        mfccDeltaMeanSec = []
        mfccDeltaStdSec = []
        zeroXrateMinSec = []

	#one second samples are stored
	#(mean value of all frames processed per second)
	inputSamples = int(np.floor(len(data)/rate))
	print "duration: ", inputSamples

	for i in range(inputSamples):

		mfccsMean = []
        	mfccsStd = []
        	mfccsLogPower = []
        	mfccs_delta_mean = []
        	mfccs_delta_std = []
        	freqs = []
	        zeroXrates = []

		#We are overlapping our samples by 50%
		#Because the frame is shifted to 1/2 its position
		for j in range(binWidth*2):
			frame = int(round(i*j*frameSize/2))
			sample = data[frame:frame+frameSize]
			sample = librosa.util.normalize(sample)

			#Short-Fourier Transform on the sample
			stft = np.abs(librosa.stft(sample))
			
			#Get the index of the highest signal
			maxIndex = np.where(stft==np.max(stft))
			freq = maxIndex[0][0]*binWidth  #Get frequency of the highest signal
			
			if(freq > 300 and freq < 3200):	#if the pitch is within the human vocal range then store the features
				freqs.append(freq)
				mfcc = librosa.feature.mfcc(sample, n_mfcc=26)
				mfccsLogPower.append(np.mean(mfcc[0]))
				mfccsMean.append(np.mean(mfcc[1:13]))
				mfccsStd.append(np.std(mfcc[1:13]))
				mfcc_delta = librosa.feature.delta(mfcc[14:26])
				mfccs_delta_mean.append(np.mean(mfcc_delta))
				mfccs_delta_std.append(np.std(mfcc_delta))
				zeroXrate = librosa.feature.zero_crossing_rate(sample)
				zeroXrates.append(zeroXrate)
		#The end of each second
		#This is where we actually want to log some data into the csv
		if(freq > 300 and freq < 3200):		#Only log if pitch is within human vocal range
			freqMinSec.append(np.min(freqs))
			freqMaxSec.append(np.max(freqs))
			freqStdSec.append(np.std(freqs))
			mfccLogPowerSec.append(np.mean(mfccsLogPower))
			mfccMeanSecond.append(np.mean(mfccsMean))
			mfccStdSec.append(np.mean(mfccsStd))
			mfccDeltaMeanSec.append(np.mean(mfcc_delta))
			mfccDeltaStdSec.append(np.mean(mfccs_delta_mean))
			zeroXrateMinSec.append(np.min(zeroXrates))

	return freqMinSec, freqMaxSec, freqStdSec, mfccLogPowerSec, mfccMeanSecond, mfccStdSec, mfccDeltaMeanSec, mfccDeltaStdSec, zeroXrateMinSec

#extract_features("wav/bad/Simple Tensorflow Speech Recognizer.wav")

