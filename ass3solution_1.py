import numpy as np
from scipy.signal import medfilt, find_peaks
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
import glob
import os
import math

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

def comp_acf(inputVector, bIsNormalized = True):    
    if bIsNormalized:        
        norm = np.dot(inputVector, inputVector)    
    else:        
        norm = 1    
    afCorr = np.correlate(inputVector, inputVector, "full") / norm    
    afCorr = afCorr[np.arange(inputVector.size-1, afCorr.size)]    
    return afCorr

def get_f0_from_acf (r, fs):    
    eta_min = 1    
    afDeltaCorr = np.diff(r)    
    eta_tmp = np.argmax(afDeltaCorr > 0)    
    eta_min = np.max([eta_min, eta_tmp])    
    f = np.argmax(r[np.arange(eta_min + 1, r.size)])    
    f = fs / (f + eta_min + 1)  
    return f

def track_pitch_acfmod(x,blockSize,hopSize,fs): 
    xb,timeInSec=block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros((xb.shape[0],1))    
    for i in range(xb.shape[0]):
       r = comp_acf(xb[i,:])
       f0[i] = get_f0_from_acf(r,fs)
    return f0,timeInSec
   

def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb,t)

########################### A. Maximum spectral peak based pitch tracker ###########################
def compute_spectrogram(xb, fs):
    numblocks,blocksize=np.shape(xb)
    X=np.zeros((int(blocksize/2+1),numblocks))
    for i in range(numblocks):
        b=xb[i]
        b = np.multiply(b, compute_hann(blocksize))
        magnitudes = np.abs(np.fft.rfft(b))
        X[:,i]=magnitudes
        fInHz = np.abs(np.fft.rfftfreq(blocksize, 1.0/fs)) 
        # positive frequencies
    return X,fInHz

def track_pitch_fftmax(x, blockSize, hopSize, fs):
    xb,timeInSec=block_audio(x,blockSize,hopSize,fs)
    X,fInHz=compute_spectrogram(xb, fs)
    f0=np.zeros((len(X[0]),1))
    for i in range(len(X[0])):
        magnitude=X[:,i]
        index=np.argmax(magnitude)
        f0[i]= fInHz[index]
    return f0, timeInSec

########################### B. HPS (Harmonic Product Spectrum) based pitch tracker ###########################

def get_f0_from_Hps(X, fs, order):
    blocklen,numblocks=X.shape
    f0=np.zeros((numblocks,1))
    fInHz =np.linspace(0,fs/2,blocklen)
    for i in range(numblocks):
        spec=X[:,i]
        spec = spec*spec
        spec_copy=spec
        for j in range(2,order+1):
            spec_new = spec_copy[::j]
            spec[:len(spec_new)]= spec[:len(spec_new)]*spec_new
            spec=spec[:len(spec_new)]
        index=np.argmax(spec)
        f0[i]=fInHz[index]                           
    return f0

'''def get_f0_from_Hps(X, fs, order):
    blocklen,numblocks=X.shape
    f0=np.zeros((numblocks,1))
    
    len_dec = int((X.shape[0] - 1) / order)
    dec_spectrum = X[np.arange(len_dec), :]
  
    
    for j in range(1, order):
        X_d = X[::(j + 1), :]
        dec_spectrum *= X_d[np.arange(0, len_dec), :]

    min_freq = 150
    no_zeros = int(round(min_freq / fs * 2 * (X.shape[0] - 1)))
    f0 = np.argmax(dec_spectrum[np.arange(no_zeros, dec_spectrum.shape[0])], axis=0)

    f0 = (f0 + no_zeros) / (X.shape[0] - 1) * fs / 2
    print(np.shape(f0))
    return f0'''

def track_pitch_hps(x, blockSize, hopSize, fs):
    xb,timeInSec=block_audio(x,blockSize,hopSize,fs)
    X,fInHz=compute_spectrogram(xb, fs)
    order=4
    f0=get_f0_from_Hps(X, fs, order)
    return f0,timeInSec


########################### C. Voicing Detection ###########################

def extract_rms(xb):
    numBlocks, blockSize=xb.shape 
    rmsDb=np.zeros((numBlocks,1))
    for i in range(numBlocks):
        b=xb[i]
        rms = np.sqrt(np.mean(np.square(b)))
        if rms < 1e-5:
            rms= 1e-5
        rms = 20*np.log10(rms)
        rmsDb[i]=rms
    return rmsDb    

def create_voicing_mask(rmsDb, thresholdDb):
    mask=np.zeros((len(rmsDb),1))
    for i in range(len(rmsDb)):
        if rmsDb[i] < thresholdDb:
            mask[i]=0
        else:
            mask[i]=1
    return mask

def apply_voicing_mask(f0, mask):
    f0Adj=f0*mask
    return f0Adj


########################### D.  Different evaluation metrics  ###########################             
def eval_voiced_fp(estimation, annotation):
    annotation_zero=0
    misclassify=0
    for i in range(len(estimation)):
        if annotation[i]==0:
            annotation_zero+=1
            if estimation[i]!=0:
                misclassify+=1
    pfp=misclassify/annotation_zero
    return pfp

def eval_voiced_fn(estimation, annotation):
    annotation_pos=0
    misclassify=0
    for i in range(len(estimation)):
        if annotation[i]!=0:
            annotation_pos+=1
            if estimation[i]==0:
                misclassify+=1
    pfn=misclassify/annotation_pos
    return pfn

def convert_freq2midi(fInHz, fA4InHz = 440):
    def convert_freq2midi_scalar(f, fA4InHz):
 
        if f <= 0:
            return 0
        else:
            return (69 + 12 * np.log2(f/fA4InHz))
        fInHz = np.asarray(fInHz)
        if fInHz.ndim == 0:
           return convert_freq2midi_scalar(fInHz,fA4InHz)

    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
       return convert_freq2midi_scalar(fInHz,fA4InHz)

    midi = np.zeros(fInHz.shape)
    for k,f in enumerate(fInHz):
        midi[k] =  convert_freq2midi_scalar(f,fA4InHz)
            
    return midi
        
def eval_rms(estimateInHz, groundtruthInHz):
    if np.abs(groundtruthInHz).sum() <= 0:
        return 0

    # truncate longer vector
    if groundtruthInHz.size > estimateInHz.size:
        estimateInHz = estimateInHz[np.arange(0,groundtruthInHz.size)]
    elif estimateInHz.size > groundtruthInHz.size:
        groundtruthInHz = groundtruthInHz[np.arange(0,estimateInHz.size)]

    diffInCent = 100*(convert_freq2midi(estimateInHz) - convert_freq2midi(groundtruthInHz))

    rms = np.sqrt(np.mean(diffInCent**2))
    return rms

def eval_pitchtrack_v2(estimation, annotation):
    if len(estimation)!= len(annotation):
        print('illegal estimation vector')
    else:
        pitchInMidi_esti=convert_freq2midi(estimation)
        pitchInMidi_anno=convert_freq2midi(annotation)
        errCentRms=eval_rms(pitchInMidi_esti, pitchInMidi_anno)
        pfn=eval_voiced_fn(estimation, annotation)
        pfn=np.array(pfn)
        pfp=eval_voiced_fp(estimation, annotation)
        pfp=np.array(pfp)
    return errCentRms, pfp, pfn
        

########################### E.  Evaluation  ########################### 
def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):
    if method == 'acf':
        f0,timeInSec = track_pitch_acfmod(x,blockSize,hopSize,fs)
    elif method == 'max':
        f0,timeInSec = track_pitch_fftmax(x, blockSize, hopSize, fs)
    elif method == 'hps':
        f0,timeInSec = track_pitch_hps(x, blockSize, hopSize, fs)
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    rmsDb = extract_rms(xb)
    mask = create_voicing_mask(rmsDb, voicingThres)
    f0Adj = apply_voicing_mask(f0, mask)
    return f0Adj, timeInSec

def executeassign3(complete_path_to_data_folder):
    fs = 44100
    f1 = 441
    f2 = 882
    x1 = np.arange(fs)
    x2 = np.arange(fs, 2 * fs)
    y1 = np.sin(2 * np.pi * f1 * (x1 / fs))
    y2 = np.sin(2 * np.pi * f2 * (x2 / fs))
    x = np.append(x1, x2)
    y = np.append(y1, y2)
    xb, t = block_audio(y,1024,512,fs)
    magnitude,freq=compute_spectrogram(xb,fs)
    f0_fftmax1, timeInSec=track_pitch_fftmax(y, 1024, 512, fs)
    f0_hps,timeInSec1=track_pitch_hps(y, 1024, 512, fs)
    plt.figure(1)
    plt.plot(timeInSec,f0_fftmax1)
    plt.title('f0 of sine wave detected with PitchTrackMax')
    plt.xlabel('time in s')
    plt.ylabel('frequency in Hz')
    diff=np.zeros((len(f0_fftmax1),1))
    diff1=np.zeros((len(f0_hps),1))
    for i in range(len(f0_fftmax1)):
        frequency=441
        if timeInSec[i]>=1:
            frequency=882
        diff[i]=abs(f0_fftmax1[i]-frequency)
        diff1[i]=abs(f0_hps[i]-frequency)
    plt.figure(2)
    plt.plot(timeInSec,diff)
    plt.title('absolute error for PitchTrackMax')
    plt.xlabel('time in s')
    plt.ylabel('frequency in Hz')
    f0_hps,timeInSec1=track_pitch_hps(y, 1024, 512, fs)
    plt.figure(3)
    plt.plot(timeInSec1,f0_hps)
    plt.title('f0 of sine wave detected with HPS')
    plt.xlabel('time in s')
    plt.ylabel('frequency in Hz')
    plt.figure(4)
    plt.plot(timeInSec,diff1)
    plt.title('absolute error for HPS')
    plt.xlabel('time in s')
    plt.ylabel('frequency in Hz')
    f0_fftmax2, timeInSec=track_pitch_fftmax(y, 2048, 512, fs)
    plt.figure(5)
    plt.plot(timeInSec,f0_fftmax2)
    plt.title('f0 of sine wave detected with PitchTrackMax(blocksize=2048)')
    plt.xlabel('time in s')
    plt.ylabel('frequency in Hz')
    diff2=np.zeros((len(f0_fftmax2),1))
    for i in range(len(f0_fftmax2)):
        frequency=441
        if timeInSec[i]>=1:
            frequency=882
        diff2[i]=abs(f0_fftmax2[i]-frequency)
    plt.figure(6)
    plt.plot(timeInSec,diff2)
    plt.title('absolute error for PitchTrackMax(blocksize=2048)')
    plt.xlabel('time in s')
    plt.ylabel('frequency in Hz')
    plt.show()
    files=os.listdir(complete_path_to_data_folder)
    performance_metrics_fftmax=np.zeros((3,1))
    performance_metrics_hps=np.zeros((3,1))
    count=0
    for file in files:
        if file.endswith('.txt'):
            count+=1
            print("test file :" + file)
            fp = open(complete_path_to_data_folder + '/' + file)
            annotation=[]
            for line in fp:
                tokens = line.split()
                annotation.append(float(tokens[2]))
            fs,data = read(complete_path_to_data_folder + '/' + file.replace('.f0.Corrected.txt','.wav'))
            estimation1,timeInSec=track_pitch_fftmax(data, 1024, 512, fs)
            errCentRms1, pfp1, pfn1=eval_pitchtrack_v2(estimation1,annotation)
            performance_metrics1=np.vstack((errCentRms1, pfp1, pfn1))
            performance_metrics_fftmax+=performance_metrics1
            estimation2,timeInSec=track_pitch_hps(data, 1024, 512, fs)
            errCentRms2, pfp2, pfn2=eval_pitchtrack_v2(estimation2,annotation)
            performance_metrics2=np.vstack((errCentRms2, pfp2, pfn2))
            performance_metrics_hps+=performance_metrics2
            estimation_40_acf,timeInSec=track_pitch(data, 1024, 512, fs, 'acf', -40)
            errCentRms_40_acf, pfp_40_acf, pfn_40_acf=eval_pitchtrack_v2(estimation_40_acf,annotation)
            print("Evaluate function: track_picth " )
            print("errCentRms_40_acf : % 10.3E, pfp_40_acf : % 10.3E, pfn_40_acf: % 10.3E" %(errCentRms_40_acf, pfp_40_acf, pfn_40_acf)) 
            estimation_40_fftmax,timeInSec=track_pitch(data, 1024, 512, fs, 'max', -40)
            errCentRms_40_fftmax, pfp_40_fftmax, pfn_40_fftmax=eval_pitchtrack_v2(estimation_40_fftmax,annotation)
            print("errCentRms_40_fftmax : % 10.3E, pfp_40_fftmax : % 10.3E, pfn_40_fftmax % 10.3E" %(errCentRms_40_fftmax, pfp_40_fftmax, pfn_40_fftmax))
            estimation_40_hps,timeInSec=track_pitch(data, 1024, 512, fs, 'hps', -40)
            errCentRms_40_hps, pfp_40_hps, pfn_40_hps=eval_pitchtrack_v2(estimation_40_hps,annotation)
            print("errCentRms_40_hps : % 10.3E, pfp_40_hps : % 10.3E, pfn_40_hps % 10.3E" %(errCentRms_40_hps, pfp_40_hps, pfn_40_hps))
            estimation_20_acf,timeInSec=track_pitch(data, 1024, 512, fs, 'acf', -20)
            errCentRms_20_acf, pfp_20_acf, pfn_20_acf=eval_pitchtrack_v2(estimation_20_acf,annotation)
            print("errCentRms_20_acf : % 10.3E, pfp_20_acf : % 10.3E, pfn_20_acf: % 10.3E" %(errCentRms_20_acf, pfp_20_acf, pfn_20_acf)) 
            estimation_20_fftmax,timeInSec=track_pitch(data, 1024, 512, fs, 'max', -20)
            errCentRms_20_fftmax, pfp_20_fftmax, pfn_20_fftmax=eval_pitchtrack_v2(estimation_20_fftmax,annotation)
            print("errCentRms_20_fftmax : % 10.3E, pfp_20_fftmax : % 10.3E, pfn_20_fftmax % 10.3E" %(errCentRms_20_fftmax, pfp_20_fftmax, pfn_20_fftmax))
            estimation_20_hps,timeInSec=track_pitch(data, 1024, 512, fs, 'hps', -20)
            errCentRms_20_hps, pfp_20_hps, pfn_20_hps=eval_pitchtrack_v2(estimation_20_hps,annotation)
            print("errCentRms_20_hps : % 10.3E, pfp_20_hps : % 10.3E, pfn_20_hps % 10.3E" %(errCentRms_20_hps, pfp_20_hps, pfn_20_hps))
    average_performance_metrics_hps=performance_metrics_hps/count
    print("Evaluate function:  track_pitch_hps() " )
    print("errCentRms_average_for_ track_pitch_hps : % 10.3E, pfp_average_for_ track_pitch_hps : % 10.3E, pfn_average_for_ track_pitch_hps % 10.3E" %(average_performance_metrics_hps[0],average_performance_metrics_hps[1],average_performance_metrics_hps[2]))
    average_performance_metrics_fftmax=performance_metrics_fftmax/count
    print("Evaluate function:  track_pitch_fftmax() " )
    print("errCentRms_average_for_ track_pitch_fftmax : % 10.3E, pfp_average_for_ track_pitch_fftmax : % 10.3E, pfn_average_for_ track_pitch_fftmax % 10.3E" %(average_performance_metrics_fftmax[0],average_performance_metrics_fftmax[1],average_performance_metrics_fftmax[2]))
            
def track_pitch_mod2(x, blockSize, hopSize, fs):
    xb,timeInSec=block_audio(x,blockSize,hopSize,fs)
    f0 = np.zeros((xb.shape[0],1))  
    for i in range(xb.shape[0]):
       r = comp_acf(xb[i,:])
       r1 = medfilt(r, kernel_size=(5,1));       
       f0[i]=get_f0_from_acf(r1,fs)
    return f0,timeInSec

       
       
       
    
if __name__ == "__main__":
    
    executeassign3('/Users/yuyifei/Desktop/6201/homework3/trainData')
    
    
        







