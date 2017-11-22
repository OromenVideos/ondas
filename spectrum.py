"""
I, the owner of the YouTube channel 

    Oromen 
    https://www.youtube.com/channel/UClLhHAJP6BBAGvGVTFE7l9g


Used this program for generating the animations and processing sounds in the video

    "Ondas, Sonido y Espectros"
    https://youtu.be/s7DeLWXeWgY

and I therefore submit this piece of code to the Public Domain.

"""

__author__ = 'OromenVideos'

# Uncomment these two lines if run in a terminal without graphics support: 
# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, filtfilt


def load_wave(flute=True, save_wave=False, filename='wave.wav'):
    """If 'flute' is True, it loads the wavefile 'flute.wav' into the variable 'y'.
    If not, it generates a square or delta pulses wave.
    If 'save_wave' is True, it saves the generated/loaded wave in the file 'filename'"""
    global Fs, y, Ts, t, n, k, T, frq

    if flute:
        Fs, y = wavfile.read('flute.wav')
        #Fs, y = wavfile.read('a.wav')

        #normalize:
        if y.dtype in [np.dtype('int32'), np.dtype('int16')]:
            y2 = y/np.iinfo(y.dtype).max
            y = y2
        elif y.dtype is np.dtype('uint8'):
            y2 = (y-np.iinfo(y.dtype).max//2)/2
            y = y2
        elif y.dtype is np.dtype('float32'):
            raise Exception("Unsupported type {}".format(y.dtype))

        y = np.array([(i[0]+i[1])/2 for i in y])

        Ts = 1./Fs
        t = np.arange(0, len(y)*Ts, Ts)

    else:

        Fs = 48000  # sampling rate
        Ts = 1.0/Fs # sampling interval
        t = np.arange(0,2,Ts) # time vector

        #square wave
        y = np.array([-.9 if (i//50)%2==0 else .9 for i in range(len(t))])

        #delta pulses
        # y = np.array([.9 if i%300==0 else -.9/299 for i in range(len(t))])

        if save_wave:
            wavfile.write(filename, Fs, y)

    n = len(t) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range


# ===== Signal Fourier transform =====

def fourier(show_plot=True, save_image=False, filename='fourier'):
    """It takes the wave saved in the variable 'y' and saves its Fourier transform in the
    variable 'Y'. If 'show_plot' is True, it shows a part of the wave.
    If 'save_image' is True, it saves the image of the wave and its transform in the file 'filename'"""

    global Y

    Y = np.fft.fft(y)/n # fft computing and normalization
    if 0 in Y:
        Y = np.array([1e-17 if i==0 else i for i in Y[range(n//2)]])
    else:
        Y = Y[range(n//2)]

    if show_plot or save_image:
        plt.figure(figsize=(12,6.75))
        plt.subplot(211)
        plt.plot(t[:500],y[:500])
        plt.yticks([])
        plt.xticks([])
        plt.subplot(212)
        plt.semilogy(frq[:len(frq)//2], abs(Y[:len(frq)//2]),'r') # plotting the spectrum
        plt.yticks([])
        plt.xticks([])

        if show_plot:
            plt.show()
        if save_image:
            plt.savefig(filename)


# ===== Signal reconstruction =====

def r(Y, threshold=1e-5, n_freqs=200):
    """It takes the 'n_freqs' sine waves of the spectrum 'Y' with amplitude higher than 'threshold' and adds them."""
    freqs = []
    suma = np.zeros(n)

    for j in range(len(Y)):
        if abs(Y[j]) > threshold:
            bol = False
            for fq in freqs:
                if .98 < fq/frq[j] < 1.02:
                    bol = True
                    break
            if bol:
                continue
            freqs.append(frq[j])
            suma += -abs(Y[j])*np.sin(2*np.pi*frq[j]*t)
            if len(freqs) == n_freqs:
                return suma
    return suma

def reconstruct(create_wave_file=False, filename='wave_function', show_plot=True, flauta=True, save_image=False):
    """Using the function r(), it reconstructs the signal from its fourier transform 'Y'. 
    If 'create_wave_file' is True, it saves the reconstructed signal in the wav file 'filename'.
    If 'show_plot' is True, it shows an image of the reconstruction.
    If 'flauta' is True, it divides the wave in pieces, generates a signal from the pieces and adds them in a single one.
    If 'save_image' is True, it saves the image with the name 'filename'"""

    if flauta:
        # window shift
        Dn = int(Fs/24)
        # window width
        ns = int(Fs*.075)

        suma = np.zeros(n)
        for i in range(n//Dn):
            if i*Dn + ns > n:
                ns = n-i*Dn
            Y_w = np.fft.fft(np.concatenate([np.zeros(i*Dn), y[i*Dn:i*Dn+ns]*np.hamming(ns), np.zeros(n-i*Dn-ns)])/n)[range(n//2)] 
            suma2 = np.concatenate([np.zeros(i*Dn), (r(Y_w)[i*Dn:i*Dn+ns])*np.hamming(ns), np.zeros(n-i*Dn-ns)])
            suma += suma2
        suma /= max(suma)
    else:
        suma = r(Y, 1e-4, 10)
        #suma = r(Y, abs(Y[0]*1.5))/np.abs(Y).max()

    if create_wave_file:
        wavfile.write(filename+'.wav', Fs, suma)

    if show_plot or save_image:
        plt.figure(figsize=(12,6.75))
        plt.subplot(211)
        plt.plot(t[:500],y[:500])
        plt.yticks([])
        plt.xticks([])
        plt.subplot(212)
        plt.plot(t[:500],suma[:500],'r') # plotting the spectrum
        plt.yticks([])
        plt.xticks([])

        if show_plot:
            plt.show()
        if save_image:
            plt.savefig(filename)

    else:
        return suma


def save_images(log_scale=False):
    """Saves a series of sequences in 24fps that divide the wave function 'y' in pieces and computes
    the fourier transform of each.

    You should have a folder named 'images' in the same directory than the program."""

    # window shift
    Dn = int(Fs/24)
    # window width
    ns = int(Fs*.075)

    max_amp_y = max(y)

    max_amp_Y = 1e-4

    bol_break = False
    print("Number of images: {}".format(n//Dn))
    for i in range(n//Dn):
        if i*Dn + ns > n:
            ns = n-i*Dn
        t_w, y_w = t[i*Dn:i*Dn+ns], y[i*Dn:i*Dn+ns]*np.hamming(ns)
        Y_w = np.fft.fft(np.concatenate([np.zeros(i*Dn), y_w, np.zeros(n-i*Dn-ns)]))/n # fft computing and normalization
        Y_w = Y_w[range(n//2)]
        plt.figure(figsize=(12,6.75))
        plt.subplot(211)
        plt.ylabel('Señal de audio')
        plt.xlabel('Tiempo')
        plt.plot(t_w, y_w)
        x1,x2,_,_ = plt.axis()
        plt.axis((x1,x2,-max_amp_y, max_amp_y))
        plt.yticks([])
        plt.xticks([])
        plt.subplot(212)
        plt.ylabel('Fourier')
        plt.xlabel('Frecuencia')
        if log_scale:
            plt.semilogy(frq, abs(Y_w))
        else:
            plt.plot(frq, abs(Y_w))
        x1,x2,_,_ = plt.axis()
        plt.axis((x1, x2, 10**-8 if log_scale else 0, max_amp_Y))
        plt.yticks([])
        plt.xticks([])
        plt.savefig('images/fourier_log_{:0{}}'.format(i,int(np.log(1+len(y)/Dn)/np.log(10))+1))
        plt.close()
        print("{}/{}: {:.2%}".format(i+1, n//Dn, (i+1)/(n//Dn)))


def spec(sequence=True, save_image=True, filename='spectrogram'):
    """Shows the spectrogram of the signal 'y' and if 'save_image' is True, it saves it in the file 'filename'.
    If 'sequence' is True, then the program generates a sequence of 
    images in the folder 'images' with the name 'spectrogram_xxx' in 24fps. """
    fspec, tspec, Sxx = spectrogram(y, fs=Fs, noverlap=int(Fs*0.015), nperseg=int(Fs*.03), scaling='spectrum')

    spec_min = Sxx.min()
    spec_max = Sxx.max()

    if save_image:
        plt.pcolormesh(tspec, fspec, Sxx, norm=colors.LogNorm(vmin=spec_min, vmax=spec_max), cmap=plt.cm.get_cmap('inferno'))
        plt.axis('off')
        plt.savefig(filename)
        plt.close()

    if sequence:
        nn = len(Sxx[0])
        Dn = Fs*nn/(n*24)

        for i in range(1,int(n*24/Fs)):
            Sxx2 = np.array([ [ l[j] if j < i*Dn else spec_min for j in range(len(l))] for l in Sxx ])
            plt.pcolormesh(tspec, fspec, Sxx2, norm=colors.LogNorm(vmin=spec_min, vmax=spec_max), cmap=plt.cm.get_cmap('inferno'))
            plt.yticks([])
            plt.xticks([])
            plt.savefig('images/'+filename+'_{:0{}}'.format(i,int(np.log(1+nn/Dn)/np.log(10))+1))
            plt.close()
            print("{}/{}: {:.2%}".format(i, nn//Dn-1, (i)/(nn//Dn -1)))


def filter(create_wave_file=True, filename='filtered.wav', show=False):
    """This function generates the vowels /aeiou/ from the wave 'y'. If 'create_wave_file' is True, it saves it in the file
    'filename'. If 'show' is True, it shows the filters used for the vowel and the spectrum of the resulting signal."""

    v_a = [850, 1610, 2500]
    v_e = [500, 1800, 2500]
    v_i = [300, 2150, 3000]
    v_o = [500, 875, 2400]
    v_u = [350, 800, 2400]

    suma = []
    for l in [v_a,v_e,v_i,v_o,v_u]:

        z = .01
        w0 = 2*3.1415*l[0]
        F1 = np.array([-(2*3.1415*w)**2 for w in frq])/np.array([1j*w*(-(2*3.1415*w)**2 + 2j*z*w0*(2*3.1415*w) + w0**2) for w in frq])
        w0 = 2*3.1415*l[1]
        F2 = .9*np.array([-(2*3.1415*w)**2 for w in frq])/np.array([1j*w*(-(2*3.1415*w)**2 + 2j*z*w0*(2*3.1415*w) + w0**2) for w in frq])
        w0 = 2*3.1415*l[2]
        z = .03
        F3 = .7*np.array([-(2*3.1415*w)**2 for w in frq])/np.array([1j*w*(-(2*3.1415*w)**2 + 2j*z*w0*(2*3.1415*w) + w0**2) for w in frq])

        w0 = 100000
        LP = 100/np.array([(2j*3.1415*w + w0) for w in frq])

        filt = 1e2*(F1+F2+F3+LP)

        Y_filtered = 10*Y*filt

        suma = np.concatenate([suma, r(Y_filtered, 1e-2, 40)[:n//4]])

    if create_wave_file:
        wavfile.write(filename, Fs, suma)

    if show:
        fig, ax = plt.subplots(2, 1)

        ax[0].loglog(frq[:9000],abs(filt[:9000]))
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('x(t)')
        ax[1].plot(frq, abs(Y_filtered))
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('x\'(t)')

        plt.show()

    else:
        return suma


def sin(n=48, fr=False, amp=False):
    """It generates 'n' images of a sine wave. If 'fr' is True, it variates its frequency. If 'amp' is True, it variates its amplitude."""

    t = np.arange(0,10,.01)
    for i in range(n):
        if fr:
            if i < n//2:
                y = np.sin(2*np.pi*((i+1)*t/5+i*np.ones(len(t))*2/n))
            else:
                y = np.sin(2*np.pi*((n-i)*t/5+i*np.ones(len(t))*2/n))
        elif amp:
            if i < n//2:
                y = i*np.sin(2*np.pi*(t+i*np.ones(len(t))*2/n))*2/n
            else:
                y = (n-i)*np.sin(2*np.pi*(t+i*np.ones(len(t))*2/n))*2/n
        else:
            y = np.sin(2*np.pi*(t+i*np.ones(len(t))*2/n))

        plt.figure(figsize=(17,5))
        plt.plot(t, y, linewidth=3)
        x1,x2,_,_ = plt.axis()
        plt.axis((x1,x2,-1.1, 1.1))
        plt.axis('off')
        plt.savefig('images/sin_wave_b_' + ('chirp_' if fr else '') + ('amp_'if amp else '') + '{:02}'.format(i))
        plt.close()
        print("{}/{}: {:.2%}".format(i+1, n, (i+1)/n))

def square(n=48):
    """It generates 'n' images of a square wave. """

    t = np.arange(0,10,.01)

    for i in range(n):
        y = [ -1 if int(j+i*4/n)%2==0 else 1 for j in t]

        plt.figure(figsize=(17,5))
        plt.plot(t, y, linewidth=3)
        x1,x2,_,_ = plt.axis()
        plt.axis((x1,x2,-1.1, 1.1))
        plt.axis('off')
        plt.savefig('images/square_wave_{:02}'.format(i))
        plt.close()
        print("{}/{}: {:.2%}".format(i+1, n, (i+1)/n))

def pulses(n=48):
    """It generates 'n' images of a pulse train. """

    t = np.arange(1000)

    for i in range(n):
        y = [ 1 if int(j+4*i)%100==0 else -1 for j in t]

        plt.figure(figsize=(17,5))
        plt.plot(t, y, linewidth=3)
        x1,x2,_,_ = plt.axis()
        plt.axis((x1,x2,-1.1, 1.1))
        plt.axis('off')
        plt.savefig('images/pulses_{:02}'.format(i))
        plt.close()
        print("{}/{}: {:.2%}".format(i+1, n, (i+1)/n))


def envelope():
    """It calculates the envelope of the spectrum 'Y' and plots it in two images: one with it and other without it."""

    env = np.array([])

    w = 200

    for i in range(len(frq)//w +1):
        env = np.concatenate([env, np.max(np.abs(Y[i*w:(i+1)*w if (i+1)*w<=len(Y) else -1]))*np.ones(w if (i+1)*w<=len(Y) else len(Y)-i*w)])
    
    b, a = butter(1, .003)

    env = filtfilt(b, a, env)
    env = filtfilt(b, a, env)

    nn = len(frq)//2

    plt.figure(figsize=(12,6.75))
    plt.semilogy(frq[:nn], np.abs(Y[:nn]), 'r', lw=.5)
    plt.yticks([])
    plt.xticks([])
    plt.savefig('envelope_without')

    plt.figure(figsize=(12,6.75))
    plt.semilogy(frq[:nn], np.abs(Y[:nn]), 'r', lw=.5)
    plt.semilogy(frq[:nn], np.abs(env[:nn]), 'g', lw=2)
    plt.yticks([])
    plt.xticks([])
    plt.savefig('envelope_with')


##############################################################################################################
#### BEFORE UNCOMMENTING AND EXECUTING, CHOOSE THE DESIRED FUNCTIONS AND ITS PARAMETERS TO AVOID PROBLEMS ####
##############################################################################################################


# load_wave(flute=False, save_wave=True, filename='pulses.wav')

# fourier(show_plot=False, save_image=True, filename='a_fourier')

# reconstruct(create_wave_file=True, filename='recunstructed_flute2', show_plot=False, flauta=False, save_image=True)

# save_images(log_scale=True)

# spec(sequence=True, save_image=False, filename='spectrogram')

# filter(show=False, filename='generated_vowels.wav')

# sin()

# square()

# pulses(50)

# envelope()