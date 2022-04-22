import torchaudio
import matplotlib
import matplotlib.pyplot as plt

[width, height] = matplotlib.rcParams['figure.figsize']
if width < 10:
  matplotlib.rcParams['figure.figsize'] = [width * 2.5, height]


if __name__ == "__main__":
    # filename = "/apdcephfs/private_helinwang/tsss/tsss_mixed/train/train_1.wav"
    filename = "/apdcephfs/private_helinwang/tsss/tsss_mixed/test_offset/test_offset_10_re.wav"
    waveform, sample_rate = torchaudio.load(filename)
    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))
    plt.figure()
    plt.plot(waveform.t().numpy())
    # plt.title('test_offset_100_mix')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig('test_offset_10_re.png')
