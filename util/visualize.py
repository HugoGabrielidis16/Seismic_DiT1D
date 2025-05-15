
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

y_0_rgb = (0/255, 0/255, 0/255)  # black
y_rgb = (230/255, 159/255, 0/255)  # orange
x_rgb = (86/255, 180/255, 233/255)  # blue

def amplitude_graph(y,generate,x,idx,show = False,path = "generated_samples/"):
    """
    Plot the amplitude of the input and output signals.
    Args:
        y (torch.Tensor): The original signal tensor of shape (3, n_samples).
        generate (torch.Tensor): The generated signal tensor of shape (3, n_samples).
        x (torch.Tensor): The input signal tensor of shape (3, n_samples).
        idx (int): The index for saving the plot.
        path (str): The path to save the plot.
        show (bool): Whether to show the plot or not.
    """
    direction = ["E-W", "N-S", "U-D"]
    num_channels = y.shape[0]
    try:
        generate = generate.detach().cpu()
        y = y.detach().cpu()
        x = x.detach().cpu()
    except:
        pass
    y_min = y.min()
    y_max = y.max()
    for i in range(num_channels):
        #plt.figure(figsize=(25, 10))
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(1,1, 1)

        ax.plot(y[i], label = '$y_{0}$', color = y_0_rgb, linewidth=3, zorder = 1)
        ax.plot(generate[i], label = '$y$', color = y_rgb, linewidth=1, zorder = 2)
        #ax.plot(x[i], label = 'x', color = "black", linewidth=1, zorder = 2)

    
        ax.set_title(f"{direction[i]}", fontsize = 25)
        ax.set_xlabel("t[s]", fontsize = 25)
        ax.set_ylabel(f"a(t)[m/s²]", fontsize = 25)
        ax.legend(fontsize = 10, loc='lower left')
        ax.set_xticks(np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60])*100)
        ax.set_xticklabels(np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60]), fontsize=25)

        # Create evenly spaced ticks
        num_ticks = 6  # You can adjust this number as needed
        ticks = np.linspace(y_min, y_max, num_ticks)

        # Round the ticks to make them cleaner
        ticks = np.round(ticks, decimals=2)  # Adjust decimals as needed

        # Set the ticks
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in ticks], fontsize=25)

        plt.tight_layout()
        ax.legend(fontsize = 25, loc='lower left')
        if show:
            plt.show()
        title = f"{path}/Amplitude_graph_{idx}_{direction[i]}.svg"
        plt.savefig(title)

       
def frequencyloglog_graph(y, generate,x, idx,path = "generated_samples/",show = False):
    """
    
    Plot the frequency spectrum of the input and output signals.

    Args:
        y (torch.Tensor): The original signal tensor of shape (3, n_samples).
        generate (torch.Tensor): The generated signal tensor of shape (3, n_samples).
        x (torch.Tensor): The input signal tensor of shape (3, n_samples).
        idx (int): The index for saving the plot.
        path (str): The path to save the plot.
        show (bool): Whether to show the plot or not.
    """
    
    
    try:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        generate = generate.cpu().numpy()
    except:
        pass
    viridis = cm.get_cmap('viridis')

    components = ["E-W", "N-S", "U-D"]
    for i in range(3):
        fig1, axs1 = plt.subplots(1, 1, figsize=(15, 10))
        n_samples = len(y[i])
        freq = np.fft.fftfreq(n_samples, d=0.005000)  # 100Hz sampling
        fft_y = np.abs(np.fft.fft(y[i]))
        fft_generate = np.abs(np.fft.fft(generate[i]))
        f_mse = np.abs(fft_y - fft_generate)**2
        fft_x = np.abs(np.fft.fft(x[i]))
        # Only plot positive frequencies
        pos_freq_mask = freq > 0
        axs1.loglog(freq[pos_freq_mask], fft_x[pos_freq_mask], label='$x$', color=x_rgb, linewidth =2,zorder =0)
        axs1.loglog(freq[pos_freq_mask], fft_y[pos_freq_mask], label='$y_{0}$', color=y_0_rgb, linewidth = 3,zorder = 1 )
        axs1.loglog(freq[pos_freq_mask], fft_generate[pos_freq_mask], label='$y$', color=y_rgb, linewidth = 2, zorder = 2)
        # Add vertical line at cutoff frequency
        #axs1.axvline(x=30, color='k', linestyle=':', label='Cutoff (30 Hz)')
        axs1.set_title(f"{components[i]}", fontsize = 25)
        axs1.set_xlabel('Frequency (Hz)')

        axs1.set_xticks([0.01,0.1,1,10,30])    
        axs1.set_yticks([10e-4,10e-3,10e-2,10e-1,10e-0,10])
        axs1.set_xticklabels(axs1.get_xticks(), fontsize=25)
        axs1.set_yticklabels(axs1.get_yticks(), fontsize=25)
       
        axs1.set_ylabel(f"a(f)[m/s²]", fontsize = 25)
        axs1.set_xlabel("f[Hz]", fontsize = 25)
        axs1.legend(fontsize = 25, loc='lower left')
        axs1.grid(True)
        axs1.set_xlim(0, 30)
        axs1.set_ylim(0.01,300)
        if show:
            plt.show()
        plt.tight_layout()
        #title = f'{path}/Frequency_{idx}_Spectrumloglog{components[i]}.png'
        title = f'{path}/Frequency_{idx}_Spectrumloglog{components[i]}.svg'
        plt.savefig(title)



def visualize_samples(x,y):
    x = np.array(x)
    y = np.array(y)
    print(f"MSE: {np.mean((x - y)**2)}")
    components = ['E', 'N', 'Z']
    colors = ['b', 'g', 'r']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(3):
        axs[0, i].plot(x[i], label='Input')
        axs[1, i].plot(y[i], label='Output')
        axs[0, i].set_title(f"Component X {components[i]} - Time Domain")
        axs[0, i].legend()
        axs[0, i].set_xlabel('Sample')
        axs[0, i].set_ylabel('Amplitude')
        axs[1, i].set_title(f"Component Y {components[i]} - Time Domain")
        axs[1, i].set_xlabel('Sample')
        axs[1, i].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        n_samples = len(x[i])
        freq = np.fft.fftfreq(n_samples, d=1/100)  # 100Hz sampling
        fft_x = np.abs(np.fft.fft(x[i]))
        fft_y = np.abs(np.fft.fft(y[i]))
        pos_freq_mask = freq > 0
        axs1[i].loglog(freq[pos_freq_mask], fft_x[pos_freq_mask], 
                      label='X', color='blue')
        axs1[i].loglog(freq[pos_freq_mask], fft_y[pos_freq_mask], 
                      label='Y', color='red')
        axs1[i].axvline(x=30, color='k', linestyle=':', label='Cutoff (30 Hz)')
        axs1[i].set_title(f"Component {components[i]} - Log Frequency Domain")
        axs1[i].set_xlabel('Frequency (Hz)')
        axs1[i].set_ylabel('Magnitude')
        axs1[i].legend()
        axs1[i].grid(True)
        axs1[i].set_xlim(0, 30)  # Showing up to 30Hz to see cutoff effect
    plt.tight_layout()
    plt.show()
    # Normal frequency domain plots
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        pos_freq_mask = freq > 0
        pos_freq = freq[pos_freq_mask]
        pos_fft_x = fft_x[pos_freq_mask]
        pos_fft_y = fft_y[pos_freq_mask]
        axs2[i].plot(pos_freq, pos_fft_x, 
                    label='X', color='blue')
        axs2[i].plot(pos_freq, pos_fft_y, 
                    label='Y', color='red')
        # Add vertical line at cutoff frequency
        axs2[i].axvline(x=30, color='k', linestyle=':', label='Cutoff (30 Hz)')
        axs2[i].set_title(f"Component {components[i]} - Linear Frequency Domain")
        axs2[i].set_xlabel('Frequency (Hz)')
        axs2[i].set_ylabel('Magnitude')
        axs2[i].legend()
        axs2[i].grid(True)
        # Set x-axis limit to focus on relevant frequency range
        axs2[i].set_xlim(0, 30)  # Showing up to 50 Hz to see cutoff effect
    plt.tight_layout()
    plt.show()