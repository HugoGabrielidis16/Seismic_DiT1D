import os
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedFormatter, FixedLocator
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from obspy.signal.tf_misfit import eg, pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

SAMPLING_RATE = 100

color_y_0 = 0
color_y = 100 # bleu plus clair
color_x = 200
# Autre combinaison de couleurs


y_0_rgb = (0/255, 0/255, 0/255)  # black
y_rgb = (230/255, 159/255, 0/255)  # orange
x_rgb = (86/255, 180/255, 233/255)  # blue



def amplitude_graph(y,
                    generate,
                    x,
                    path,
                    idx,
                    x_magnitude = None,
                    y_magnitude = None,
                    normalized = False,
                    format = "png",
                    show_x = False,
                    station_name = None):
    direction = ["E-W", "N-S", "U-D"]
    num_channels = y.shape[0]
    #print(viridis(0))
    if x_magnitude is not None and y_magnitude is not None:
        y_magnitude = y_magnitude.view(3,1)
        x_magnitude = x_magnitude.view(3,1)
        y = (y*y_magnitude).detach().cpu() if y_magnitude is not None else y.detach().cpu()
        generate = (generate*y_magnitude).detach().cpu() if y_magnitude is not None else generate.detach().cpu()
        x = (x*x_magnitude).detach().cpu() if x_magnitude is not None else x.detach().cpu()
    else:
        y = y.detach().cpu()
        generate = generate.detach().cpu()
        x = x.detach().cpu()
    y_min = min(y.min(), generate.min())
    y_max = max(y.max(), generate.max())
    for i in range(num_channels):
        #plt.figure(figsize=(25, 10))
        #plt.figure(figsize=(15, 10))
        plt.figure(figsize=(8, 4))
        ax = plt.subplot(1,1, 1)

        ax.plot(y[i], label = '$y_{0}$', color = y_0_rgb , linewidth=3, zorder = 1) # black
        ax.plot(generate[i], label = '$y$', color = y_rgb, linewidth=2, zorder = 2) # orange
        if show_x:
            ax.plot(x[i], label = '$x$', color = x_rgb, linewidth=2, zorder = 2)

    
        ax.set_title(f"{direction[i]}", fontsize = 25)
        ax.set_xlabel("t[s]", fontsize = 25)
        ax.set_ylabel(f"a(t)[m/s²]", fontsize = 25)
        ax.legend(fontsize = 10, loc='lower left')
        ax.set_xticks(np.array([0,12,24,36,48,60])*100)
        ax.set_xticklabels(np.array([0,12,24,36,48,60]), fontsize=25)

        if normalized:
            ax.set_yticks(np.array([-1.0,-0.5,0.0,0.5,1.0]))
            ax.set_yticklabels(np.array([-1.0,-0.5,0.0,0.5,1.0]), fontsize=25)
            ax.set_ylim(-1.1,1.1)
            ax.set_ylabel(f"a(t)[1]", fontsize = 25)
        else:
            num_ticks = 3  # You can adjust this number as needed
            ticks = np.linspace(0,y_max, num_ticks)
            ticks = np.stack((-ticks, ticks), axis=1).flatten()
            ticks = np.round(ticks, decimals=2)  # Adjust decimals as needed
            ax.set_yticks(ticks)
            ax.set_yticklabels([f'{tick:.2f}' for tick in ticks], fontsize=25)

        plt.tight_layout()
        ax.legend(fontsize = 15, loc='lower right',frameon=False)
        if station_name is not None:
            title = f"{path}/{station_name}_Amplitude_graph_{idx}_{direction[i]}.{format}"
            title_png = f"{path}/{station_name}_Amplitude_graph_{idx}_{direction[i]}.png"
        else:
            if normalized:
                title = f"{path}/Normalized_Amplitude_graph_{idx}_{direction[i]}.{format}"
            else:
                title = f"{path}/Amplitude_graph_{idx}_{direction[i]}.{format}"
        print(f"Amplitude graph saved to {title}")
        plt.savefig(title)


def frequency_loglog(y, 
                       generate, 
                       x, 
                       path, 
                       idx, 
                       format="png", 
                       x_magnitude=None, 
                       y_magnitude=None,
                       station_name=None, 
                       normalized=False,
                       xlim = 30):
    if x_magnitude is not None and y_magnitude is not None:
        x_magnitude = x_magnitude.view(3,1)
        y_magnitude = y_magnitude.view(3,1)
        x = (x*x_magnitude).cpu().numpy()  if x_magnitude is not None else x.cpu().numpy()
        y = (y*y_magnitude).cpu().numpy()  if y_magnitude is not None else y.cpu().numpy()
        generate = (generate*y_magnitude).cpu().numpy() if y_magnitude is not None else generate.cpu().numpy()
    else:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        generate = generate.cpu().numpy()
    
    viridis = cm.get_cmap('viridis')
    fft_y_max = max(np.max(np.abs(np.fft.fft(y))), np.max(np.abs(np.fft.fft(generate))))
    components = ["E-W", "N-S", "U-D"]
    
    for i in range(3):
        fig1, axs1 = plt.subplots(1, 1, figsize=(8, 4))
        n_samples = len(y[i])
        freq = np.fft.fftfreq(n_samples, d=0.005000)  # 100Hz sampling
        fft_y = np.abs(np.fft.fft(y[i]))
        fft_generate = np.abs(np.fft.fft(generate[i]))
        fft_x = np.abs(np.fft.fft(x[i]))
        
        # Only plot positive frequencies
        pos_freq_mask = freq > 0
        axs1.loglog(freq[pos_freq_mask], fft_y[pos_freq_mask], label='$y_{0}$', color=y_0_rgb, linewidth=3, zorder=1)
        axs1.loglog(freq[pos_freq_mask], fft_x[pos_freq_mask], label='$x$', color=x_rgb, linewidth=3, zorder=2)
        axs1.loglog(freq[pos_freq_mask], fft_generate[pos_freq_mask], label='$y$', color=y_rgb, linewidth=2, zorder=2)
        
        axs1.set_title(f"{components[i]}", fontsize=25)
        axs1.set_xlabel('Frequency (Hz)')
        axs1.set_xticks([0.01, 0.1, 1, 10, 100])
        axs1.set_xticklabels(axs1.get_xticks(), fontsize=15)
        
        if fft_y_max > 100:
            ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
            labels = ["0.001", "0.01", "0.1", "1", "10", "100", "1000"]
            axs1.set_ylim(1e-3, 2e3)
        elif fft_y_max < 10:
            ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            labels = ["0.0001", "0.001", "0.01", "0.1", "1", "10"]
            axs1.set_ylim(1e-4, 20)
        else:
            ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
            labels = ["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"]
            axs1.set_ylim(1e-4, 200)
        
        axs1.set_yticks(ticks) 
        axs1.set_yticklabels(labels)

        #axs1.yaxis.set_major_locator(ScalarFormatter())
        #axs1.yaxis.set_major_formatter(ScalarFormatter())
        
        if normalized:
            axs1.set_ylabel(f"a(f)[1]", fontsize=25)
        else:
            axs1.set_ylabel(f"a(f)[m/s²]", fontsize=25)
        axs1.set_ylabel(f"a(f)[m/s²]", fontsize=25)
        axs1.set_xlabel("f[Hz]", fontsize=25)
        
        axs1.legend(fontsize=15, loc='lower left', frameon=False)
        axs1.grid(True)
        axs1.set_xlim(0, xlim)
        
        fig1.canvas.draw()
        
        plt.tight_layout()
        
        if station_name is not None:
            title = f'{path}/{station_name}_Frequency_{idx}_Spectrumloglog{components[i]}.{format}'
        else:
            if normalized:
                title = f'{path}/Normalized_Frequency_{idx}_Spectrumloglog{components[i]}.{format}'
            else:
                title = f'{path}/Frequency_{idx}_Spectrumloglog{components[i]}.{format}'
        print(f"Frequency graph saved to {title}")
        plt.savefig(title)
        plt.close(fig1)
        