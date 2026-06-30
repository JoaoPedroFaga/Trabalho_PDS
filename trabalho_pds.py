import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq  # <-- NOVA IMPORTAÇÃO PARA O ESPECTRO

# ==========================================
# Exercício A: Função Genérica da HHT
# ==========================================
def calcular_hht(sinal, fs):
    """
    Aplica a Transformada de Hilbert-Huang a um sinal de qualquer tamanho.
    Retorna as IMFs, as amplitudes instantâneas e as frequências instantâneas.
    """
    emd = EMD()
    imfs = emd.emd(sinal, t)
    
    amplitudes_inst = []
    frequencias_inst = []
    
    for imf in imfs:
        sinal_analitico = hilbert(imf)
        
        amp = np.abs(sinal_analitico)
        amplitudes_inst.append(amp)
        
        fase_inst = np.unwrap(np.angle(sinal_analitico))
        freq = (np.diff(fase_inst) / (2.0 * np.pi) * fs)
        freq = np.append(freq, freq[-1]) 
        frequencias_inst.append(freq)
        
    return imfs, np.array(amplitudes_inst), np.array(frequencias_inst), t

# ==========================================
# Exercício B: Algoritmo de Detecção
# ==========================================
def detectar_insercao_cargas(imfs, amplitudes_inst, t):
    """
    Determina o instante de inserção procurando a maior descontinuidade
    (transiente) na amplitude instantânea do IMF 1, ignorando os efeitos de borda.
    """
    amp_imf1 = amplitudes_inst[0]
    margem = int(len(amp_imf1) * 0.05)
    amp_util = amp_imf1[margem:-margem]
    
    derivada_amp = np.abs(np.diff(amp_util))
    indice_pico_util = np.argmax(derivada_amp)
    
    indice_insercao = margem + indice_pico_util
    tempo_insercao = t[indice_insercao]
    
    return indice_insercao, tempo_insercao

# ==========================================
# Aplicação Principal (Main)
# ==========================================
if __name__ == "__main__":
    caminho_arquivo = "sinal_7_semruido.csv" 
    
    try:
        df = pd.read_csv(caminho_arquivo, header=None) 
        sinal_medido = df.values.flatten()
        
        amostras_por_ciclo = 128
        frequencia_rede = 60
        fs = amostras_por_ciclo * frequencia_rede  # Taxa de amostragem em Hz
        t = np.arange(len(sinal_medido))/fs
        #sinal_medido = np.sin(2*np.pi*frequencia_rede*t)

        N = len(sinal_medido) # Número total de amostras

        
        imfs, amplitudes, frequencias, t = calcular_hht(sinal_medido, fs)
        idx_insercao, tempo_insercao = detectar_insercao_cargas(imfs, amplitudes, t)
        # ---------------------------------------------------------
        # PLOTAGEM DAS IMFs (uma janela para cada IMF)
        # ---------------------------------------------------------

        n_imfs = len(imfs)

        for i in range(n_imfs):

            plt.figure(figsize=(10,4))

            plt.plot(t, imfs[i], 'b', linewidth=1)

            plt.title(f'IMF {i+1}')
            plt.xlabel('Tempo (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.xlabel('Tempo (s)')
            plt.suptitle('Funções Modais Intrínsecas (IMFs)')

            plt.tight_layout()

        if tempo_insercao is not None:
            print(f"Cargas não lineares inseridas na amostra {idx_insercao}.")
            print(f"Instante de tempo estimado: {tempo_insercao * 1000:.4f} milissegundos.")
        else:
            print("Não foi possível detectar a inserção de cargas.")

        # ---------------------------------------------------------
        # PLOTAGEM 1: Domínio do Tempo (Original e Envelopes)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(t, sinal_medido, 'b', label="Corrente Medida")
        if tempo_insercao:
            plt.axvline(x=tempo_insercao, color='r', linestyle='--', label="Inserção Detectada")
        plt.title("Sinal Original no Domínio do Tempo")
        plt.xlabel("Tempo (s)")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(t, amplitudes[0], 'g', label="Amplitude Instantânea (IMF 1)")
        if tempo_insercao:
            plt.axvline(x=tempo_insercao, color='r', linestyle='--')
        plt.title("Amplitude instantânea (IMF 1)")
        plt.xlabel("Tempo (s)")
        plt.legend()
        
        plt.tight_layout()

        # ---------------------------------------------------------
        # PLOTAGEM 2: ESPECTRO DE HILBERT
        # ---------------------------------------------------------

        t_total = []
        f_total = []
        a_total = []

        for i in range(len(imfs)):

            freq = frequencias[i]
            amp = amplitudes[i]

            mascara = (freq > 0) & (freq < 500)

            t_total.append(t[mascara])
            f_total.append(freq[mascara])
            a_total.append(amp[mascara])
        

        t_h = np.concatenate(t_total)
        f_h = np.concatenate(f_total)
        a_h = np.concatenate(a_total)

        plt.figure(figsize=(10,6))

        plt.scatter(
            t_h,
            f_h,
            c=a_h,
            s=4,
            cmap='jet'
        )

        plt.colorbar(label='Amplitude')

        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequência (Hz)')
        plt.title('Espectro de Hilbert')
        plt.axhline(
        60,
        color='purple',
        linestyle='--',
        linewidth=2,
        label='Fundamental 60 Hz'
        )
        if tempo_insercao:
            plt.axvline(
                tempo_insercao,
                color='white',
                linestyle='--',
                linewidth=2
            )

        plt.ylim(0,500)
        plt.grid(True)
        # ---------------------------------------------------------
        # PLOTAGEM DO SINAL ORIGINAL (SEM MARCAÇÃO)
        # ---------------------------------------------------------

        plt.figure(figsize=(10,4))

        plt.plot(
            t,
            sinal_medido,
            color='b',
            linewidth=1
        )

        plt.title('Sinal Original')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(e)