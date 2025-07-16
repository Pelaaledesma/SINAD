import numpy as np
from scipy import signal
from gnuradio import gr

class snr_calculator(gr.sync_block):
    """
    Bloque de GNU Radio personalizado para calcular la relación SNR (Signal-to-Noise Ratio).

    Utiliza un filtro Pasa-Banda para aislar la señal (S) y un filtro Notch IIR
    para aislar el ruido (N). Se aplica un promedio móvil a las mediciones de potencia.

    Parámetros:
    - sample_rate: Frecuencia de muestreo de la señal de entrada (Hz).
    - signal_freq: Frecuencia de la señal principal a medir (Hz).
    - num_samples: Número de muestras por bloque para el cálculo de potencia.
    - q_notch: Factor Q del filtro Notch IIR (para aislar N).
    - q_bp: Factor Q del filtro Pasa-Banda IIR (para aislar S).
    - mov_avg_len: Longitud del promedio móvil para las mediciones de potencia.
    """
    def __init__(self, sample_rate=48000.0, signal_freq=1000.0, num_samples=1024, q_notch=40.0, q_bp=40.0, mov_avg_len=50):
        # Inicializa el bloque síncrono de GNU Radio.
        # Una entrada (float), una salida (float para SNR en dB).
        gr.sync_block.__init__(
            self,
            name='SNR',
            in_sig=[np.float32], # Asumimos entrada float. Cambiar a np.complex64 si la señal es compleja.
            out_sig=[np.float32] # Salida SNR en dB
        )

        # Guarda los parámetros del bloque
        self.sample_rate = float(sample_rate)
        self.signal_freq = float(signal_freq)
        self.num_samples = int(num_samples)
        self.q_notch = float(q_notch)
        self.q_bp = float(q_bp)
        self.mov_avg_len = int(mov_avg_len)

        # Asegura que la longitud del promedio móvil sea al menos 1
        if self.mov_avg_len < 1:
            self.mov_avg_len = 1

        # Historial para el promedio móvil de la potencia de la señal (S)
        self.power_s_history = np.zeros(self.mov_avg_len)
        self.power_s_idx = 0

        # Historial para el promedio móvil de la potencia del ruido (N)
        self.power_n_history = np.zeros(self.mov_avg_len)
        self.power_n_idx = 0

        # Diseña el filtro Notch IIR (para aislar el ruido)
        try:
            self.b_notch, self.a_notch = signal.iirnotch(
                self.signal_freq, self.q_notch, self.sample_rate
            )
        except ValueError as e:
            print(f"Error al diseñar el filtro Notch para SNR: {e}")
            self.b_notch = np.array([1.0])
            self.a_notch = np.array([1.0])

        # Diseña el filtro Pasa-Banda IIR (para aislar la señal)
        # Se crea una banda de paso estrecha alrededor de signal_freq
        nyquist = 0.5 * self.sample_rate
        lowcut = (self.signal_freq - (self.signal_freq / (2 * self.q_bp))) / nyquist
        highcut = (self.signal_freq + (self.signal_freq / (2 * self.q_bp))) / nyquist
        
        # Asegurarse de que las frecuencias de corte sean válidas
        if lowcut < 0: lowcut = 0.01 # Evitar frecuencia negativa
        if highcut >= 1: highcut = 0.99 # Evitar o exceder la frecuencia de Nyquist

        try:
            # 'bandpass' para un filtro pasa-banda
            self.b_bp, self.a_bp = signal.iirfilter(
                N=4,  # Orden del filtro, puedes ajustarlo
                Wn=[lowcut, highcut],
                btype='bandpass',
                ftype='butter', # Tipo de filtro, Butterworth es común
                fs=self.sample_rate
            )
        except ValueError as e:
            print(f"Error al diseñar el filtro Pasa-Banda para SNR: {e}")
            self.b_bp = np.array([1.0])
            self.a_bp = np.array([1.0])

        # Estados de los filtros (para mantener la continuidad entre llamadas a work)
        self.zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch) * 0.0
        self.zi_bp = signal.lfilter_zi(self.b_bp, self.a_bp) * 0.0


    def work(self, input_items, output_items):
        # Obtiene el buffer de entrada y salida
        in0 = input_items[0]
        out0 = output_items[0]

        # Asegura que tenemos suficientes muestras para procesar
        if len(in0) < self.num_samples:
            return 0 # No hay suficientes muestras para un cálculo completo

        # Procesar en bloques de num_samples
        num_output_items = len(out0)
        processed_samples = 0

        for i in range(0, len(in0), self.num_samples):
            current_block = in0[i : i + self.num_samples]

            if len(current_block) < self.num_samples:
                break # El último bloque puede ser más pequeño, lo ignoramos por ahora

            # 1. Aísla la señal (S) usando el filtro Pasa-Banda
            filtered_s_signal, self.zi_bp = signal.lfilter(
                self.b_bp, self.a_bp, current_block, zi=self.zi_bp
            )
            power_s_current = np.mean(np.abs(filtered_s_signal)**2)

            # Actualizar el historial del promedio móvil para la potencia de la señal
            self.power_s_history[self.power_s_idx] = power_s_current
            self.power_s_idx = (self.power_s_idx + 1) % self.mov_avg_len
            avg_power_s = np.mean(self.power_s_history)

            # 2. Aísla el ruido (N) usando el filtro Notch (elimina la señal principal)
            filtered_n_signal, self.zi_notch = signal.lfilter(
                self.b_notch, self.a_notch, current_block, zi=self.zi_notch
            )
            power_n_current = np.mean(np.abs(filtered_n_signal)**2)

            # Actualizar el historial del promedio móvil para la potencia del ruido
            self.power_n_history[self.power_n_idx] = power_n_current
            self.power_n_idx = (self.power_n_idx + 1) % self.mov_avg_len
            avg_power_n = np.mean(self.power_n_history)

            # 3. Calcular SNR
            snr_val_linear = 0.0
            if avg_power_n > 1e-12: # Evitar división por cero o valores muy pequeños
                snr_val_linear = avg_power_s / avg_power_n
            else:
                # Si el ruido es muy bajo, SNR es muy alto (idealmente infinito)
                snr_val_linear = 1e12 # Un valor grande para representar "infinito"

            snr_db = 10 * np.log10(snr_val_linear)

            # Asigna el valor de SNR a la salida
            if processed_samples < num_output_items:
                out0[processed_samples] = snr_db
                processed_samples += 1
            else:
                break # Ya hemos llenado el buffer de salida

        return processed_samples # Retorna el número de muestras de salida producidas

