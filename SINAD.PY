import numpy as np
from scipy import signal
from gnuradio import gr

class sinad_calculator(gr.sync_block):
    """
    Bloque de GNU Radio personalizado para calcular la relación SINAD (Signal-to-Noise
    and Distortion).

    Utiliza un filtro Notch IIR para atenuar la señal principal y medir la
    potencia del ruido y la distorsión. La potencia total se mide antes del filtro.
    Se aplica un promedio móvil a las mediciones de potencia para mayor estabilidad.

    Parámetros:
    - sample_rate: Frecuencia de muestreo de la señal de entrada (Hz).
    - signal_freq: Frecuencia de la señal principal a medir (Hz).
    - num_samples: Número de muestras por bloque para el cálculo de potencia.
    - q_param: Factor Q del filtro Notch IIR (determina la selectividad del filtro).
    - mov_avg_len: Longitud del promedio móvil para las mediciones de potencia.
    """
    def __init__(self, sample_rate=48000.0, signal_freq=1000.0, num_samples=1024, q_param=40.0, mov_avg_len=50):
        # Inicializa el bloque síncrono de GNU Radio.
        # Una entrada (float), una salida (float para SINAD en dB).
        gr.sync_block.__init__(
            self,
            name='SINAD',
            in_sig=[np.float32], # Asumimos entrada float. Cambiar a np.complex64 si la señal es compleja.
            out_sig=[np.float32] # Salida SINAD en dB
        )

        # Guarda los parámetros del bloque
        self.sample_rate = float(sample_rate)
        self.signal_freq = float(signal_freq)
        self.num_samples = int(num_samples)
        self.q_param = float(q_param)
        self.mov_avg_len = int(mov_avg_len)

        # Asegura que la longitud del promedio móvil sea al menos 1
        if self.mov_avg_len < 1:
            self.mov_avg_len = 1

        # Historial para el promedio móvil de la potencia total (S+N+D)
        self.power_total_history = np.zeros(self.mov_avg_len)
        self.power_total_idx = 0

        # Historial para el promedio móvil de la potencia de ruido y distorsión (N+D)
        self.power_nd_history = np.zeros(self.mov_avg_len)
        self.power_nd_idx = 0

        # Diseña el filtro Notch IIR
        # El filtro Notch se usará para eliminar la señal principal y así medir N+D.
        try:
            self.b_notch, self.a_notch = signal.iirnotch(
                self.signal_freq, self.q_param, self.sample_rate
            )
        except ValueError as e:
            # Manejo de errores si los parámetros del filtro no son válidos
            print(f"Error al diseñar el filtro Notch: {e}")
            # Establece coeficientes de paso directo para evitar errores en work()
            self.b_notch = np.array([1.0])
            self.a_notch = np.array([1.0])

        # Estado del filtro Notch (para mantener la continuidad entre llamadas a work)
        self.zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch) * 0.0


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

            # 1. Calcular la potencia total (S+N+D) del bloque actual
            # Para señales reales: potencia = media(abs(señal)^2)
            power_total_current = np.mean(np.abs(current_block)**2)

            # Actualizar el historial del promedio móvil para la potencia total
            self.power_total_history[self.power_total_idx] = power_total_current
            self.power_total_idx = (self.power_total_idx + 1) % self.mov_avg_len
            avg_power_total = np.mean(self.power_total_history)

            # 2. Aplicar el filtro Notch IIR para obtener solo Ruido + Distorsión (N+D)
            # Usamos lfilter y mantenemos el estado (zi) para continuidad
            filtered_nd_signal, self.zi_notch = signal.lfilter(
                self.b_notch, self.a_notch, current_block, zi=self.zi_notch
            )

            # 3. Calcular la potencia de Ruido + Distorsión (N+D)
            power_nd_current = np.mean(np.abs(filtered_nd_signal)**2)

            # Actualizar el historial del promedio móvil para la potencia N+D
            self.power_nd_history[self.power_nd_idx] = power_nd_current
            self.power_nd_idx = (self.power_nd_idx + 1) % self.mov_avg_len
            avg_power_nd = np.mean(self.power_nd_history)

            # 4. Calcular SINAD
            sinad_val_linear = 0.0
            if avg_power_nd > 1e-12: # Evitar división por cero o valores muy pequeños
                sinad_val_linear = avg_power_total / avg_power_nd
            else:
                # Si N+D es muy bajo, SINAD es muy alto (idealmente infinito)
                sinad_val_linear = 1e12 # Un valor grande para representar "infinito"

            sinad_db = 10 * np.log10(sinad_val_linear)

            # Asigna el valor de SINAD a la salida
            # Cada bloque de num_samples de entrada produce 1 valor de SINAD
            if processed_samples < num_output_items:
                out0[processed_samples] = sinad_db
                processed_samples += 1
            else:
                break # Ya hemos llenado el buffer de salida

        return processed_samples # Retorna el número de muestras de salida producidas

