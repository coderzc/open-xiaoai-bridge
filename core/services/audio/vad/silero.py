import numpy as np
import onnxruntime as ort

from core.utils.file import get_model_file_path


class OnnxWrapper:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if len(x.shape) > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk {len(x.shape)}"
            )

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)
        if sr in [8000, 16000]:
            ort_inputs = {
                "input": x,
                "state": self._state,
                "sr": np.array(sr, dtype="int64"),
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return out


class _Silero:
    def __init__(self) -> None:
        self.model = OnnxWrapper(
            path=get_model_file_path("silero_vad.onnx"),
        )

    def vad(self, frames, sample_rate):
        try:
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            return self.model(audio_float32, sample_rate).item()
        except Exception as e:
            # 打印详细错误信息
            import traceback
            print(f"[Silero VAD Error] {str(e)}")
            print(f"[Silero VAD Error] frames length: {len(frames) if frames else 'None'}")
            print(f"[Silero VAD Error] sample_rate: {sample_rate}")
            traceback.print_exc()
            return None


Silero = _Silero()
