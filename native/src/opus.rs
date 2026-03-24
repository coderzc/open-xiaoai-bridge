use std::convert::TryFrom;
use std::sync::Mutex;

use audiopus::coder::{Decoder, Encoder};
use audiopus::packet::Packet;
use audiopus::{Application, Channels, MutSignals, SampleRate};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn parse_channels(channels: u8) -> PyResult<Channels> {
    match channels {
        1 => Ok(Channels::Mono),
        2 => Ok(Channels::Stereo),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "channels must be 1 or 2",
        )),
    }
}

fn parse_sample_rate(sample_rate: u32) -> PyResult<SampleRate> {
    SampleRate::try_from(sample_rate as i32).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "sample_rate must be one of: 8000, 12000, 16000, 24000, 48000",
        )
    })
}

/// Opus encoder exposed to Python via PyO3.
#[pyclass]
pub struct OpusEncoder {
    encoder: Mutex<Encoder>,
}

#[pymethods]
impl OpusEncoder {
    /// Create a new Opus encoder.
    ///
    /// Args:
    ///     sample_rate: Sample rate in Hz (8000, 12000, 16000, 24000, 48000)
    ///     channels: Number of channels (1 or 2)
    #[new]
    fn new(sample_rate: u32, channels: u8) -> PyResult<Self> {
        let sr = parse_sample_rate(sample_rate)?;
        let ch = parse_channels(channels)?;
        let encoder = Encoder::new(sr, ch, Application::Audio).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Opus encoder init failed: {}", e))
        })?;
        Ok(Self {
            encoder: Mutex::new(encoder),
        })
    }

    /// Encode PCM data (16-bit signed LE) into an Opus frame.
    ///
    /// Args:
    ///     pcm_data: Raw PCM bytes (16-bit signed little-endian)
    ///     frame_size: Number of samples per channel
    ///
    /// Returns:
    ///     Encoded Opus frame as bytes
    fn encode<'py>(
        &self,
        py: Python<'py>,
        pcm_data: &[u8],
        frame_size: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if pcm_data.len() < frame_size * 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pcm_data too short for given frame_size",
            ));
        }
        let samples: Vec<i16> = pcm_data[..frame_size * 2]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        let mut output = vec![0u8; 4000];
        let encoder = self.encoder.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Encoder lock poisoned: {}", e))
        })?;
        let len = encoder.encode(&samples, &mut output).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Opus encode failed: {}", e))
        })?;

        Ok(PyBytes::new(py, &output[..len]))
    }
}

/// Opus decoder exposed to Python via PyO3.
#[pyclass]
pub struct OpusDecoder {
    decoder: Mutex<Decoder>,
}

#[pymethods]
impl OpusDecoder {
    /// Create a new Opus decoder.
    ///
    /// Args:
    ///     sample_rate: Sample rate in Hz (8000, 12000, 16000, 24000, 48000)
    ///     channels: Number of channels (1 or 2)
    #[new]
    fn new(sample_rate: u32, channels: u8) -> PyResult<Self> {
        let sr = parse_sample_rate(sample_rate)?;
        let ch = parse_channels(channels)?;
        let decoder = Decoder::new(sr, ch).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Opus decoder init failed: {}", e))
        })?;
        Ok(Self {
            decoder: Mutex::new(decoder),
        })
    }

    /// Decode an Opus frame into PCM data (16-bit signed LE).
    ///
    /// Args:
    ///     opus_data: Encoded Opus frame bytes
    ///     frame_size: Number of samples per channel to decode
    ///     decode_fec: Whether to use forward error correction
    ///
    /// Returns:
    ///     Decoded PCM bytes (16-bit signed little-endian)
    fn decode<'py>(
        &self,
        py: Python<'py>,
        opus_data: &[u8],
        frame_size: usize,
        decode_fec: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let packet = Packet::try_from(opus_data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid Opus packet: {}", e))
        })?;

        let mut output = vec![0i16; frame_size];
        let mut decoder = self.decoder.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Decoder lock poisoned: {}", e))
        })?;
        let decoded = decoder
            .decode(
                Some(packet),
                MutSignals::try_from(&mut output[..]).unwrap(),
                decode_fec,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Opus decode failed: {}", e))
            })?;

        let bytes: Vec<u8> = output[..decoded]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();

        Ok(PyBytes::new(py, &bytes))
    }
}

/// Register opus classes into the Python module.
pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OpusEncoder>()?;
    m.add_class::<OpusDecoder>()?;
    Ok(())
}
