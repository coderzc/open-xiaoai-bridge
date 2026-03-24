use pyo3::prelude::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, LazyLock, RwLock};

pub struct PythonManager {
    functions: Arc<RwLock<HashMap<String, PyObject>>>,
}

static INSTANCE: LazyLock<PythonManager> = LazyLock::new(PythonManager::new);

impl PythonManager {
    pub fn new() -> Self {
        Self {
            functions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn instance() -> &'static Self {
        &INSTANCE
    }

    fn register_fn(&self, key: &str, function: PyObject) -> PyResult<()> {
        let mut functions = self.functions.write().unwrap();
        functions.insert(key.to_string(), function);
        Ok(())
    }

    fn unregister_fn(&self, key: &str) -> PyResult<()> {
        let mut functions = self.functions.write().unwrap();
        functions.remove(key);
        Ok(())
    }

    pub fn call_fn(&self, key: &str, arg: Option<PyObject>) -> PyResult<PyObject> {
        let functions = self.functions.read().unwrap();
        if let Some(function) = functions.get(key) {
            Python::with_gil(|py| match arg {
                None => function.call0(py),
                Some(arg) => function.call1(py, (arg,)),
            })
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "未找到函数: {}",
                key
            )))
        }
    }

    pub fn log(&self, text: String) {
        Python::with_gil(|py| {
            let result: PyResult<()> = (|| {
                let module = py.import("core.utils.logger")?;
                let logger = module.getattr("logger")?;
                logger.call_method1("info", (text.as_str(),))?;
                Ok(())
            })();
            let _ = result;
        });
    }

    pub fn log_error(&self, text: String) {
        Python::with_gil(|py| {
            let result: PyResult<()> = (|| {
                let module = py.import("core.utils.logger")?;
                let logger = module.getattr("logger")?;
                logger.call_method1("error", (text.as_str(),))?;
                Ok(())
            })();
            let _ = result;
        });
    }

    pub fn eval(&self, script: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            let c_script = CString::new(script)?;
            py.run(&c_script, None, None)?;
            Ok(())
        })
    }
}

#[pyfunction]
fn register_fn(key: &str, function: PyObject) -> PyResult<()> {
    PythonManager::instance().register_fn(key, function)
}

#[pyfunction]
fn unregister_fn(key: &str) -> PyResult<()> {
    PythonManager::instance().unregister_fn(key)
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_fn, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_fn, m)?)?;
    Ok(())
}
