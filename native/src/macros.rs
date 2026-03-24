/// 提供一个简便的日志记录宏，将消息传递给 Python 端
///
/// # 示例
///
/// ```
/// pylog!("这是一条日志消息");
/// pylog!("带有变量的日志: {}", variable);
/// ```
#[macro_export]
macro_rules! pylog {
    ($($arg:tt)*) => {
        crate::python::PythonManager::instance().log(format!($($arg)*));
    };
}

#[macro_export]
macro_rules! pylog_error {
    ($($arg:tt)*) => {
        crate::python::PythonManager::instance().log_error(format!($($arg)*));
    };
}
