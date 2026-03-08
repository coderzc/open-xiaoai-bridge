pub type AppError = Box<dyn std::error::Error>;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
