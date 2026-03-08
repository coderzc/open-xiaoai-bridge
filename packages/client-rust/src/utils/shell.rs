use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::base::AppError;

#[derive(Debug, Serialize, Deserialize)]
pub struct CommandResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub async fn run_shell(script: &str) -> Result<CommandResult, AppError> {
    let output = Command::new("/bin/sh")
        .arg("-c")
        .arg(script)
        .output()
        .await?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    Ok(CommandResult {
        stdout,
        stderr,
        exit_code,
    })
}
