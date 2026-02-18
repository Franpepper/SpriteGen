use std::io::BufRead;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

pub struct ProcessHandle {
    child: Child,
    log_rx: mpsc::Receiver<String>,
}

impl ProcessHandle {
    /// Drain available log lines and check if the process is done.
    /// Returns (new_lines, is_finished, exit_success).
    pub fn poll(&mut self) -> (Vec<String>, bool, bool) {
        let mut lines = Vec::new();
        let mut disconnected = false;

        loop {
            match self.log_rx.try_recv() {
                Ok(line) => lines.push(line),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if disconnected {
            let success = self.child.wait().map(|s| s.success()).unwrap_or(false);
            return (lines, true, success);
        }

        (lines, false, false)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn spawn_process(args: &[String], cwd: &Path) -> Result<ProcessHandle, String> {
    if args.is_empty() {
        return Err("No command provided".into());
    }

    let mut cmd = Command::new(&args[0]);
    cmd.args(&args[1..])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn process: {}", e))?;

    let (tx, rx) = mpsc::channel();

    // Spawn reader thread for stdout
    if let Some(stdout) = child.stdout.take() {
        let tx = tx.clone();
        thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
    }

    // Spawn reader thread for stderr
    if let Some(stderr) = child.stderr.take() {
        let tx = tx.clone();
        thread::spawn(move || {
            let reader = std::io::BufReader::new(stderr);
            for line in reader.lines().flatten() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
    }

    // Drop the original sender so only thread-owned clones remain.
    // When all threads finish, the receiver gets Disconnected.
    drop(tx);

    Ok(ProcessHandle { child, log_rx: rx })
}
