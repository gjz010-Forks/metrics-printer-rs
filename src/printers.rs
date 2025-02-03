use crate::PrintRecorder;

/// Logic to get the stringified metrics to some output device.
pub trait Printer {
    /// Print the given `metrics_string` to this output device.
    fn print_metrics(&self, metrics_string: String);
}

/// Prints metrics to stdout
pub struct StdoutPrinter;
impl Printer for StdoutPrinter {
    fn print_metrics(&self, metrics_string: String) {
        println!("{}", metrics_string);
    }
}
/// Prints metrics to stderr
pub struct StderrPrinter;
impl Printer for StderrPrinter {
    fn print_metrics(&self, metrics_string: String) {
        eprintln!("{}", metrics_string);
    }
}

impl Default for PrintRecorder<StdoutPrinter> {
    /// New PrinterRecorder to stdout with 1s interval no metadata printing.
    fn default() -> Self {
        PrintRecorder::new(StdoutPrinter)
    }
}
