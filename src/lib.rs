//! This crate provides you with a [metrics](metrics) recorder
//! that can print all metrics to a target of your choice in regular intervals.
//!
//! It uses a thread to print, so it doesn't interfere with other threads' work directly.
//!
//! Custom printing targets (e.g., logging frameworks) can be provided via the simple [Printer](Printer)
//! trait, while default implementations for [stdout](StdoutPrinter) and [stderr](StderrPrinter) are provided.
//!
//! # Example
//!
//! ```
//! # use std::thread;
//! # use std::time::Duration;
//! use metrics::*;
//! use metrics_printer::*;
//!
//! PrintRecorder::default().install().unwrap();
//! for _i in 0..300 {
//!     counter!("test.counter").increment(1);
//!     std::thread::sleep(Duration::from_millis(10));
//! }
//! ```
#![deny(missing_docs)]
mod printers;
mod recorder;
mod recorder_wrapper;
mod snapshot;
pub use printers::{Printer, StderrPrinter, StdoutPrinter};
pub use recorder::{default_quantiles, PrintRecorder, DEFAULT_PRINT_INTERVAL};

/// Load and install the default recorder
pub fn init() {
    PrintRecorder::default().install_if_free();
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::*;
    use std::time::{Duration, Instant};

    #[test]
    fn not_a_real_test() {
        #[allow(unused_mut)]
        let mut rec = PrintRecorder::default();
        rec.do_print_metadata();
        // uncomment to see units and descriptions
        //rec.do_print_medata();
        rec.install().unwrap();

        describe_counter!("test.counter", "A simple counter in a loop");
        describe_gauge!(
            "test.time_elapsed",
            Unit::Milliseconds,
            "The time that elapsed since starting the loop"
        );
        describe_histogram!(
            "test.time_per_iter",
            Unit::Nanoseconds,
            "The time that elapsed for every loop"
        );
        let start = Instant::now();
        let mut elapsed = start.elapsed();
        let mut last_elapsed = Duration::new(0, 0);
        while elapsed < Duration::from_secs(5) {
            let since_last_iter = elapsed - last_elapsed;
            counter!("test.counter", "test" => "not_a_real_test").increment(1);
            gauge!("test.time_elapsed", "test" => "not_a_real_test")
                .increment(elapsed.as_millis() as f64);
            histogram!("test.time_per_iter", "test" => "not_a_real_test")
                .record(since_last_iter.as_nanos() as f64);
            last_elapsed = elapsed;
            elapsed = start.elapsed();
        }
    }
}
