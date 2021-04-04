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
//! register_counter!("test.counter");
//! for _i in 0..300 {
//!     increment_counter!("test.counter");
//!     std::thread::sleep(Duration::from_millis(10));
//! }
//! ```
#![deny(missing_docs)]

use metrics::{GaugeValue, Key, NameParts, Recorder, SetRecorderError, Unit};
use metrics_util::{CompositeKey, Handle, MetricKind, Quantile, Registry, Summary};
use std::{
    collections::HashMap,
    fmt::Write,
    iter::FromIterator,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// The default interval between printing metrics
pub const DEFAULT_PRINT_INTERVAL: Duration = Duration::from_millis(1000);

/// The default set of quantiles to print.
///
/// Prints histograms as [min, median, max].
pub fn default_quantiles() -> Box<[Quantile]> {
    Box::new([Quantile::new(0.0), Quantile::new(0.5), Quantile::new(1.0)])
}

/// Load and install the default recorder
pub fn init() {
    PrintRecorder::default().install_if_free();
}

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

/// A metrics recorder that collects metrics and regularly tried to print them.
pub struct PrintRecorder<P>
where
    P: Printer + Send + Sync + 'static,
{
    registry: Registry<CompositeKey, Handle>,
    metdadata: Mutex<HashMap<NameParts, MetaDataEntry>>,
    printer: P,
    print_interval: Duration,
    print_metadata: bool,
    quantiles: Box<[Quantile]>,
}

impl Default for PrintRecorder<StdoutPrinter> {
    /// New PrinterRecorder to stdout with 1s interval no metadata printing.
    fn default() -> Self {
        PrintRecorder::new(StdoutPrinter)
    }
}

impl<P> PrintRecorder<P>
where
    P: Printer + Send + Sync + 'static,
{
    /// New PrinterRecorder with 1s interval no metadata printing.
    pub fn new(printer: P) -> Self {
        PrintRecorder {
            registry: Registry::new(),
            metdadata: Mutex::new(HashMap::new()),
            printer,
            print_interval: DEFAULT_PRINT_INTERVAL,
            print_metadata: false,
            quantiles: default_quantiles(),
        }
    }

    /// Set how often metrics should be printed (roughly)
    pub fn set_print_interval(&mut self, interval: Duration) -> &mut Self {
        self.print_interval = interval;
        self
    }

    /// Print units and descriptions together with the metrics
    pub fn do_print_medata(&mut self) -> &mut Self {
        self.print_metadata = true;
        self
    }

    /// Do not print units and descriptions together with the metrics
    pub fn skip_print_medata(&mut self) -> &mut Self {
        self.print_metadata = true;
        self
    }

    /// Select which quantiles should be printed
    pub fn select_quantiles(&mut self, quantiles: Box<[Quantile]>) -> &mut Self {
        self.quantiles = quantiles;
        self
    }

    fn insert_metadata(&self, key: NameParts, data: MetaDataEntry) {
        let mut guard = self
            .metdadata
            .lock()
            .expect("Could not acquire metadata lock");
        guard.insert(key, data);
    }

    /// Register this recorder as the global recorder,
    /// if no other recorder is already registered.
    ///
    /// If another recorder is registered, this will fail silently.
    /// This method is recommended to be used when you are trying to set the exact same
    /// recorder from multiple places and don't know/care which will get executed first.
    /// This is often the case for unit tests, for example.
    ///
    /// Also starts the background thread for printing.
    pub fn install_if_free(self) {
        if metrics::try_recorder().is_none() {
            let _ = self.install(); // ignore result
        }
    }

    /// Register this recorder as the global recorder,
    /// if no other recorder is already registered.
    ///
    /// If another recorder is registered, this will return an error.
    ///
    /// Also starts the background thread for printing.
    pub fn install(self) -> Result<(), SetRecorderError> {
        let arced = Arc::new(self);
        let wrapped = PrintRecorderWrapper(arced.clone());
        // this can still fail due to parallelism
        let res = metrics::set_boxed_recorder(Box::new(wrapped));
        if res.is_ok() {
            Self::start_thread(arced);
        }
        res
    }

    fn start_thread(arc_self: Arc<Self>) {
        std::thread::Builder::new()
            .name("stdout-recorder".to_string())
            .spawn(move || arc_self.run_loop())
            .expect("recorder thread");
    }

    fn run_loop(&self) {
        let mut last_snapshot = self.take_snapshot(HashMap::new());
        loop {
            let start = Instant::now();
            std::thread::sleep(self.print_interval);
            let sleep_time = start.elapsed();
            let snapshot = self.take_snapshot(last_snapshot.clone_summaries());
            let metrics_string = self.stringify_metrics(sleep_time, &last_snapshot, &snapshot);
            self.printer.print_metrics(metrics_string);
            last_snapshot = snapshot;
        }
    }

    fn take_snapshot(&self, mut summaries: HashMap<CompositeKey, Summary>) -> Snapshot {
        self.registry.map_collect(|key, _gen, handle| {
            let value = match key.kind() {
                MetricKind::Counter => SnapshotValue::Counter(handle.read_counter()),
                MetricKind::Gauge => SnapshotValue::Gauge(handle.read_gauge()),
                MetricKind::Histogram => {
                    let mut summary = summaries.remove(key).unwrap_or_else(Summary::with_defaults);
                    handle.read_histogram_with_clear(|entries| {
                        for entry in entries {
                            summary.add(*entry);
                        }
                    });
                    SnapshotValue::Histogram(Box::new(summary))
                }
            };
            (key.clone(), value)
        })
    }

    fn stringify_metrics(
        &self,
        time_since_last_print: Duration,
        previous_snapshot: &Snapshot,
        current_snapshot: &Snapshot,
    ) -> String {
        let mut joined = current_snapshot.join(&self.quantiles, previous_snapshot);
        joined.sort_unstable_by(|a, b| a.key().cmp(b.key()));
        let mut rows: Vec<([String; 5], Option<[String; 2]>)> = Vec::with_capacity(joined.len());
        let mut longest_key: usize = "Key".len();
        let longest_kind: usize = "Histogram".len();
        let mut longest_value: usize = "Value".len();
        let mut longest_delta: usize = "Delta".len();
        let mut longest_unit: usize = "Units".len();
        let mut longest_description: usize = "Description".len();
        for entry in joined.into_iter() {
            let key: String = entry.key_str();
            longest_key = longest_key.max(key.len());
            let kind: String = entry.kind_str().to_string();
            let value: String = entry.current_str();
            longest_value = longest_value.max(value.len());
            let delta: String = entry.delta_sec_timed_str(time_since_last_print.as_secs_f64());
            longest_delta = longest_delta.max(delta.len());
            let labels: Vec<String> = entry
                .key()
                .labels()
                .map(|label| format!("{} => {}", label.key(), label.value()))
                .collect();
            let row = [key, kind, value, delta, labels.join(", ")];
            if self.print_metadata {
                let guard = self.metdadata.lock().unwrap();
                if let Some(metadata) = guard.get(entry.key().name()) {
                    let unit = metadata
                        .unit
                        .as_ref()
                        .map(|u| u.as_canonical_label().to_string())
                        .unwrap_or_else(|| "N/A".to_string());
                    longest_unit = longest_unit.max(unit.len());
                    let description = metadata
                        .description
                        .map(|d| d.to_string())
                        .unwrap_or_else(|| "N/A".to_string());
                    longest_description = longest_description.max(description.len());
                    rows.push((row, Some([unit, description])));
                } else {
                    rows.push((row, Some(["N/A".to_string(), "N/A".to_string()])));
                }
            } else {
                rows.push((row, None));
            }
        }

        let mut output = format!("{:=^80}\n\n", " Metrics ");
        if self.print_metadata {
            writeln!(
                        &mut output,
                        "{key:<key_fill$} {kind:<kind_fill$} {value:<value_fill$} {unit:<unit_fill$} 𝚫 {delta:<delta_fill$} | {descr:<descr_fill$} | Labels",
                        key = "Key",
                        key_fill = longest_key,
                        kind = "Kind",
                        kind_fill = longest_kind,
                        value = "Value",
                        value_fill = longest_value,
                        unit = "Units",
                        unit_fill = longest_unit,
                        delta = "Delta",
                        delta_fill = longest_delta,
                        descr = "Description",
                        descr_fill = longest_description,
                    )
                    .unwrap();
        } else {
            writeln!(
                        &mut output,
                        "{key:<key_fill$} {kind:<kind_fill$} {value:<value_fill$} 𝚫 {delta:<delta_fill$} | Labels",
                        key = "Key",
                        key_fill = longest_key,
                        kind = "Kind",
                        kind_fill = longest_kind,
                        value = "Value",
                        value_fill = longest_value,
                        delta = "Delta",
                        delta_fill = longest_delta
                    )
                    .unwrap();
        }
        writeln!(&mut output, "{:-^80}", "").unwrap();
        for (row, meta_row_opt) in rows.into_iter() {
            if let Some(meta_row) = meta_row_opt {
                writeln!(
                        &mut output,
                        "{key:<key_fill$} {kind:<kind_fill$} {value:<value_fill$} {unit:<unit_fill$} 𝚫 {delta:<delta_fill$} | {descr:<descr_fill$} | {labels}",
                        key = row[0],
                        key_fill = longest_key,
                        kind = row[1],
                        kind_fill = longest_kind,
                        value = row[2],
                        value_fill = longest_value,
                        unit = meta_row[0],
                        unit_fill = longest_unit,
                        delta = row[3],
                        delta_fill = longest_delta,
                        descr = meta_row[1],
                        descr_fill = longest_description,
                        labels = row[4]
                    )
                    .unwrap();
            } else {
                writeln!(
                        &mut output,
                        "{key:<key_fill$} {kind:<kind_fill$} {value:<value_fill$} 𝚫 {delta:<delta_fill$} | {labels}",
                        key = row[0],
                        key_fill = longest_key,
                        kind = row[1],
                        kind_fill = longest_kind,
                        value = row[2],
                        value_fill = longest_value,
                        delta = row[3],
                        delta_fill = longest_delta,
                        labels = row[4]
                    )
                    .unwrap();
            }
        }
        writeln!(
            &mut output,
            "\n{:=^80}",
            format!(" After {:.3} s ", time_since_last_print.as_secs_f64())
        )
        .unwrap();

        output
    }
}

struct MetaDataEntry {
    unit: Option<Unit>,
    description: Option<&'static str>,
}
impl MetaDataEntry {
    fn new(unit: Option<Unit>, description: Option<&'static str>) -> Self {
        MetaDataEntry { unit, description }
    }
}
// impl Default for MetaDataEntry {
//  fn default() -> Self {
//      MetaDataEntry {
//          unit: None,
//          description: None,
//      }
//  }
// }

struct PrintRecorderWrapper<P>(Arc<PrintRecorder<P>>)
where
    P: Printer + Send + Sync + 'static;

impl<P> Recorder for PrintRecorderWrapper<P>
where
    P: Printer + Send + Sync + 'static,
{
    fn register_counter(&self, key: Key, unit: Option<Unit>, description: Option<&'static str>) {
        let key_name: NameParts = key.name().clone();
        let k = CompositeKey::new(MetricKind::Counter, key);
        self.0.registry.op(k, ignore, Handle::counter);
        let metadata = MetaDataEntry::new(unit, description);
        self.0.insert_metadata(key_name, metadata);
    }

    fn register_gauge(&self, key: Key, unit: Option<Unit>, description: Option<&'static str>) {
        let key_name: NameParts = key.name().clone();
        let k = CompositeKey::new(MetricKind::Gauge, key);
        self.0.registry.op(k, ignore, Handle::gauge);
        let metadata = MetaDataEntry::new(unit, description);
        self.0.insert_metadata(key_name, metadata);
    }

    fn register_histogram(&self, key: Key, unit: Option<Unit>, description: Option<&'static str>) {
        let key_name: NameParts = key.name().clone();
        let k = CompositeKey::new(MetricKind::Histogram, key);
        self.0.registry.op(k, ignore, Handle::histogram);
        let metadata = MetaDataEntry::new(unit, description);
        self.0.insert_metadata(key_name, metadata);
    }

    fn increment_counter(&self, key: Key, value: u64) {
        let k = CompositeKey::new(MetricKind::Counter, key);
        self.0
            .registry
            .op(k, |handle| handle.increment_counter(value), Handle::counter);
    }

    fn update_gauge(&self, key: Key, value: GaugeValue) {
        let k = CompositeKey::new(MetricKind::Gauge, key);
        self.0
            .registry
            .op(k, |handle| handle.update_gauge(value), Handle::gauge);
    }

    fn record_histogram(&self, key: Key, value: f64) {
        let k = CompositeKey::new(MetricKind::Histogram, key);
        self.0.registry.op(
            k,
            |handle| handle.record_histogram(value),
            Handle::histogram,
        );
    }
}

struct Snapshot {
    values: HashMap<CompositeKey, SnapshotValue>,
}

impl Snapshot {
    fn clone_summaries(&self) -> HashMap<CompositeKey, Summary> {
        self.values
            .iter()
            .flat_map(|(key, value)| {
                if let SnapshotValue::Histogram(summary) = value {
                    let s: Summary = summary.as_ref().clone();
                    Some((key.clone(), s))
                } else {
                    None
                }
            })
            .collect()
    }

    fn join(&self, quantiles: &[Quantile], other: &Snapshot) -> Vec<DeltaEntry> {
        let mut joined = Vec::with_capacity(self.values.len());
        for (key, value) in self.values.iter() {
            let entry = match other.values.get(key) {
                Some(other_value) => match (value, other_value) {
                    (SnapshotValue::Counter(self_v), SnapshotValue::Counter(other_v)) => {
                        DeltaEntry::Counter {
                            key: key.key().clone(),
                            current: *self_v,
                            delta: self_v - other_v,
                        }
                    }
                    (SnapshotValue::Gauge(self_v), SnapshotValue::Gauge(other_v)) => {
                        DeltaEntry::Gauge {
                            key: key.key().clone(),
                            current: *self_v,
                            delta: *self_v - *other_v,
                        }
                    }
                    (SnapshotValue::Histogram(self_h), SnapshotValue::Histogram(other_h)) => {
                        DeltaEntry::Histogram {
                            key: key.key().clone(),
                            quantiles: quantiles.iter().cloned().collect(),
                            current: self_h.clone(),
                            delta: HistogramDelta::from_join(quantiles, other_h, self_h),
                        }
                    }
                    _ => {
                        unreachable!("Same keys must have same types in all snapshots!")
                    }
                },
                None => match value {
                    SnapshotValue::Counter(v) => DeltaEntry::Counter {
                        key: key.key().clone(),
                        current: *v,
                        delta: *v,
                    },
                    SnapshotValue::Gauge(v) => DeltaEntry::Gauge {
                        key: key.key().clone(),
                        current: *v,
                        delta: *v,
                    },
                    SnapshotValue::Histogram(h) => DeltaEntry::Histogram {
                        key: key.key().clone(),
                        quantiles: quantiles.iter().cloned().collect(),
                        current: h.clone(),
                        delta: HistogramDelta::from_summary(quantiles, h),
                    },
                },
            };
            joined.push(entry);
        }
        joined
    }
}

impl FromIterator<(CompositeKey, SnapshotValue)> for Snapshot {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (CompositeKey, SnapshotValue)>,
    {
        let mut values = HashMap::new();
        for (key, value) in iter {
            values.insert(key, value);
        }
        Snapshot { values }
    }
}

enum SnapshotValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Box<Summary>),
}

struct HistogramDelta {
    quantiles: Box<[(Quantile, f64)]>,
    new_samples: usize,
}
impl HistogramDelta {
    fn from_summary(quantiles: &[Quantile], summary: &Summary) -> Self {
        let quantile_values: Box<[(Quantile, f64)]> = collect_quantiles(quantiles, summary);
        // delta from nothing
        HistogramDelta {
            quantiles: quantile_values,
            new_samples: summary.count(),
        }
    }

    fn from_join(quantiles: &[Quantile], old: &Summary, new: &Summary) -> Self {
        let quantile_values: Box<[(Quantile, f64)]> = quantiles
            .iter()
            .map(|q| {
                let v =
                    new.quantile(q.value()).unwrap_or(0.0) - old.quantile(q.value()).unwrap_or(0.0);
                (q.clone(), v)
            })
            .collect();
        HistogramDelta {
            quantiles: quantile_values,
            new_samples: new.count() - old.count(),
        }
    }
}

enum DeltaEntry {
    Counter {
        key: Key,
        current: u64,
        // this is always positive because counter is increasing only
        delta: u64,
    },
    Gauge {
        key: Key,
        current: f64,
        delta: f64,
    },
    Histogram {
        key: Key,
        quantiles: Box<[Quantile]>,
        current: Box<Summary>,
        delta: HistogramDelta,
    },
}

impl DeltaEntry {
    fn key(&self) -> &Key {
        match self {
            DeltaEntry::Counter { key, .. } => key,
            DeltaEntry::Gauge { key, .. } => key,
            DeltaEntry::Histogram { key, .. } => key,
        }
    }

    fn key_str(&self) -> String {
        format! {"{}", self.key().name()}
    }

    fn kind_str(&self) -> &'static str {
        match self {
            DeltaEntry::Counter { .. } => "Counter",
            DeltaEntry::Gauge { .. } => "Gauge",
            DeltaEntry::Histogram { .. } => "Histogram",
        }
    }

    fn current_str(&self) -> String {
        match self {
            DeltaEntry::Counter { current, .. } => format!("{}", current),
            DeltaEntry::Gauge { current, .. } => format!("{}", current),
            DeltaEntry::Histogram {
                current, quantiles, ..
            } => {
                let quantile_values = collect_quantiles(quantiles, current);
                format!(
                    "[{}]#{}",
                    quantile_strings(&quantile_values, false).join(", "),
                    current.count()
                )
            }
        }
    }

    fn delta_sec_timed_str(&self, delta_sec: f64) -> String {
        match self {
            DeltaEntry::Counter { delta, .. } => {
                format!("{}/s", (*delta as f64) / delta_sec)
            }
            DeltaEntry::Gauge { delta, .. } => format!("{:+}", delta),
            DeltaEntry::Histogram { delta, .. } => {
                format!(
                    "[{}]#{:+}",
                    quantile_strings(&delta.quantiles, true).join(", "),
                    delta.new_samples
                )
            }
        }
    }
}

fn quantile_strings(quantiles: &[(Quantile, f64)], format_signed: bool) -> Vec<String> {
    if format_signed {
        quantiles
            .iter()
            .map(|(q, v)| format!("{}={:+}", q.label(), v))
            .collect()
    } else {
        quantiles
            .iter()
            .map(|(q, v)| format!("{}={}", q.label(), v))
            .collect()
    }
}

fn collect_quantiles(quantiles: &[Quantile], summary: &Summary) -> Box<[(Quantile, f64)]> {
    quantiles
        .iter()
        .map(|q| {
            let v = summary.quantile(q.value()).unwrap_or(0.0);
            (q.clone(), v)
        })
        .collect()
}

#[allow(clippy::needless_lifetimes)]
fn ignore<'a, T: 'static>(_t: &'a T) {
    // do nothing
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::*;
    use std::time::Duration;

    #[test]
    fn not_a_real_test() {
        #[allow(unused_mut)]
        let mut rec = PrintRecorder::default();
        // uncomment to see units and descriptions
        //rec.do_print_medata();
        rec.install().unwrap();

        register_counter!("test.counter", "A simple counter in a loop", "test" => "not_a_real_test");
        register_gauge!("test.time_elapsed", Unit::Milliseconds, "The time that elapsed since starting the loop", "test" => "not_a_real_test");
        register_histogram!("test.time_per_iter", Unit::Nanoseconds, "The time that elapsed for every loop", "test" => "not_a_real_test");
        let start = Instant::now();
        let mut elapsed = start.elapsed();
        let mut last_elapsed = Duration::new(0, 0);
        while elapsed < Duration::from_secs(5) {
            let since_last_iter = elapsed - last_elapsed;
            increment_counter!("test.counter", "test" => "not_a_real_test");
            gauge!("test.time_elapsed", elapsed.as_millis() as f64, "test" => "not_a_real_test");
            histogram!("test.time_per_iter", since_last_iter.as_nanos() as f64, "test" => "not_a_real_test");
            last_elapsed = elapsed;
            elapsed = start.elapsed();
        }
    }
}
