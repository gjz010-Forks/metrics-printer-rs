use std::{
    collections::HashMap,
    sync::{atomic::Ordering, Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

use crate::{
    recorder_wrapper::PrintRecorderWrapper,
    snapshot::{Snapshot, SnapshotValue},
    Printer,
};
use metrics::{Key, SetRecorderError, Unit};
use metrics_util::{
    registry::{GenerationalAtomicStorage, Registry},
    storage::Summary,
    CompositeKey, MetricKind, Quantile,
};
use std::fmt::Write;

/// The default interval between printing metrics
pub const DEFAULT_PRINT_INTERVAL: Duration = Duration::from_millis(1000);

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct MetadataKey(pub MetricKind, pub metrics::KeyName);

/// A metrics recorder that collects metrics and regularly tried to print them.
pub struct PrintRecorder<P>
where
    P: Printer + Send + Sync + 'static,
{
    registry: Registry<Key, GenerationalAtomicStorage>,
    metadata: RwLock<HashMap<MetadataKey, MetaDataEntry>>,
    printer: P,
    print_interval: Duration,
    print_metadata: bool,
    quantiles: Box<[Quantile]>,
}

impl<P> PrintRecorder<P>
where
    P: Printer + Send + Sync + 'static,
{
    pub(crate) fn registry(&self) -> &Registry<Key, GenerationalAtomicStorage> {
        &self.registry
    }
    /// New PrinterRecorder with 1s interval no metadata printing.
    pub fn new(printer: P) -> Self {
        PrintRecorder {
            registry: Registry::new(GenerationalAtomicStorage::atomic()),
            metadata: RwLock::new(HashMap::new()),
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
    pub fn do_print_metadata(&mut self) -> &mut Self {
        self.print_metadata = true;
        self
    }

    /// Do not print units and descriptions together with the metrics
    pub fn skip_print_metadata(&mut self) -> &mut Self {
        self.print_metadata = true;
        self
    }

    /// Select which quantiles should be printed
    pub fn select_quantiles(&mut self, quantiles: Box<[Quantile]>) -> &mut Self {
        self.quantiles = quantiles;
        self
    }
    /*
    pub(crate) fn ensure_register_metadata(&self, kind: MetricKind, key: &Key) {
        // update metadata
        let mut guard = self
            .metadata
            .write()
            .expect("Could not acquire metadata lock");
        let metadata_key = MetadataKey(kind, key.name().to_owned().into());
        if !guard.contains_key(&metadata_key) {
            guard.insert(
                metadata_key,
                MetaDataEntry {
                    description: None,
                    unit: None,
                },
            );
        }
    }
     */
    pub(crate) fn set_metadata_if_missing(
        &self,
        kind: MetricKind,
        key: metrics::KeyName,
        data: MetaDataEntry,
    ) -> bool {
        let mut guard = self
            .metadata
            .write()
            .expect("Could not acquire metadata lock");
        let k = MetadataKey(kind, key);
        if let Some(x) = guard.get(&k) {
            if x.description.is_some() {
                return false;
            }
        }
        guard.insert(k, data);

        return true;
    }

    fn take_snapshot(&self, mut summaries: HashMap<CompositeKey, Summary>) -> Snapshot {
        let mut snapshot_values = Vec::new();
        self.registry.visit_counters(|key, counter| {
            let value = counter.get_inner().load(Ordering::Relaxed);
            snapshot_values.push((
                CompositeKey::new(MetricKind::Counter, key.clone()),
                SnapshotValue::Counter(value),
            ));
        });
        self.registry.visit_gauges(|key, gauge| {
            let value_bits = gauge.get_inner().load(Ordering::Relaxed);
            let value = f64::from_bits(value_bits);
            snapshot_values.push((
                CompositeKey::new(MetricKind::Gauge, key.clone()),
                SnapshotValue::Gauge(value),
            ));
        });
        self.registry.visit_histograms(|key, histogram| {
            let key2 = CompositeKey::new(MetricKind::Histogram, key.clone());
            let mut summary = summaries
                .remove(&key2)
                .unwrap_or_else(Summary::with_defaults);
            histogram.get_inner().clear_with(|entries| {
                for entry in entries {
                    summary.add(*entry);
                }
            });
            snapshot_values.push((key2, SnapshotValue::Histogram(Box::new(summary))));
        });
        snapshot_values.into_iter().collect()
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
                let guard = self.metadata.read().unwrap();
                if let Some(metadata) = guard.get(&entry.metadata_key()) {
                    let unit = metadata
                        .unit
                        .as_ref()
                        .map(|u| u.as_canonical_label().to_string())
                        .unwrap_or_else(|| "N/A".to_string());
                    longest_unit = longest_unit.max(unit.len());
                    let description = metadata
                        .description
                        .as_ref()
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
        #[allow(unused_must_use)]
        self.install(); // ignore result
    }

    /// Register this recorder as the global recorder,
    /// if no other recorder is already registered.
    ///
    /// If another recorder is registered, this will return an error.
    ///
    /// Also starts the background thread for printing.
    pub fn install(self) -> Result<(), SetRecorderError<Box<PrintRecorderWrapper<P>>>> {
        let arced = Arc::new(self);
        let wrapped = PrintRecorderWrapper(arced.clone());
        // this can still fail due to parallelism
        let res = metrics::set_global_recorder(Box::new(wrapped));
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
}

pub struct MetaDataEntry {
    unit: Option<Unit>,
    description: Option<metrics::SharedString>,
}
impl MetaDataEntry {
    pub fn new(unit: Option<Unit>, description: Option<metrics::SharedString>) -> Self {
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

/// The default set of quantiles to print.
///
/// Prints histograms as [min, median, max].
pub fn default_quantiles() -> Box<[Quantile]> {
    Box::new([Quantile::new(0.0), Quantile::new(0.5), Quantile::new(1.0)])
}
