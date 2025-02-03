use std::{collections::HashMap, iter::FromIterator};

use metrics::Key;
use metrics_util::{storage::Summary, CompositeKey, MetricKind, Quantile};

use crate::recorder::MetadataKey;

pub struct Snapshot {
    values: HashMap<CompositeKey, SnapshotValue>,
}

impl Snapshot {
    pub fn clone_summaries(&self) -> HashMap<CompositeKey, Summary> {
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

    pub fn join(&self, quantiles: &[Quantile], other: &Snapshot) -> Vec<DeltaEntry> {
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

pub enum SnapshotValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Box<Summary>),
}

pub struct HistogramDelta {
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

pub enum DeltaEntry {
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
    pub fn key(&self) -> &Key {
        match self {
            DeltaEntry::Counter { key, .. } => key,
            DeltaEntry::Gauge { key, .. } => key,
            DeltaEntry::Histogram { key, .. } => key,
        }
    }

    pub fn metadata_key(&self) -> MetadataKey {
        match self {
            DeltaEntry::Counter { key, .. } => {
                MetadataKey(MetricKind::Counter, key.name().to_owned().into())
            }
            DeltaEntry::Gauge { key, .. } => {
                MetadataKey(MetricKind::Gauge, key.name().to_owned().into())
            }
            DeltaEntry::Histogram { key, .. } => {
                MetadataKey(MetricKind::Histogram, key.name().to_owned().into())
            }
        }
    }
    pub fn key_str(&self) -> String {
        format! {"{}", self.key().name()}
    }

    pub fn kind_str(&self) -> &'static str {
        match self {
            DeltaEntry::Counter { .. } => "Counter",
            DeltaEntry::Gauge { .. } => "Gauge",
            DeltaEntry::Histogram { .. } => "Histogram",
        }
    }

    pub fn current_str(&self) -> String {
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

    pub fn delta_sec_timed_str(&self, delta_sec: f64) -> String {
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
