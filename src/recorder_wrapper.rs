use std::sync::Arc;

use metrics::{Key, Recorder, Unit};
use metrics_util::MetricKind;

use crate::{recorder::MetaDataEntry, PrintRecorder, Printer};

pub struct PrintRecorderWrapper<P>(pub(crate) Arc<PrintRecorder<P>>)
where
    P: Printer + Send + Sync + 'static;

impl<P> Recorder for PrintRecorderWrapper<P>
where
    P: Printer + Send + Sync + 'static,
{
    fn describe_counter(
        &self,
        key: metrics::KeyName,
        unit: Option<Unit>,
        description: metrics::SharedString,
    ) {
        self.0.set_metadata_if_missing(
            MetricKind::Counter,
            key,
            MetaDataEntry::new(unit, Some(description)),
        );
    }

    fn describe_gauge(
        &self,
        key: metrics::KeyName,
        unit: Option<Unit>,
        description: metrics::SharedString,
    ) {
        self.0.set_metadata_if_missing(
            MetricKind::Gauge,
            key,
            MetaDataEntry::new(unit, Some(description)),
        );
    }

    fn describe_histogram(
        &self,
        key: metrics::KeyName,
        unit: Option<Unit>,
        description: metrics::SharedString,
    ) {
        self.0.set_metadata_if_missing(
            MetricKind::Histogram,
            key,
            MetaDataEntry::new(unit, Some(description)),
        );
    }

    fn register_counter(&self, key: &Key, _metadata: &metrics::Metadata<'_>) -> metrics::Counter {
        self.0
            .registry()
            .get_or_create_counter(key, |c| c.clone().into())
    }

    fn register_gauge(&self, key: &Key, _metadata: &metrics::Metadata<'_>) -> metrics::Gauge {
        self.0
            .registry()
            .get_or_create_gauge(key, |c| c.clone().into())
    }

    fn register_histogram(
        &self,
        key: &Key,
        _metadata: &metrics::Metadata<'_>,
    ) -> metrics::Histogram {
        self.0
            .registry()
            .get_or_create_histogram(key, |c| c.clone().into())
    }
}
