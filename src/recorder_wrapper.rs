use std::sync::Arc;

use metrics::{Key, Recorder, SetRecorderError, Unit};
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

    fn register_counter(&self, key: &Key, metadata: &metrics::Metadata<'_>) -> metrics::Counter {
        //self.0.ensure_register_metadata(MetricKind::Counter, key);
        // now register metric
        self.0
            .registry()
            .get_or_create_counter(key, |c| c.clone().into())
    }

    fn register_gauge(&self, key: &Key, metadata: &metrics::Metadata<'_>) -> metrics::Gauge {
        //self.0.ensure_register_metadata(MetricKind::Gauge, key);
        // now register metric
        self.0
            .registry()
            .get_or_create_gauge(key, |c| c.clone().into())
    }

    fn register_histogram(
        &self,
        key: &Key,
        metadata: &metrics::Metadata<'_>,
    ) -> metrics::Histogram {
        //self.0.ensure_register_metadata(MetricKind::Histogram, key);
        // now register metric
        self.0
            .registry()
            .get_or_create_histogram(key, |c| c.clone().into())
    }
}

/*
impl<P> Recorder for PrintRecorderWrapper<P>
where
    P: Printer + Send + Sync + 'static,
{
    fn register_counter(&self, key: Key, unit: Option<Unit>, description: Option<&'static str>) {
        let key_name = key.name();
        let k = CompositeKey::new(MetricKind::Counter, key);
        self.0.registry.op(k, ignore, Handle::counter);
        let metadata = MetaDataEntry::new(unit, description);
        self.0.insert_metadata(key_name, metadata);
    }

    fn register_gauge(&self, key: Key, unit: Option<Unit>, description: Option<&'static str>) {
        let key_name = key.name();
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

    fn describe_counter(&self, key: metrics::KeyName, unit: Option<Unit>, description: metrics::SharedString) {
        todo!()
    }

    fn describe_gauge(&self, key: metrics::KeyName, unit: Option<Unit>, description: metrics::SharedString) {
        todo!()
    }

    fn describe_histogram(&self, key: metrics::KeyName, unit: Option<Unit>, description: metrics::SharedString) {
        todo!()
    }
}
*/

#[allow(clippy::needless_lifetimes)]
fn ignore<'a, T: 'static>(_t: &'a T) {
    // do nothing
}
