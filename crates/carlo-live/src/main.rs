//! Live Deribit feed (step 8): subscribe to option tickers, reprice each update
//! with Monte Carlo, and report the time from message received to repriced.
//!
//! Read-only: uses Deribit's public market-data channels, no account and no orders.
//!
//! ```text
//! cargo run -p carlo-live --release -- --instrument BTC-29JAN27-72000-C --pin 2
//! ```
//!
//! Everything runs on one thread so that `--pin` pins all of the work. Ctrl-C (or
//! `--max-messages`) stops it and prints the latency percentiles.

mod instrument;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use carlo::{Exec, Sim, Vanilla, black76, gbm};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use hdrhistogram::Histogram;
use serde_json::Value;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use instrument::OptionName;

const URL: &str = "wss://www.deribit.com/ws/api/v2";

#[derive(Parser, Debug)]
#[command(about = "Reprice Deribit BTC options live and measure latency")]
struct Args {
    /// Option to follow, e.g. BTC-29JAN27-72000-C. Repeat for several.
    #[arg(long = "instrument", required = true)]
    instruments: Vec<String>,
    /// Monte Carlo paths per repricing.
    #[arg(long, default_value_t = 100_000)]
    paths: u64,
    /// Pin the thread to this core (Linux; on macOS it is only a hint).
    #[arg(long)]
    pin: Option<usize>,
    /// Stop after this many ticker updates.
    #[arg(long)]
    max_messages: Option<u64>,
    /// Print every update, not just the summary.
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(core) = args.pin {
        pin(core)?;
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building the tokio runtime")?
        .block_on(run(args))
}

fn pin(core: usize) -> Result<()> {
    let cores = core_affinity::get_core_ids().context("listing CPU cores")?;
    let id = *cores
        .get(core)
        .with_context(|| format!("core {core} does not exist ({} cores)", cores.len()))?;
    if core_affinity::set_for_current(id) {
        eprintln!("pinned to core {core}");
    } else {
        eprintln!("could not pin to core {core}: not supported on this OS, continuing unpinned");
    }
    Ok(())
}

async fn run(args: Args) -> Result<()> {
    let options: Vec<(String, OptionName)> = args
        .instruments
        .iter()
        .map(|n| Ok((n.clone(), OptionName::parse(n)?)))
        .collect::<Result<_>>()?;

    let (mut ws, _) = tokio::time::timeout(Duration::from_secs(10), connect_async(URL))
        .await
        .context("connecting to Deribit timed out")?
        .context("connecting to Deribit")?;
    let channels: Vec<String> = options
        .iter()
        .map(|(n, _)| format!("ticker.{n}.100ms"))
        .collect();
    let sub = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "public/subscribe",
        "params": { "channels": channels },
    });
    ws.send(Message::Text(sub.to_string().into()))
        .await
        .context("sending the subscription")?;
    eprintln!("subscribed to {}", channels.join(", "));

    let mut hist = Histogram::<u64>::new(3).context("creating the latency histogram")?;
    let mut seen = 0u64;
    let sim = Sim {
        paths: args.paths,
        seed: 42,
        exec: Exec::Sequential,
    };

    // Created once: a fresh `ctrl_c()` per iteration could miss a signal that arrives
    // while an update is being repriced.
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    loop {
        let msg = tokio::select! {
            msg = ws.next() => msg,
            _ = &mut ctrl_c => break,
        };
        let Some(msg) = msg else {
            bail!("Deribit closed the connection");
        };
        let received = Instant::now();
        let Message::Text(text) = msg.context("reading from the WebSocket")? else {
            continue;
        };
        let v: Value = serde_json::from_str(&text).context("parsing a Deribit message")?;
        if let Some(err) = v.get("error") {
            bail!("Deribit returned an error: {err}");
        }
        if let Some(subscribed) = v.get("result") {
            // Deribit acknowledges with the channels it subscribed; an unknown or expired
            // instrument is left out rather than reported as an error.
            let missing = unsubscribed(&channels, subscribed);
            if !missing.is_empty() {
                bail!("Deribit did not subscribe {}", missing.join(", "));
            }
            continue;
        }
        let Some(data) = v.pointer("/params/data") else {
            continue;
        };
        let tick = Tick::from_json(data)?;
        let Some((_, opt)) = options.iter().find(|(n, _)| *n == tick.instrument) else {
            continue;
        };

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let vanilla = Vanilla {
            k: opt.strike,
            t: opt.years_to_expiry(now),
            right: opt.right,
        };
        let sigma = tick.mark_iv / 100.0;
        let est = gbm::european(tick.forward, &vanilla, sigma, &sim)?;
        let elapsed = received.elapsed();
        hist.record(elapsed.as_micros() as u64)
            .context("recording latency")?;
        seen += 1;

        if args.verbose {
            let exact = black76::price(tick.forward, &vanilla, sigma)?;
            println!(
                "{} F={:.2} iv={:.2}% deribit={:.2} mc={:.2}±{:.2} black76={:.2} latency={}µs",
                tick.instrument,
                tick.forward,
                tick.mark_iv,
                tick.mark_price * tick.forward,
                est.price,
                est.std_err,
                exact,
                elapsed.as_micros(),
            );
        }
        if args.max_messages.is_some_and(|max| seen >= max) {
            break;
        }
    }

    if seen == 0 {
        eprintln!("no ticker updates received");
        return Ok(());
    }
    println!(
        "updates={seen} paths={} latency µs: p50={} p90={} p99={} max={}",
        args.paths,
        hist.value_at_quantile(0.50),
        hist.value_at_quantile(0.90),
        hist.value_at_quantile(0.99),
        hist.max(),
    );
    Ok(())
}

/// The fields of a ticker update this program uses.
struct Tick {
    instrument: String,
    /// Mark price in BTC.
    mark_price: f64,
    /// Forward for the option's expiry, in USD.
    forward: f64,
    /// Mark implied volatility in percent.
    mark_iv: f64,
}

impl Tick {
    fn from_json(data: &Value) -> Result<Self> {
        let num = |key: &str| {
            data.get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("ticker update without numeric {key:?}: {data}"))
        };
        Ok(Self {
            instrument: data
                .get("instrument_name")
                .and_then(Value::as_str)
                .context("ticker update without instrument_name")?
                .to_owned(),
            mark_price: num("mark_price")?,
            forward: num("underlying_price")?,
            mark_iv: num("mark_iv")?,
        })
    }
}

/// The requested channels missing from a `public/subscribe` result.
fn unsubscribed<'a>(requested: &'a [String], result: &Value) -> Vec<&'a str> {
    let granted: Vec<&str> = result
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    requested
        .iter()
        .map(String::as_str)
        .filter(|c| !granted.contains(c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_channels_left_out_of_the_subscription() {
        let requested = vec![
            "ticker.BTC-29JAN27-72000-C.100ms".to_string(),
            "ticker.BTC-29JAN27-99000-C.100ms".to_string(),
        ];
        let all = serde_json::json!(requested);
        assert!(unsubscribed(&requested, &all).is_empty());
        let partial = serde_json::json!(["ticker.BTC-29JAN27-72000-C.100ms"]);
        assert_eq!(
            unsubscribed(&requested, &partial),
            ["ticker.BTC-29JAN27-99000-C.100ms"]
        );
        assert_eq!(unsubscribed(&requested, &Value::Null).len(), 2);
    }
}
