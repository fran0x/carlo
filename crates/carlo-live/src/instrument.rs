//! Parsing Deribit option names such as `BTC-29JAN27-72000-C`.

use anyhow::{Context, Result, bail};
use carlo::Right;

/// Deribit options expire at 08:00 UTC.
const EXPIRY_HOUR_SECS: i64 = 8 * 3600;

#[derive(Debug, Clone, PartialEq)]
pub struct OptionName {
    pub strike: f64,
    pub right: Right,
    /// Expiry as Unix seconds.
    pub expiry: i64,
}

impl OptionName {
    pub fn parse(name: &str) -> Result<Self> {
        let parts: Vec<&str> = name.split('-').collect();
        let [_, date, strike, right] = parts.as_slice() else {
            bail!("{name:?} is not an option name like BTC-29JAN27-72000-C");
        };
        let strike: f64 = strike
            .parse()
            .with_context(|| format!("strike in {name:?}"))?;
        let right = match *right {
            "C" => Right::Call,
            "P" => Right::Put,
            other => bail!("option type {other:?} in {name:?} is not C or P"),
        };
        let expiry = parse_date(date).with_context(|| format!("expiry in {name:?}"))?;
        Ok(Self {
            strike,
            right,
            expiry,
        })
    }

    /// Years from `now` (Unix seconds) to expiry.
    pub fn years_to_expiry(&self, now: i64) -> f64 {
        (self.expiry - now) as f64 / (365.0 * 86_400.0)
    }
}

/// `29JAN27` → Unix seconds at 08:00 UTC that day.
fn parse_date(s: &str) -> Result<i64> {
    const MONTHS: [&str; 12] = [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    ];
    if s.len() < 6 || !s.is_ascii() {
        bail!("{s:?} is not a date like 29JAN27");
    }
    let (day, rest) = s.split_at(s.len() - 5);
    let (month, year) = rest.split_at(3);
    let day: u32 = day.parse()?;
    let month = MONTHS
        .iter()
        .position(|&m| m == month)
        .with_context(|| format!("month {month:?}"))? as u32
        + 1;
    let year: i64 = 2000 + year.parse::<i64>()?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if !(1..=month_days).contains(&day) {
        bail!("day {day} out of range for {s}");
    }
    Ok(days_from_civil(year, month, day) * 86_400 + EXPIRY_HOUR_SECS)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_deribit_name() {
        let o = OptionName::parse("BTC-29JAN27-72000-C").unwrap();
        assert_eq!(o.strike, 72_000.0);
        assert_eq!(o.right, Right::Call);
        // 2027-01-29T08:00:00Z
        assert_eq!(o.expiry, 1_801_209_600);
    }

    #[test]
    fn parses_single_digit_days_and_puts() {
        let o = OptionName::parse("BTC-5DEC26-60000-P").unwrap();
        assert_eq!(o.right, Right::Put);
        // 2026-12-05T08:00:00Z
        assert_eq!(o.expiry, 1_796_457_600);
    }

    #[test]
    fn rejects_non_options() {
        assert!(OptionName::parse("BTC-PERPETUAL").is_err());
        assert!(OptionName::parse("BTC-29XYZ27-72000-C").is_err());
        assert!(OptionName::parse("BTC-29JAN27-72000-X").is_err());
    }

    #[test]
    fn rejects_impossible_dates() {
        assert!(OptionName::parse("BTC-31FEB27-72000-C").is_err());
        assert!(OptionName::parse("BTC-29FEB27-72000-C").is_err());
        assert!(OptionName::parse("BTC-31APR27-72000-C").is_err());
        assert!(OptionName::parse("BTC-29FEB28-72000-C").is_ok());
    }
}
