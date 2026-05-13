"""
cost_module.py

Provides hourly prices for (provider, region, instance_type).

Modes:
- PRICE_SOURCE_MODE = "live"       -> fetch via fetch_prices.get_prices_live()
- PRICE_SOURCE_MODE = "csv"        -> read from output/prices.csv
- PRICE_SOURCE_MODE = "simulated"  -> deterministic price changes by decision cycle
"""

import csv
from typing import Dict, Tuple, Any
import importlib

import fetch_prices
from fetch_prices import get_prices_live, CSV_PRICE_PATH as FETCH_CSV_PATH


# =========================
# CONFIG MODE
# =========================
# Choose ONE:
# PRICE_SOURCE_MODE = "live"
# PRICE_SOURCE_MODE = "csv"
PRICE_SOURCE_MODE = "csv"  # change to "simulated" when you want the auto-switch demo

CSV_PRICE_PATH = FETCH_CSV_PATH  # "output/prices.csv"


# =========================
# INTERNAL STATE
# =========================
_price_table: Dict[Tuple[str, str, str], float] = {}
_cpu_capacity_table: Dict[Tuple[str, str, str], int] = {}
_prices_loaded: bool = False

# Simulated mode cycle counter (increments once per controller cycle when refresh_prices() called)
_sim_cycle: int = 0


# =========================
# LOADERS
# =========================
def _load_prices_from_csv(path: str = CSV_PRICE_PATH) -> None:
    global _price_table, _cpu_capacity_table, _prices_loaded

    table_price: Dict[Tuple[str, str, str], float] = {}
    table_cpu: Dict[Tuple[str, str, str], int] = {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["provider"].strip().lower(),
                row["region"].strip().lower(),
                row["instance_type"].strip().lower(),
            )
            hourly = float(row["price_per_hour_usd"])
            cpu_cap = int(row.get("cpu_capacity_vcpu", 0))

            table_price[key] = hourly
            table_cpu[key] = cpu_cap

    _price_table = table_price
    _cpu_capacity_table = table_cpu
    _prices_loaded = True


def _load_prices_from_live() -> None:
    global _price_table, _cpu_capacity_table, _prices_loaded

    # Ensure we load the latest code if you edited fetch_prices.py
    importlib.reload(fetch_prices)

    live_dict = get_prices_live()

    table_price: Dict[Tuple[str, str, str], float] = {}
    table_cpu: Dict[Tuple[str, str, str], int] = {}

    for _, row in live_dict.items():
        key = (
            row["provider"].strip().lower(),
            row["region"].strip().lower(),
            row["instance_type"].strip().lower(),
        )
        table_price[key] = float(row["price_per_hour_usd"])
        table_cpu[key] = int(row.get("cpu_capacity_vcpu", 0))

    _price_table = table_price
    _cpu_capacity_table = table_cpu
    _prices_loaded = True


def _get_prices_simulated(cycle: int) -> Dict[str, Dict[str, Any]]:
    """
    Deterministic simulated pricing by cycle.

    Pattern (repeats every 9 cycles):
      cycles 0-2: OpenStack cheap
      cycles 3-5: OpenStack expensive
      cycles 6-8: OpenStack medium
    AWS stays constant (or you can also vary it).

    This is perfect for a thesis demo because it is reproducible.
    """
    cycle_mod = cycle % 9

    aws_price = 0.0216  # keep fixed for demo
    if cycle_mod <= 2:
        os_price = 0.0100
    elif cycle_mod <= 5:
        os_price = 0.0300
    else:
        os_price = 0.0150

    return {
        "aws": {
            "provider": "aws",
            "region": "eu-north-1",
            "instance_type": "t3.small",
            "price_per_hour_usd": aws_price,
            "cpu_capacity_vcpu": 2,
        },
        "openstack": {
            "provider": "openstack",
            "region": "oslo",
            "instance_type": "aem.2c2r.50g",
            "price_per_hour_usd": os_price,
            "cpu_capacity_vcpu": 2,
        },
    }


def _load_prices_from_simulated() -> None:
    global _price_table, _cpu_capacity_table, _prices_loaded

    sim_dict = _get_prices_simulated(_sim_cycle)

    table_price: Dict[Tuple[str, str, str], float] = {}
    table_cpu: Dict[Tuple[str, str, str], int] = {}

    for _, row in sim_dict.items():
        key = (
            row["provider"].strip().lower(),
            row["region"].strip().lower(),
            row["instance_type"].strip().lower(),
        )
        table_price[key] = float(row["price_per_hour_usd"])
        table_cpu[key] = int(row.get("cpu_capacity_vcpu", 0))

    _price_table = table_price
    _cpu_capacity_table = table_cpu
    _prices_loaded = True


def _ensure_prices_loaded() -> None:
    global _prices_loaded

    if _prices_loaded:
        return

    if PRICE_SOURCE_MODE == "csv":
        _load_prices_from_csv()
    elif PRICE_SOURCE_MODE == "live":
        _load_prices_from_live()
    elif PRICE_SOURCE_MODE == "simulated":
        _load_prices_from_simulated()
    else:
        raise ValueError(f"Unknown PRICE_SOURCE_MODE={PRICE_SOURCE_MODE!r}")


# =========================
# PUBLIC API
# =========================
def refresh_prices() -> None:
    """
    Call this once per controller cycle.

    - In live mode: allows new API fetch.
    - In csv mode: allows re-read of updated CSV.
    - In simulated mode: increments the simulated cycle (so prices change automatically).
    """
    global _prices_loaded, _sim_cycle

    if PRICE_SOURCE_MODE == "simulated":
        _sim_cycle += 1

    _prices_loaded = False
    _ensure_prices_loaded()


def get_sim_cycle() -> int:
    """Useful for logging/graphs in simulated mode."""
    return _sim_cycle


def get_hourly_price(provider: str, region: str, instance_type: str, force_refresh: bool = False) -> float:
    if force_refresh:
        refresh_prices()
    else:
        _ensure_prices_loaded()

    key = (provider.lower(), region.lower(), instance_type.lower())
    if key not in _price_table:
        raise ValueError(f"Price not found for provider={provider}, region={region}, instance_type={instance_type}")
    return _price_table[key]


def get_cpu_capacity(provider: str, region: str, instance_type: str) -> int:
    _ensure_prices_loaded()
    key = (provider.lower(), region.lower(), instance_type.lower())
    if key not in _cpu_capacity_table:
        raise ValueError(f"CPU capacity not found for provider={provider}, region={region}, instance_type={instance_type}")
    return _cpu_capacity_table[key]


def estimate_cost_for_hours(provider: str, region: str, instance_type: str, hours: float) -> float:
    hourly = get_hourly_price(provider, region, instance_type)
    return hourly * hours


def get_all_prices() -> Dict[Tuple[str, str, str], float]:
    _ensure_prices_loaded()
    return dict(_price_table)