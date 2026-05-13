"""
fetch_prices.py

Fetch real prices from public cloud APIs (AWS & Azure) and
prepare them for your cost module / decision engine.

It supports two ways of using it:

  MODE 1 (live, in-memory):
      from fetch_prices import get_prices_live
      prices = get_prices_live()

  MODE 2 (CSV/simulated for later use):
      python fetch_prices.py
      -> writes output/prices.csv

Requirements:
    pip install requests
"""

import csv
import os
from typing import Dict, Any

import requests

# Where we will write the CSV for mode 2
CSV_PRICE_PATH = "output/prices.csv"


# ============================
#  AWS PRICE FROM PUBLIC API
# ============================

def _fetch_aws_t3_small_eu_north_1_price() -> float:
    """
    Fetch Linux on-demand hourly price for:
        AWS EC2 t3.small in eu-north-1 (Stockholm)
    using the AWS Price List Bulk API (public, no auth).

    High-level steps:
      1) Download offer index (all AWS services)
      2) Find AmazonEC2 offer
      3) Download region index for EC2
      4) Find eu-north-1 price list URL
      5) Load that JSON and filter for t3.small Linux OnDemand
    """

    # 1) Offer index: lists all services (AmazonEC2, AmazonS3, ...) URL: https://aws.amazon.com/ec2/pricing/on-demand/
    offer_index_url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json"
    offer_index = requests.get(offer_index_url, timeout=180).json()

    ec2_offer = offer_index["offers"]["AmazonEC2"]
    region_index_rel = ec2_offer["currentRegionIndexUrl"]
    region_index_url = "https://pricing.us-east-1.amazonaws.com" + region_index_rel

    # 2) Region index: tells us which URL to use for eu-north-1
    region_index = requests.get(region_index_url, timeout=180).json()
    regions = region_index["regions"]
    region_code = "eu-north-1"

    if region_code not in regions:
        raise RuntimeError(f"eu-north-1 not found in EC2 region index")

    current_version_rel = regions[region_code]["currentVersionUrl"]
    price_list_url = "https://pricing.us-east-1.amazonaws.com" + current_version_rel

    # 3) Region-specific EC2 price list
    price_data = requests.get(price_list_url, timeout=180).json()

    products = price_data["products"]
    terms = price_data["terms"]["OnDemand"]

    target_instance = "t3.small"
    target_sku = None

    # 4) Find the SKU for t3.small Linux shared tenancy
    for sku, product in products.items():
        if product.get("productFamily") != "Compute Instance":
            continue
        attrs = product.get("attributes", {})

        if (
            attrs.get("instanceType") == target_instance
            and attrs.get("operatingSystem") == "Linux"
            and attrs.get("tenancy") == "Shared"
            and attrs.get("capacitystatus", "Used") == "Used"
            and attrs.get("preInstalledSw", "NA") in ("NA", "")
        ):
            target_sku = sku
            break

    if not target_sku:
        raise RuntimeError("Could not find t3.small Linux OnDemand in AWS price list")

    # 5) Get the OnDemand pricePerUnit USD for that SKU
    sku_terms = terms[target_sku]
    # There is usually one termCode and one priceDimension – take first
    for term_code, term_data in sku_terms.items():
        price_dimensions = term_data["priceDimensions"]
        for dim_code, dim in price_dimensions.items():
            usd_str = dim["pricePerUnit"]["USD"]
            return float(usd_str)

    raise RuntimeError("Could not extract AWS pricePerUnit USD for t3.small")


# ============================
#  OPENSTACK PRICE (synthetic)
# ============================

def _get_openstack_price_aem_2c2r_50g() -> float:
    """
    Your university OpenStack does not have a public pricing API,
    so we use a synthetic price (document this in the thesis).

    You can adjust this number later if you want.
    """
    return 0.0100  # USD/hour


# ============================
#  MODE 1: LIVE (dict in memory)
# ============================

def get_prices_live() -> Dict[str, Dict[str, Any]]:
    """
    MODE 1: Call this to get current prices from APIs as a Python dict.

    Example:
        from fetch_prices import get_prices_live
        prices = get_prices_live()
        print(prices["aws"]["price_per_hour_usd"])
    """
    aws_price = _fetch_aws_t3_small_eu_north_1_price()
    openstack_price = _get_openstack_price_aem_2c2r_50g()

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
            "price_per_hour_usd": openstack_price,
            "cpu_capacity_vcpu": 2,
        },
    }


# ============================
#  MODE 2: CSV (for cost_module)
# ============================

def sync_prices_to_csv(csv_path: str = CSV_PRICE_PATH) -> str:
    """
    MODE 2: Fetch live prices and write them into a CSV file.

    The CSV has columns:
        provider,region,instance_type,price_per_hour_usd,cpu_capacity_vcpu

    Returns:
        The path of the CSV that was written.
    """
    prices = get_prices_live()

    # Ensure directory (e.g. 'config/') exists
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        "provider",
        "region",
        "instance_type",
        "price_per_hour_usd",
        "cpu_capacity_vcpu",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in prices.items():
            writer.writerow(
                {
                    "provider": row["provider"],
                    "region": row["region"],
                    "instance_type": row["instance_type"],
                    "price_per_hour_usd": row["price_per_hour_usd"],
                    "cpu_capacity_vcpu": row["cpu_capacity_vcpu"],
                }
            )

    return csv_path


# ============================
#  CLI USAGE (run as script)
# ============================

if __name__ == "__main__":
    # Running this file directly uses MODE 2:
    #   python fetch_prices.py
    # -> calls the APIs
    # -> writes output/prices.csv
    try:
        path = sync_prices_to_csv(CSV_PRICE_PATH)
        print("✅ Price CSV updated:", path)
    except Exception as e:
        print("❌ Failed to update prices:", e)
