"""
tools/load_test.py — Simple load test for the /predict endpoint.
Usage: python tools/load_test.py --url http://localhost:8000 --n 100
"""
import argparse
import json
import random
import statistics
import time
import urllib.request
import urllib.error

def random_payload():
    return {
        "annual_income"      : random.uniform(20000, 200000),
        "age"                : random.uniform(18, 75),
        "debt_to_income"     : random.uniform(0, 0.9),
        "credit_history_yrs" : random.uniform(0, 30),
        "num_open_accounts"  : random.randint(0, 15),
        "missed_payments"    : random.randint(0, 10),
        "loan_amount"        : random.uniform(1000, 100000),
        "employment_years"   : random.uniform(0, 30),
    }

def send_request(url, payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            latency = (time.time() - start) * 1000
            return latency, resp.status, None
    except urllib.error.HTTPError as e:
        latency = (time.time() - start) * 1000
        return latency, e.code, str(e)
    except Exception as e:
        latency = (time.time() - start) * 1000
        return latency, 0, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--n",   type=int, default=100)
    args = parser.parse_args()

    print(f"Load testing {args.url} with {args.n} requests...\n")

    latencies = []
    errors    = 0
    status_counts = {}

    for i in range(args.n):
        payload = random_payload()
        latency, status, error = send_request(args.url, payload)
        latencies.append(latency)
        status_counts[status] = status_counts.get(status, 0) + 1
        if error or status != 200:
            errors += 1
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{args.n} requests...")

    latencies.sort()
    print("\n" + "="*50)
    print("LOAD TEST RESULTS")
    print("="*50)
    print(f"Total requests : {args.n}")
    print(f"Errors         : {errors} ({errors/args.n*100:.1f}%)")
    print(f"Status codes   : {status_counts}")
    print(f"\nLatency (ms):")
    print(f"  Min    : {min(latencies):.1f}")
    print(f"  Mean   : {statistics.mean(latencies):.1f}")
    print(f"  Median : {statistics.median(latencies):.1f}")
    print(f"  p95    : {latencies[int(len(latencies)*0.95)]:.1f}")
    print(f"  p99    : {latencies[int(len(latencies)*0.99)]:.1f}")
    print(f"  Max    : {max(latencies):.1f}")
    p95 = latencies[int(len(latencies)*0.95)]
    print(f"\n{'✅ p95 < 200ms TARGET MET' if p95 < 200 else '❌ p95 > 200ms target missed'}")
    print("="*50)

if __name__ == "__main__":
    main()
