#!/usr/bin/env python3
"""
HelixForge Performance Test Runner

Runs automated performance tests and generates reports.

Usage:
    python tests/performance/run_performance_tests.py --target http://localhost:8000
    python tests/performance/run_performance_tests.py --target http://localhost:8000 --users 50 --duration 300
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_locust_test(
    target: str,
    users: int = 50,
    spawn_rate: int = 5,
    duration: int = 120,
    output_dir: str = "outputs/performance",
) -> dict:
    """Run Locust performance test and return results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_prefix = f"{output_dir}/test_{timestamp}"
    html_report = f"{output_dir}/report_{timestamp}.html"

    locustfile = Path(__file__).parent / "locustfile.py"

    cmd = [
        "locust",
        "-f", str(locustfile),
        "--host", target,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration}s",
        "--headless",
        "--csv", csv_prefix,
        "--html", html_report,
        "--only-summary",
    ]

    print(f"Running performance test against {target}")
    print(f"  Users: {users}, Spawn rate: {spawn_rate}/s, Duration: {duration}s")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 60,
        )
        elapsed = time.time() - start_time

        print("=" * 60)
        print("PERFORMANCE TEST RESULTS")
        print("=" * 60)
        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        # Parse results from CSV
        stats = parse_stats_csv(f"{csv_prefix}_stats.csv")

        return {
            "success": result.returncode == 0,
            "duration": elapsed,
            "users": users,
            "spawn_rate": spawn_rate,
            "target": target,
            "timestamp": timestamp,
            "stats": stats,
            "csv_files": [
                f"{csv_prefix}_stats.csv",
                f"{csv_prefix}_stats_history.csv",
                f"{csv_prefix}_failures.csv",
            ],
            "html_report": html_report,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Test timed out",
            "duration": duration,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "Locust not installed. Install with: pip install locust",
        }


def parse_stats_csv(csv_path: str) -> dict:
    """Parse Locust stats CSV file."""
    stats = {}

    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return stats

        headers = lines[0].strip().split(",")
        for line in lines[1:]:
            values = line.strip().split(",")
            if len(values) >= len(headers):
                row = dict(zip(headers, values))
                endpoint = row.get("Name", "unknown")
                stats[endpoint] = {
                    "requests": int(row.get("Request Count", 0)),
                    "failures": int(row.get("Failure Count", 0)),
                    "median_response_time": float(row.get("Median Response Time", 0)),
                    "avg_response_time": float(row.get("Average Response Time", 0)),
                    "min_response_time": float(row.get("Min Response Time", 0)),
                    "max_response_time": float(row.get("Max Response Time", 0)),
                    "requests_per_sec": float(row.get("Requests/s", 0)),
                }

    except (FileNotFoundError, ValueError, KeyError):
        pass

    return stats


def run_simple_load_test(target: str, duration: int = 60) -> dict:
    """Run a simple load test without Locust (fallback)."""
    import concurrent.futures
    import urllib.request
    import urllib.error

    print(f"Running simple load test against {target}/health")
    print(f"Duration: {duration}s")

    results = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "response_times": [],
    }

    def make_request():
        try:
            start = time.time()
            with urllib.request.urlopen(f"{target}/health", timeout=10) as response:
                response.read()
            elapsed = time.time() - start
            return True, elapsed
        except Exception:
            return False, 0

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        while time.time() - start_time < duration:
            futures = [executor.submit(make_request) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures):
                success, elapsed = future.result()
                results["total_requests"] += 1
                if success:
                    results["successful_requests"] += 1
                    results["response_times"].append(elapsed)
                else:
                    results["failed_requests"] += 1
            time.sleep(0.1)

    # Calculate statistics
    if results["response_times"]:
        sorted_times = sorted(results["response_times"])
        results["avg_response_time"] = sum(sorted_times) / len(sorted_times)
        results["min_response_time"] = sorted_times[0]
        results["max_response_time"] = sorted_times[-1]
        results["p50_response_time"] = sorted_times[len(sorted_times) // 2]
        results["p95_response_time"] = sorted_times[int(len(sorted_times) * 0.95)]
        results["requests_per_sec"] = results["total_requests"] / duration

    return results


def check_thresholds(stats: dict) -> list:
    """Check if performance meets thresholds."""
    issues = []

    thresholds = {
        "max_avg_response_time_ms": 500,
        "max_p95_response_time_ms": 1000,
        "max_failure_rate_percent": 1.0,
        "min_requests_per_sec": 10,
    }

    aggregated = stats.get("Aggregated", {})

    if aggregated:
        avg_time = aggregated.get("avg_response_time", 0)
        if avg_time > thresholds["max_avg_response_time_ms"]:
            issues.append(
                f"Average response time ({avg_time:.0f}ms) exceeds threshold "
                f"({thresholds['max_avg_response_time_ms']}ms)"
            )

        total = aggregated.get("requests", 0)
        failures = aggregated.get("failures", 0)
        if total > 0:
            failure_rate = (failures / total) * 100
            if failure_rate > thresholds["max_failure_rate_percent"]:
                issues.append(
                    f"Failure rate ({failure_rate:.2f}%) exceeds threshold "
                    f"({thresholds['max_failure_rate_percent']}%)"
                )

        rps = aggregated.get("requests_per_sec", 0)
        if rps < thresholds["min_requests_per_sec"]:
            issues.append(
                f"Requests per second ({rps:.1f}) below threshold "
                f"({thresholds['min_requests_per_sec']})"
            )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Run HelixForge performance tests")
    parser.add_argument(
        "--target",
        default="http://localhost:8000",
        help="Target URL for testing",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=50,
        help="Number of concurrent users",
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        default=5,
        help="Users spawned per second",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/performance",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Run simple load test (no Locust required)",
    )

    args = parser.parse_args()

    if args.simple:
        results = run_simple_load_test(args.target, args.duration)
        print("\n" + "=" * 60)
        print("SIMPLE LOAD TEST RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2))
    else:
        results = run_locust_test(
            target=args.target,
            users=args.users,
            spawn_rate=args.spawn_rate,
            duration=args.duration,
            output_dir=args.output_dir,
        )

        if results.get("success"):
            issues = check_thresholds(results.get("stats", {}))
            if issues:
                print("\n⚠️  Performance issues detected:")
                for issue in issues:
                    print(f"  - {issue}")
                sys.exit(1)
            else:
                print("\n✓ All performance thresholds met")
                if results.get("html_report"):
                    print(f"  Report: {results['html_report']}")
        else:
            print(f"\n❌ Test failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
