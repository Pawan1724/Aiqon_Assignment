"""
scripts/audit_dataset.py  –  Run the full dataset audit and generate reports.

Usage:
    python -m scripts.audit_dataset
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_data
from src.data_validation import run_full_audit
from src import CONFIG, logger


def main():
    logger.info("=== Dataset Audit ===")

    # Load raw data
    df = load_raw_data()

    # Determine absolute report directory
    reports_dir = CONFIG["paths"]["reports_dir"]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(reports_dir):
        reports_dir = os.path.join(project_root, reports_dir)

    # Run audit
    results = run_full_audit(df, save_dir=reports_dir)

    # Print summary
    print(f"\nDataset shape: {results['shape']}")
    print(f"Unique shipment IDs: {results['unique_ids']}")
    print(f"Duplicated ID rows: {results['duplicated_id_count']}")
    print(f"Missing targets: {results['missing_targets']}")
    print(f"Timestamp flags: {len(results['timestamp_flags'])} issues")
    print(f"Negative delays: {len(results['negative_delays'])} rows")

    if "status_distribution" in results:
        print(f"Status distribution: {results['status_distribution']}")

    print(f"\nReport saved to: {reports_dir}/data_quality_report.md")
    return results


if __name__ == "__main__":
    main()
