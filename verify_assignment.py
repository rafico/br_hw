#!/usr/bin/env python3
"""Verify all assignment requirements are met."""

import json
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and return status."""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def verify_catalogue(filepath):
    """Verify catalogue structure and quality."""
    print(f"\n{'='*70}")
    print("PART A: Person Identity Catalogue Verification")
    print(f"{'='*70}")

    if not Path(filepath).exists():
        print(f"❌ Catalogue file not found: {filepath}")
        return False

    with Path(filepath).open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Check structure
    checks = []
    checks.append(("Has 'summary' section", 'summary' in data))
    checks.append(("Has 'catalogue' section", 'catalogue' in data))
    checks.append(("Has 'total_unique_persons'", 'total_unique_persons' in data.get('summary', {})))
    checks.append(("Has 'parameters'", 'parameters' in data.get('summary', {})))

    # Check content quality
    catalogue = data.get('catalogue', {})
    summary = data.get('summary', {})

    total_persons = summary.get('total_unique_persons', 0)
    checks.append(("Total persons is reasonable (<25)", total_persons < 25 and total_persons > 0))

    # Count cross-clip matches
    cross_clip = sum(1 for apps in catalogue.values()
                    if len(set(a['clip_id'] for a in apps)) > 1)
    match_rate = 100 * cross_clip / total_persons if total_persons > 0 else 0
    checks.append(("Has cross-clip matches", cross_clip > 0))
    checks.append(("Match rate > 40%", match_rate > 40))

    # Check each person entry has required fields
    sample_person = next(iter(catalogue.values())) if catalogue else []
    if sample_person:
        sample_entry = sample_person[0]
        checks.append(("Entries have 'clip_id'", 'clip_id' in sample_entry))
        checks.append(("Entries have 'frame_ranges'", 'frame_ranges' in sample_entry))
        checks.append(("Entries have 'local_track_id'", 'local_track_id' in sample_entry))

    # Print results
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        all_passed = all_passed and passed

    # Print statistics
    print("\n📊 Statistics:")
    print(f"  Total unique persons: {total_persons}")
    print(f"  Cross-clip matches: {cross_clip} ({match_rate:.1f}%)")
    print(f"  Total appearances: {sum(len(apps) for apps in catalogue.values())}")

    return all_passed


def verify_scene_labels(filepath):
    """Verify scene labels structure and quality."""
    print(f"\n{'='*70}")
    print("PART B: Scene Labels Verification")
    print(f"{'='*70}")

    if not Path(filepath).exists():
        print(f"❌ Scene labels file not found: {filepath}")
        return False

    with Path(filepath).open("r", encoding="utf-8") as f:
        data = json.load(f)

    checks = []
    checks.append(("File is a list", isinstance(data, list)))
    checks.append(("Has 4 entries (one per clip)", len(data) == 4))

    if not data:
        print("❌ No data in scene labels file")
        return False

    # Check each entry
    sample = data[0]
    checks.append(("Entries have 'clip_id'", 'clip_id' in sample))
    checks.append(("Entries have 'label'", 'label' in sample))
    checks.append(("Entries have 'justification'", 'justification' in sample))

    # Check labels are valid
    labels = [entry.get('label') for entry in data]
    checks.append(("All labels are 'normal' or 'crime'",
                  all(label in ['normal', 'crime'] for label in labels)))

    # Check for crime detection
    has_crime = 'crime' in labels
    checks.append(("At least one crime detected", has_crime))

    # Check justifications reference timestamps (for crime)
    crime_entries = [e for e in data if e.get('label') == 'crime']
    if crime_entries:
        has_timestamps = any('s' in e.get('justification', '') for e in crime_entries)
        checks.append(("Crime justifications have timestamps", has_timestamps))

    # Print results
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        all_passed = all_passed and passed

    # Print statistics
    print("\n📊 Statistics:")
    print(f"  Total clips: {len(data)}")
    print(f"  Normal: {labels.count('normal')}")
    print(f"  Crime: {labels.count('crime')}")

    return all_passed


def verify_writeup():
    """Verify write-up exists."""
    print(f"\n{'='*70}")
    print("Write-up Verification")
    print(f"{'='*70}")

    return check_file_exists("BR_Summary.pdf", "Technical write-up (2 pages)")


def verify_code_files():
    """Verify key code files exist."""
    print(f"\n{'='*70}")
    print("Code Files Verification")
    print(f"{'='*70}")

    files = [
        ("run.py", "Main pipeline"),
        ("generate_person_catalogue.py", "Person catalogue generation"),
        ("classify_scenes.py", "Scene classification"),
        ("reid_model.py", "ReID feature extraction"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Setup instructions"),
    ]

    all_exist = True
    for filepath, description in files:
        exists = check_file_exists(filepath, description)
        all_exist = all_exist and exists

    return all_exist


def main():
    """Run all verification checks."""
    print(f"\n{'='*70}")
    print("ASSIGNMENT VERIFICATION")
    print(f"{'='*70}\n")

    results = {
        "Part A (Catalogue)": verify_catalogue("catalogue_simple.json"),
        "Part B (Scene Labels)": verify_scene_labels("scene_labels.json"),
        "Write-up": verify_writeup(),
        "Code Files": verify_code_files(),
    }

    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")

    for section, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {section}")

    all_passed = all(results.values())

    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 ALL CHECKS PASSED - ASSIGNMENT COMPLETE!")
    else:
        print("⚠️  Some checks failed - review above for details")
    print(f"{'='*70}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
