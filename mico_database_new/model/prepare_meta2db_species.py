"""Build a species-level Meta2DB abundance batch from the complete raw profiles.

The profile files contain three values per sample (``cnt``, ``una`` and
``sco``).  This program uses only ``cnt`` and taxonomy rows with a species
assignment, aggregates strain records to their species lineage within each
project, and normalizes each sample to relative abundance.

Output is a sparse, long-format CSV suitable for a controlled database import;
it deliberately does not use the paper-only genus files under ``prepared/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from array import array
from collections import Counter, defaultdict
from pathlib import Path


RANKS = (
    ("superkingdom", "k"),
    ("phylum", "p"),
    ("class", "c"),
    ("order", "o"),
    ("family", "f"),
    ("genus", "g"),
    ("species", "s"),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "meta2db"
    parser = argparse.ArgumentParser(description="Extract a normalized species-level Meta2DB batch.")
    parser.add_argument("--profiles-dir", default=str(data_root / "microbiome_profiles"))
    parser.add_argument("--taxonomy", default=str(data_root / "taxonomy" / "meta2db_taxonomic_lineage_file.csv"))
    parser.add_argument("--metadata", default=str(data_root / "metadata" / "meta2db_metadata.csv"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "meta2db_species"))
    return parser.parse_args()


def clean_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def standard_species_name(record: dict[str, str]) -> str | None:
    if not record.get("species", "").strip():
        return None
    parts = []
    for column, prefix in RANKS:
        value = clean_name(record.get(column, ""))
        if value:
            parts.append(f"{prefix}__{value}")
    if not any(part.startswith("s__") for part in parts):
        return None
    return ".".join(parts)


def project_name_from_profile(path: Path) -> str:
    marker = "_wint2023_"
    if marker not in path.name:
        raise ValueError(f"Unexpected profile filename: {path.name}")
    return path.name.split(marker, 1)[0]


def read_profile_header(path: Path) -> tuple[list[str], list[int], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        value_types = next(reader)
        next(reader)  # tax_id descriptor row
    if len(header) != len(value_types):
        raise ValueError(f"Malformed profile header: {path.name}")
    count_columns = [index for index, value_type in enumerate(value_types) if value_type.strip() == "cnt"]
    profile_samples = [header[index].strip() for index in count_columns]
    if not profile_samples or len(profile_samples) != len(set(profile_samples)):
        raise ValueError(f"Missing or duplicate cnt sample IDs: {path.name}")
    return header, count_columns, profile_samples


def collect_profile_taxids(profiles: list[Path]) -> set[str]:
    taxids: set[str] = set()
    for profile in profiles:
        with profile.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader); next(reader); next(reader)
            for row in reader:
                if row and row[0].strip():
                    taxids.add(row[0].strip())
    return taxids


def load_species_taxonomy(taxonomy_path: Path, profile_taxids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with taxonomy_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            tax_id = record.get("tax_id", "").strip()
            if tax_id not in profile_taxids:
                continue
            standard_name = standard_species_name(record)
            if standard_name:
                mapping[tax_id] = standard_name
    return mapping


def load_metadata(metadata_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    # Meta2DB metadata contains a small number of legacy non-UTF-8 free-text
    # characters.  The fields used below are identifiers and categorical data;
    # replacement keeps the row structure intact without altering abundance.
    with metadata_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            project = row.get("project_name", "").strip()
            filename_match = row.get("filename_match", "").strip()
            if project and filename_match:
                result.setdefault((project, filename_match), row)
    return result


def parse_count(value: str, path: Path, taxid: str) -> float:
    if not value.strip():
        return 0.0
    try:
        count = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid cnt value in {path.name}, tax_id={taxid}: {value!r}") from exc
    if count < 0:
        raise ValueError(f"Negative cnt value in {path.name}, tax_id={taxid}")
    return count


def main() -> None:
    args = parse_args()
    profiles = sorted(Path(args.profiles_dir).glob("*.csv"))
    taxonomy_path = Path(args.taxonomy)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    if not profiles or not taxonomy_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Profiles, taxonomy, or metadata input is missing")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_taxids = collect_profile_taxids(profiles)
    taxonomy = load_species_taxonomy(taxonomy_path, profile_taxids)
    metadata = load_metadata(metadata_path)
    if not taxonomy:
        raise ValueError("No profile tax IDs mapped to a species taxonomy")

    abundance_path = output_dir / "meta2db_species_relative_abundance_long.csv"
    samples_path = output_dir / "meta2db_species_samples.csv"
    abundance_temp = abundance_path.with_suffix(abundance_path.suffix + ".tmp")
    samples_temp = samples_path.with_suffix(samples_path.suffix + ".tmp")
    project_summary: list[dict[str, int | str]] = []
    emitted_rows = 0
    unmatched_metadata = 0
    total_samples = 0
    seen_sample_ids: set[str] = set()

    with abundance_temp.open("w", encoding="utf-8", newline="") as abundance_handle, samples_temp.open(
        "w", encoding="utf-8", newline=""
    ) as samples_handle:
        abundance_writer = csv.writer(abundance_handle, lineterminator="\n")
        samples_writer = csv.writer(samples_handle, lineterminator="\n")
        abundance_writer.writerow(["sample_id", "microbe_name_standard", "abundance_value", "source_taxonomy_rank"])
        samples_writer.writerow([
            "sample_id", "project_name", "profile_sample", "group", "disease", "body_site", "health_disease_stat"
        ])

        for profile in profiles:
            project = project_name_from_profile(profile)
            _, count_columns, profile_samples = read_profile_header(profile)
            sample_ids = [f"{project}|{sample}" for sample in profile_samples]
            if seen_sample_ids.intersection(sample_ids):
                raise ValueError(f"Duplicate full sample IDs in {profile.name}")
            seen_sample_ids.update(sample_ids)
            totals = [0.0] * len(sample_ids)

            # First pass: totals for per-sample species-level normalization.
            with profile.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                next(reader); next(reader); next(reader)
                for row in reader:
                    if not row or row[0].strip() not in taxonomy:
                        continue
                    if len(row) <= max(count_columns):
                        raise ValueError(f"Short data row in {profile.name}, tax_id={row[0]!r}")
                    for index, column in enumerate(count_columns):
                        totals[index] += parse_count(row[column], profile, row[0].strip())
            if any(total == 0 for total in totals):
                raise ValueError(f"Zero species total in {profile.name}")

            # Second pass: aggregate multiple strain/tax_id records into the same species key.
            species_values: dict[str, array] = {}
            with profile.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                next(reader); next(reader); next(reader)
                for row in reader:
                    if not row:
                        continue
                    species_name = taxonomy.get(row[0].strip())
                    if not species_name:
                        continue
                    values = species_values.setdefault(species_name, array("d", [0.0]) * len(sample_ids))
                    for index, column in enumerate(count_columns):
                        values[index] += parse_count(row[column], profile, row[0].strip())

            project_rows = 0
            for species_name, values in species_values.items():
                for index, count in enumerate(values):
                    if count:
                        abundance_writer.writerow([sample_ids[index], species_name, format(count / totals[index], ".12g"), "species"])
                        emitted_rows += 1
                        project_rows += 1

            for full_sample_id, profile_sample in zip(sample_ids, profile_samples):
                record = metadata.get((project, profile_sample.split("_", 1)[0]))
                if record is None:
                    unmatched_metadata += 1
                    record = {}
                samples_writer.writerow([
                    full_sample_id,
                    project,
                    profile_sample,
                    record.get("health_disease_stat", ""),
                    record.get("disease_category", ""),
                    record.get("host_body_product", ""),
                    record.get("health_disease_stat", ""),
                ])
            total_samples += len(sample_ids)
            project_summary.append({"project": project, "samples": len(sample_ids), "nonzero_rows": project_rows})
            print(f"Completed {project}: samples={len(sample_ids)}, rows={project_rows}", flush=True)

    abundance_temp.replace(abundance_path)
    samples_temp.replace(samples_path)
    summary = {
        "profiles": len(profiles),
        "samples": total_samples,
        "profile_tax_ids": len(profile_taxids),
        "species_tax_ids": len(taxonomy),
        "nonzero_species_rows": emitted_rows,
        "metadata_unmatched_samples": unmatched_metadata,
        "taxonomy_rank": "species",
        "normalization_method": "per_sample_species_count",
        "abundance_unit": "relative_abundance",
        "name_format": "k__...p__...g__...s__...",
        "projects": project_summary,
    }
    (output_dir / "meta2db_species_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
