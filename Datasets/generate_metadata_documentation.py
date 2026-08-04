"""Generate metadata documentation reports from global master metadata."""

from pathlib import Path

from reporting.metadata_documentation import MetadataDocumentationGenerator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    eda_docs_dir = root / "docs" / "eda"
    generator = MetadataDocumentationGenerator(
        metadata_path=root / "Datasets" / "processed" / "global_master_metadata.csv",
        metadata_dictionary_md_path=eda_docs_dir / "metadata_dictionary.md",
        metadata_dictionary_csv_path=eda_docs_dir / "metadata_dictionary.csv",
        reports_dir=eda_docs_dir,
    )
    outputs = generator.generate_all()
    for output in outputs:
        print(output.relative_to(root))


if __name__ == "__main__":
    main()
