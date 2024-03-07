from pathlib import Path

import pandas
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator


def save_file(dir, row):
    with open(dir / str(row.name), mode="w") as f:
        f.write(row.text)


def prepare_data(base_dir, prepared_dir):
    df = pandas.read_csv(
        base_dir / "data" / "raw" / "all-data.csv",
        encoding="iso-8859-1",
        names=["sentiment", "text"],
    )

    tqdm.pandas(desc="preparing files")
    df.progress_apply(lambda row: save_file(dir=prepared_dir, row=row), axis=1)


def _main():
    base_dir = Path(__file__).parents[1]
    prepared_dir = base_dir / "data" / "prepared"
    prepare_data(base_dir=base_dir, prepared_dir=prepared_dir)

    loader = DirectoryLoader(base_dir / "data")
    documents = loader.load()

    generator = TestsetGenerator.with_openai()

    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=10,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    )


if __name__ == "__main__":
    _main()
