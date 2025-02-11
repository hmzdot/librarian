import argparse
import os
from loader import load_vault
from vecdb import VecDB
from prompt import generate_response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("vault_path", type=str)
    parser.add_argument("search_query", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    folder_name = os.path.basename(args.vault_path)
    embeddings = load_vault(args.vault_path, cache_prefix=folder_name, progress=False)
    vecdb = VecDB.from_embeddings(embeddings)
    generate_response(args.search_query, vecdb)


if __name__ == "__main__":
    main()
