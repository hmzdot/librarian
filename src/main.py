import argparse
from rag_chain import generate_response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("vault_path", type=str)
    parser.add_argument("search_query", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    response_stream = generate_response(args.search_query, args.vault_path)
    for token in response_stream:
        print(token.content, end="", flush=True)


if __name__ == "__main__":
    main()
