import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--attribute_name", default="genre", type=str)
    
    args = parser.parse_args()

    print(args.attribute_name)

    if args.attribute_name == "genre":
        genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
        array, index = pd.factorize(genres_df["genre"])
        genres_df["genre"] = array
        genres_df.groupby("item")["genre"].apply(list).to_json(
            "/opt/ml/input/data/train/Ml_item2attributes_genre.json"
        )

    elif args.attribute_name == "director":
        genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
        item_df = pd.DataFrame(genres_df["item"].unique(), columns = ["item"])
        directors_df = pd.read_csv("../data/train/directors.tsv", sep="\t")
        array, index = pd.factorize(directors_df["director"])
        directors_df["director"] = array
        directors_df = item_df.merge(directors_df, how = "left")
        directors_df.fillna(max(directors_df["director"].unique()) + 1, inplace = True)
        directors_df["director"] = list(map(int, directors_df["director"]))
        directors_df.groupby("item")["director"].apply(list).to_json(
            "/opt/ml/input/data/train/Ml_item2attributes_director.json"
        )

    elif args.attribute_name == "writer":
        genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
        item_df = pd.DataFrame(genres_df["item"].unique(), columns = ["item"])
        writers_df = pd.read_csv("../data/train/writers.tsv", sep="\t")
        array, index = pd.factorize(writers_df["writer"])
        writers_df["writer"] = array
        writers_df = item_df.merge(writers_df, how = "left")
        writers_df.fillna(max(writers_df["writer"].unique()) + 1, inplace = True)
        writers_df["writer"] = list(map(int, writers_df["writer"]))
        writers_df.groupby("item")["writer"].apply(list).to_json(
            "/opt/ml/input/data/train/Ml_item2attributes_writer.json"
        )

if __name__ == "__main__":
    main()
