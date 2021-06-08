from pathlib import Path
from argparse import ArgumentParser
from datastreams import DataStream


def main():
    parser = ArgumentParser(description="Load lifelong data streams")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset_size", default=10000, type=int)
    parser.add_argument("--testset_size", default=1000, type=int)
    parser.add_argument("--save_as_csv", action="store_true")
    parser.add_argument(
        "--stream", 
        default="standard", 
        type=str, 
        choices=[
            "standard", 
            "long", 
            "forgetting", 
            "intransigence",
            "final_accuracy", 
            "aultc", 
            "multidomain_A", 
            "multidomain_B", 
            "multilingual_A", 
            "multilingual_B"
        ]
    )
    args = parser.parse_args()
    if args.stream=="standard":
        stream = ["boolq", "udpos", "wic", "few_rel", "yelp_review"]
    elif args.stream=="long":
        stream = [
            "udpos", "pan_ner", "wic", "few_rel", "ag_news", 
            "ans_topic", "dbpedia", "reviews", "yelp_review", "boolq"
        ]
    elif args.stream=="forgetting":
        stream = ["few_rel", "udpos", "wic", "yelp_review", "boolq"]
    elif args.stream=="intransigence":
        stream = ["yelp_review", "udpos", "wic", "boolq", "few_rel"]
    elif args.stream=="final_accuracy":
        stream = ["wic", "udpos", "boolq", "few_rel", "yelp_review"]
    elif args.stream=="aultc":
        stream = ["udpos", "wic", "few_rel", "yelp_review", "boolq"]
    elif args.stream=="multidomain_A":
        stream = ["yelp_review", "ag_news", "dbpedia", "reviews", "ans_topic"]
    elif args.stream=="multidomain_B":
        stream = ["mnli_fiction", "mnli_government", "mnli_slate", "mnli_telephone", "mnli_travel"]
    elif args.stream=="multilingual_A":
        stream = ["udpos_ar", "udpos_hi", "udpos_tr", "udpos_fi", "udpos_zh"]
    elif args.stream=="multilingual_B":
        stream = ["pan_ner_ar", "pan_ner_bn", "pan_ner_tr", "pan_ner_fi", "pan_ner_zh"]
    
    datastream = DataStream(stream, "train_split")
    teststream = DataStream(stream, "test_split")
    datastream.shuffle_datasets(args.seed)
    datastream.resize_datasets(args.dataset_size)
    teststream.limit_datasets(args.testset_size)

    print("Training Examples")
    print(datastream.sample_examples())
    print("Test Stream Summary")
    print(teststream.summary())

    if args.save_as_csv:
        path = Path(f"data/{args.stream}")
        path.mkdir(exist_ok=True, parents=True)
        datastream.save(path/"train")
        teststream.save(path/"test")
        print(f"{args.stream} stream saved as csv at {path}")


if __name__=="__main__":
    main()