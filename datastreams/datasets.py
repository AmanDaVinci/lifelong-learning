from datastreams.transforms import *

dataset_configs = {
    "udpos": {
        "path": "universal_dependencies",
        "name": "en_lines",
        "train_split": "train",
        "test_split": "test",
        "transform": udpos
    },
    "pan_ner": {
        "path": "wikiann",
        "name": "en",
        "train_split": "train",
        "test_split": "test",
        "transform": wikiann
    },
    "few_rel": {
        "path": "few_rel",
        "train_split": "train_wiki",
        "test_split": "val_wiki",
        "transform": few_rel
    },
    "record": {
        "path": "super_glue",
        "name": "record",
        "train_split": "train",
        "test_split": "validation",
        "transform": record
    },
    "reviews": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "transform": amazon_reviews
    },
    # --------------------------------------------------------------------------
    # multilingual pos
    # --------------------------------------------------------------------------
    "udpos_ar": {
        "path": "universal_dependencies",
        "name": "ar_padt",
        "train_split": "train",
        "test_split": "test",
        "transform": udpos
    },
    "udpos_hi": {
        "path": "universal_dependencies",
        "name": "hi_hdtb",
        "train_split": "train",
        "test_split": "test",
        "transform": udpos
    },
    "udpos_tr": {
        "path": "universal_dependencies",
        "name": "tr_boun",
        "train_split": "train",
        "test_split": "test",
        "transform": udpos
    },
    "udpos_fi": {
        "path": "universal_dependencies",
        "name": "fi_tdt",
        "train_split": "train",
        "test_split": "test",
        "transform": udpos
    },
    "udpos_zh": {
        "path": "universal_dependencies",
        "name": "zh_gsd",
        "train_split": "train",
        "test_split": "test",
        "transform": udpos
    },
    # --------------------------------------------------------------------------
    # multilingual
    # --------------------------------------------------------------------------
    "pan_ner_ar": {
        "path": "wikiann",
        "name": "ar",
        "train_split": "train",
        "test_split": "test",
        "transform": wikiann
    },
    "pan_ner_bn": {
        "path": "wikiann",
        "name": "bn",
        "train_split": "train",
        "test_split": "test",
        "transform": wikiann
    },
    "pan_ner_tr": {
        "path": "wikiann",
        "name": "tr",
        "train_split": "train",
        "test_split": "test",
        "transform": wikiann
    },
    "pan_ner_fi": {
        "path": "wikiann",
        "name": "fi",
        "train_split": "train",
        "test_split": "test",
        "transform": wikiann
    },
    "pan_ner_zh": {
        "path": "wikiann",
        "name": "zh",
        "train_split": "train",
        "test_split": "test",
        "transform": wikiann
    },
    # --------------------------------------------------------------------------
    # NLI multidomain
    # --------------------------------------------------------------------------
    "mnli_fiction": {
        "path": "multi_nli",
        "train_split": "train", 
        "test_split": "validation_matched", 
        "filter_column": "genre",
        "filter_value": "fiction",
        "transform": mnli
    },
    "mnli_government": {
        "path": "multi_nli",
        "train_split": "train", 
        "test_split": "validation_matched", 
        "filter_column": "genre",
        "filter_value": "government",
        "transform": mnli
    },
    "mnli_slate": {
        "path": "multi_nli",
        "train_split": "train", 
        "test_split": "validation_matched", 
        "filter_column": "genre",
        "filter_value": "slate",
        "transform": mnli
    },
    "mnli_telephone": {
        "path": "multi_nli",
        "train_split": "train", 
        "test_split": "validation_matched", 
        "filter_column": "genre",
        "filter_value": "telephone",
        "transform": mnli
    },
    "mnli_travel": {
        "path": "multi_nli",
        "train_split": "train", 
        "test_split": "validation_matched", 
        "filter_column": "genre",
        "filter_value": "travel",
        "transform": mnli
    },

    # --------------------------------------------------------------------------
    # multidomain
    # --------------------------------------------------------------------------
    "ag_news": {
        "path": "ag_news",
        "train_split": "train", 
        "test_split": "test", 
        "transform": ag_news
    },
    "ans_topic": {
        "path": "yahoo_answers_topics",
        "train_split": "train", 
        "test_split": "test", 
        "transform": yahoo_answers_topics
    },
    "dbpedia": {
        "path": "dbpedia_14",
        "train_split": "train", 
        "test_split": "test", 
        "transform": dbpedia
    },
    "yelp_review": {
        "path": "yelp_review_full",
        "train_split": "train", 
        "test_split": "test", 
        "transform": yelp_review_full
    },
    # --------------------------------------------------------------------------
    # reviews multidomain
    # --------------------------------------------------------------------------
    "reviews_home": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "home",
        "transform": amazon_reviews
    },
    "reviews_apparel": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "apparel",
        "transform": amazon_reviews
    },
    "reviews_wireless": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "wireless",
        "transform": amazon_reviews
    },
    "reviews_beauty": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "beauty",
        "transform": amazon_reviews
    },
    "reviews_drugstore": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "drugstore",
        "transform": amazon_reviews
    },
    "reviews_kitchen": {
        "path": "amazon_reviews_multi",
        "name": "en", 
        "train_split": "train", 
        "test_split": "test", 
        "filter_column": "product_category",
        "filter_value": "kitchen",
        "transform": amazon_reviews
    },
    # --------------------------------------------------------------------------
    # NLI multilingual
    # --------------------------------------------------------------------------
    "mnli": {
        "path": "glue",
        "name": "mnli", 
        "train_split": "train", 
        "test_split": "validation_matched", 
        "transform": mnli
    },
    "xnli": {
        "path": "xtreme",
        "name": "XNLI", 
        "train_split": "validation", 
        "test_split": "test", 
        "transform": xnli
    },
    "xnli_ar": {
        "path": "xtreme",
        "name": "XNLI", 
        "train_split": "validation", 
        "test_split": "test", 
        "filter_column": "language",
        "filter_value": "ar",
        "transform": xnli
    },
    "xnli_de": {
        "path": "xtreme",
        "name": "XNLI", 
        "train_split": "validation", 
        "test_split": "test", 
        "filter_column": "language",
        "filter_value": "de",
        "transform": xnli
    },
    "xnli_es": {
        "path": "xtreme",
        "name": "XNLI", 
        "train_split": "validation", 
        "test_split": "test", 
        "filter_column": "language",
        "filter_value": "es",
        "transform": xnli
    },
    "xnli_hi": {
        "path": "xtreme",
        "name": "XNLI", 
        "train_split": "validation", 
        "test_split": "test", 
        "filter_column": "language",
        "filter_value": "hi",
        "transform": xnli
    },
    "xnli_ru": {
        "path": "xtreme",
        "name": "XNLI", 
        "train_split": "validation", 
        "test_split": "test", 
        "filter_column": "language",
        "filter_value": "ru",
        "transform": xnli
    },
    "xnli_sw": {
        "path": "xtreme",
        "name": "XNLI", 
        "train_split": "validation", 
        "test_split": "test", 
        "filter_column": "language",
        "filter_value": "sw",
        "transform": xnli
    },
    # --------------------------------------------------------------------------
    # superglue
    # --------------------------------------------------------------------------
    "boolq": {
        "path": "super_glue",
        "name": "boolq",
        "train_split": "train",
        "test_split": "validation",
        "transform": boolq
    },
    "multirc": {
        "path": "super_glue",
        "name": "multirc",
        "train_split": "train",
        "test_split": "validation",
        "transform": multirc
    },
    "cb": {
        "path": "super_glue",
        "name": "cb",
        "train_split": "train",
        "test_split": "validation",
        "transform": cb
    },
    "copa": {
        "path": "super_glue",
        "name": "copa",
        "train_split": "train",
        "test_split": "validation",
        "transform": copa
    },
    "rte": {
        "path": "super_glue",
        "name": "rte",
        "train_split": "train",
        "test_split": "validation",
        "transform": rte
    },
    "wic": {
        "path": "super_glue",
        "name": "wic",
        "train_split": "train",
        "test_split": "validation",
        "transform": wic
    },
    "wsc": {
        "path": "super_glue",
        "name": "wsc",
        "train_split": "train",
        "test_split": "validation",
        "transform": wsc
    },
}