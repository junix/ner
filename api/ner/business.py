from entity_detect.ner import load_predict

_pred = load_predict(output_keyword=True)


def get_entity(query_string):
    return _pred(query_string)
