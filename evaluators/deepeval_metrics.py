from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric # type: ignore

def get_relevance_metric():
    return AnswerRelevancyMetric(threshold=0.75)

def get_faithfulness_metric():
    return FaithfulnessMetric(threshold=0.7)