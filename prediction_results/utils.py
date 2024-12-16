def cal_HeldOut_metric(annotated_instances):
    num_true=0
    for instance in annotated_instances:
        annotated_preference = instance["annotated_preference"]
        golden_preference = instance["golden_preference"]
        if annotated_preference == golden_preference:
            num_true += 1
    metric={
        "num_true":num_true,
        "num_annotated_instances": len(annotated_instances),
        "acc": num_true/len(annotated_instances),
    }
    return metric