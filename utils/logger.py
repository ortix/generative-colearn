from collections import defaultdict
nested_dict = lambda: defaultdict(nested_dict)
logger = nested_dict()