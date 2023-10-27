def flatten_list(nested_list):
    nested_list = nested_list.copy()
    while nested_list:
        sublist = nested_list.pop(0)
        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist
