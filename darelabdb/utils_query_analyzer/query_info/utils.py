def compare_set_of_objects(list1: list, list2: list) -> bool:
    """Compares 2 lists of objects without considering order"""
    for item in list1:
        if item not in list2:
            return False
        else:
            list2.remove(item)

    if len(list2) == 0:
        return True
