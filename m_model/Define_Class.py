def definition_class():
    char_list = []
    f = open("./data/trained_weights/defined_class.txt", "r", encoding="utf-8")
    lines = f.readlines()

    for line in lines:
        char_list.append(line[0])
    f.close()

    return char_list
