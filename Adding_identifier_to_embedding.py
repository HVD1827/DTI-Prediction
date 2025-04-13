def adding_identifier_to_identifier():
    with open('./data/input_embedding_dict/ic/dict_ic.list', 'r') as list_file:

        list_lines = list_file.readlines()

    with open('./data/input_embedding_dict/ic/entity_embedding_dt_ic.csv', 'r') as countries_file:

        with open('./data/input_embedding_dict/ic/output.dat', 'w') as output_file:

            for line in countries_file:
                line = line.strip()
                if list_lines:

                    second_column = list_lines.pop(0).strip().split()[1]

                    new_line = f"{second_column} {line}"

                    output_file.write(new_line + '\n')
                else:

                    output_file.write(line + '\n')

    with open('./data/input_embedding_dict/ic/output.dat', 'r') as input_file:

        with open('./data/input_embedding_dict/ic/ic_KGE_drug.dat', 'w') as output_file:
            cmpt = 0
            for line in input_file:
                line = line.strip()
                if line:
                    cmpt = cmpt + 1
                    if cmpt < 170:
                        new_line = 'u' + line

                        output_file.write(new_line + '\n')
                    elif cmpt == 170:
                        output_file.write(new_line)

    with open('./data/input_embedding_dict/ic/output.dat', 'r') as input_file:
        with open('./data/input_embedding_dict/ic/ic_KGE_target.dat', 'w') as output_file:
            cmpt = 0
            for line in input_file:
                line = line.strip()
                if line:
                    cmpt = cmpt + 1
                    if cmpt > 170:
                        new_line = 'i' + line

                        output_file.write(new_line + '\n')
                    elif cmpt == 373:
                        output_file.write(new_line)

