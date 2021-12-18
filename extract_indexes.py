import ast
import pandas as pd

errors = ['No Extraction', 'Wrong Tag', 'Wrong Range', 'Wrong Range and tag', 'Num correct tags']
PICK_AN_ERROR_INDEX = 0  # 0 means No Extraction and so on.
PICK_A_GOLD_TAG = 'LOCATION'  # Set to None to view all tags


def convert_cell_to_tag(cell):
    tags = []
    for my_tuple in cell:
        tag = list(set(word.replace('B-', '').replace('I-', '') for word in my_tuple[0]))
        tags.append(tag[0])
    return tags


def convert_string_to_list(cell):
    return ast.literal_eval(cell)


def get_indexes_of_filtered_errors(ERROR_TYPE_DF_PATH, PICK_AN_ERROR_INDEX, PICK_A_GOLD_TAG):
    def count_tag_in_cell(cell):
        if PICK_A_GOLD_TAG == None:
            return len(cell)
        return len([tag for tag in cell if tag == PICK_A_GOLD_TAG])

    df3 = pd.read_csv(ERROR_TYPE_DF_PATH)
    df3 = df3.iloc[:, :-1]

    for error in errors:
        df3[error] = df3[error].map(convert_string_to_list)
        df3[error] = df3[error].map(convert_cell_to_tag)

    df3['Error Counts'] = df3[errors[PICK_AN_ERROR_INDEX]].map(count_tag_in_cell)

    indexes = []
    for i, row in df3.iterrows():
        if row['Error Counts'] != 0:
            indexes.append(i)
    return indexes

##
# get_indexes_of_filtered_errors(r'C:\Users\quang\PycharmProjects\DL_NLP_TUH\NLP\NER\NER_Error_analysis\Output\df_error_types_PhoBERT_gold.csv',
#                                PICK_AN_ERROR_INDEX,
#                                PICK_A_GOLD_TAG)
