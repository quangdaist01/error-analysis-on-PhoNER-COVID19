import ast
import pandas as pd

colors = {
        'DATE': '#EA2F86',  # https://www.schemecolor.com/bright-rainbow-gradient.php
        'GENDER': '#F09C0A',
        'JOB': '#FAE000',
        'LOCATION': '#93E223',
        'NAME': '#4070D3',
        'O': '#7B7D70',  # màu gray ở https://www.schemecolor.com/rural-stay.php
        'ORGANIZATION': '#493C9E',
        'PATIENT_ID': '#ED3D07',  # màu đỏ https://www.schemecolor.com/bright-summer-beach.php
        'SYMPTOM_AND_DISEASE': '#F2E8D8',  # https://www.schemecolor.com/pastels-for-men.php
        'TRANSPORTATION': '#C4A69B'}


def load_dataset(filepath, concatenate_words=False):
    """
    Load the dataset from the text file
    :param concatenate_words: whether to keep the sample as a single string instance or a list of words
    :param filepath: path to the .txt file
    :return: sentences, labels
    """
    with open(filepath, encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(line.replace('\n', ''))

    sentences = []
    labels = []

    words = []
    label = []
    for line in lines:
        if len(line.split()) < 2:
            if concatenate_words:
                sentences.append(' '.join(words))
                labels.append(' '.join(label))
            else:
                sentences.append(words)
                labels.append(label)
            words = []
            label = []
        else:
            words.append(line.split()[0])
            label.append(line.split()[1])

    return sentences, labels


def get_unique_tags(filepath):
    """
    Get all unique tags that are present in one .txt file
    :param filepath: path to the .txt file
    :return: list of tags
    """
    with open(filepath, encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(line.replace('\n', ''))

    all_tags = set()
    for line in lines:
        if len(line.split()) == 2:
            tag = line.split()[1].replace('B-', '').replace('I-', '')
            all_tags.add(tag)

    return list(all_tags)


def get_tokens_labels_spans(sentences, labels):
    """
    Get all tokens, labels, spans from a specific list of sentences and labels
    :param sentences: list of sentences, each sentence is a string
    :param labels: list of labels (tags) of the corresponding sentenc, each label of a string is also in the string format
    :return: list of tokens, list of labels, list of spans
    """
    all_tokens = []
    all_labels = []
    all_spans = []
    for i in range(len(sentences)):
        hi = sentences[i]
        ho = labels[i]
        span = []
        last_position = 0
        for index, char in enumerate(hi):
            if char == " ":
                span.append([last_position, index])
                last_position = index + 1
        span.append([last_position, index])

        annotation = ho.split()
        tokens = hi.split()

        all_tokens.append([tokens])
        all_labels.append([annotation])
        all_spans.append([span])
    return all_tokens, all_labels, all_spans


def build_display_elements(tokens, annotations, spans):
    # convert the annotations to the format used in displacy
    all_ann = []

    for sent_id, sent_info in enumerate(tokens):
        sent_length = len(tokens[sent_id])

        last_ann = 'O'
        last_start = None
        last_end = None
        for token_id in range(sent_length):
            this_ann = annotations[sent_id][token_id]

            # separated cases:
            if this_ann != last_ann:
                if last_ann != 'O':
                    # write last item
                    new_ent = {}
                    new_ent['start'] = last_start
                    new_ent['end'] = last_end
                    new_ent['label'] = last_ann[2:]
                    all_ann.append(new_ent)

                # record this instance
                last_ann = 'O' if this_ann == 'O' else 'I' + this_ann[1:]
                last_start = spans[sent_id][token_id][0]
                last_end = spans[sent_id][token_id][1]

            else:
                last_ann = this_ann
                last_end = spans[sent_id][token_id][1]

        if last_ann != 'O':
            new_ent = {}
            new_ent['start'] = last_start
            new_ent['end'] = last_end
            new_ent['label'] = last_ann[2:]
            all_ann.append(new_ent)
    return all_ann


errors = ['No Extraction', 'No Annotation', 'Wrong Tag', 'Wrong Range', 'Wrong Range and tag', 'Num correct tags']
PICK_AN_ERROR_INDEX = 0  # 0 means No Extraction and so on.
PICK_A_GOLD_TAG = 'LOCATION'  # Set to None to view all tags


def convert_cell_to_tag(cell):
    """
    Convert each cell of the dataframe to list of tags
    :param cell: List of tuples, each tuple has 2 lists: 1 contains words, the other contains labels
    :return: List of list, each inner list contains, labels from corresponding tuple of the input cell
    """
    tags = []
    for my_tuple in cell:
        tag = list(set(word.replace('B-', '').replace('I-', '') for word in my_tuple[0]))
        tags.append(tag[0])
    return tags


def convert_string_to_list(cell):
    """
    Cells of the imported result file are not interpreted as lists, so each cell has to be manually converted into a list
    :param cell: cell in which the string content can be casted to list
    :return: list
    """
    return ast.literal_eval(cell)


def get_indexes_of_filtered_errors(ERROR_TYPE_DF_PATH, PICK_AN_ERROR, PICK_A_GOLD_TAG):
    """
    :param ERROR_TYPE_DF_PATH: path to the error analysis table file
    :param PICK_AN_ERROR: choose one error in list of errors
    :param PICK_A_GOLD_TAG: choose one tag
    :return: return a list of indices of sentences that have PICK_A_GOLD_TAG tag and have fallen in to an error type
    """

    def count_tag_in_cell(cell):
        if PICK_A_GOLD_TAG == None:
            return len(cell)
        return len([tag for tag in cell if tag == PICK_A_GOLD_TAG])

    df3 = pd.read_csv(ERROR_TYPE_DF_PATH)
    df3 = df3.iloc[:, :-1]

    for error in errors:
        df3[error] = df3[error].map(convert_string_to_list)
        df3[error] = df3[error].map(convert_cell_to_tag)

    df3['Error Counts'] = df3[PICK_AN_ERROR].map(count_tag_in_cell)

    indexes = []
    for i, row in df3.iterrows():
        if row['Error Counts'] != 0:
            indexes.append(i)
    return indexes
