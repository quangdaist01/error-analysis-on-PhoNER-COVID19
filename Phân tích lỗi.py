import pandas as pd


def load_dataset(filepath):
    with open(filepath, encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(line.replace('\n', ''))

    all_sentences = []
    all_tags = []

    words = []
    tag = []
    for line in lines:
        if len(line.split()) < 2:
            all_sentences.append(words)
            all_tags.append(tag)
            words = []
            tag = []
        else:
            words.append(line.split()[0])
            tag.append(line.split()[1])

    if words:
        all_sentences.append(words)
        all_tags.append(tag)

    return all_sentences, all_tags


def get_unique_tags(filepath):
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


##
class ErrorTypesGold:
    def __init__(self, tags_true, tags_pred, words_true, words_pred):
        self.result = {'No Extraction': [], 'No Annotation': [], 'Wrong Tag': [], 'Wrong Range': [], 'Wrong Range and tag': [], 'Num correct tags': []}
        self.tags_true = tags_true
        self.tags_pred = tags_pred
        self.words_true = words_true
        self.words_pred = words_pred
        self.spans_tags_true, self.spans_O_tags_true = self._get_span_of(tags_true)
        self.spans_tags_pred, self.spans_O_tags_pred = self._get_span_of(tags_pred)

    def _get_span_of(self, tags):
        entities = []
        single_entity = []

        for i, tag in enumerate(tags):
            if 'I-' in tag:
                single_entity.append(i)
            elif single_entity:
                if tag == 'O':
                    entities.append(single_entity)
                    single_entity = []
                elif 'B-' in tag:
                    entities.append(single_entity)
                    single_entity = [i]
            elif 'B-' in tag:
                single_entity.append(i)
        if single_entity:
            entities.append(single_entity)

        empty_spans = []
        empty_span = []

        for i, tag in enumerate(tags):
            if not empty_span and tag == 'O':
                empty_span.append(i)
            elif not any(x in tag for x in ['B-', 'I-']):
                empty_span.append(i)
            elif empty_span and any(x in tag for x in ['B-', 'I-']):

                empty_spans.append(empty_span)
                empty_span = []
        if empty_span:
            empty_spans.append(empty_span)

        return entities, empty_spans

    def _get_tags_true_in(self, span, exclude_o_tag=False):
        start, end = span[0], span[-1] + 1
        tags = self.tags_true[start: end]
        if exclude_o_tag:
            tags = list(set(word.replace('B-', '').replace('I-', '') for word in tags if word != 'O'))
        else:
            tags = list(set(word.replace('B-', '').replace('I-', '') for word in tags))
        return tags

    def _get_tags_pred_in(self, span, exclude_o_tag=False):
        start, end = span[0], span[-1] + 1
        tags = self.tags_pred[start: end]
        if exclude_o_tag:
            tags = list(set(word.replace('B-', '').replace('I-', '') for word in tags if word != 'O'))
        else:
            tags = list(set(word.replace('B-', '').replace('I-', '') for word in tags))
        return tags

    def _is_correct_range(self, span):
        return span in self.spans_tags_pred

    def _is_overlap(self, span, with_O_spans_tags_true=False):

        if not with_O_spans_tags_true:
            for that_span in self.spans_tags_pred:
                if len([bound for bound in span if bound in that_span]) == len(span):
                    return False
        else:
            for that_span in self.spans_O_tags_true:
                if len([bound for bound in span if bound in that_span]) == len(span):
                    return False

        return True

    def check(self):
        for span in self.spans_tags_true:
            start, end = span[0], span[-1] + 1
            tags_pred = self._get_tags_pred_in(span)
            tags_true = self._get_tags_true_in(span)
            raw_tag_true = self.tags_true[start: end]
            raw_tag_pred = self.tags_pred[start: end]
            raw_tag_true_to_end = self.tags_true[start:]
            raw_tag_pred_to_end = self.tags_pred[start:]
            raw_words = self.words_true[start: end]

            if self._is_correct_range(span):
                if tags_pred != tags_true:
                    self.result['Wrong Tag'].append((raw_tag_true, raw_tag_pred, raw_words))
                elif tags_pred == ['O'] * len(tags_pred):
                    self.result['No Extraction'].append((raw_tag_true, raw_tag_pred, raw_words))
                else:
                    self.result['Num correct tags'].append((raw_tag_true, raw_tag_pred, raw_words))
            else:
                if self._is_overlap(span):
                    if raw_tag_pred == ['O'] * len(raw_tag_pred):
                        self.result['No Extraction'].append((raw_tag_true, raw_tag_pred, raw_words))
                    elif tags_true[0] in tags_pred:
                        if 'O' in tags_pred:
                            self.result['Wrong Range'].append((raw_tag_true, raw_tag_pred, raw_words))
                        elif len([tag for tag in raw_tag_pred if 'B-' in tag]) != 1:
                            self.result['Wrong Range'].append((raw_tag_true, raw_tag_pred, raw_words))
                        elif len(raw_tag_pred) >= end + 1 and 'I-' in raw_tag_pred[end]:
                            self.result['Wrong Range'].append((raw_tag_true, raw_tag_pred, raw_words))

                    else:
                        self.result['Wrong Range and tag'].append((raw_tag_true, raw_tag_pred, raw_words))  # print(raw_tag_true, raw_tag_pred, raw_words, sep='\n')  # print(raw_tag_true_to_end)  # print(raw_tag_pred_to_end)  # print('---------------')

                else:
                    if len([tag for tag in raw_tag_pred if 'B-' in tag]) != 1:
                        if tags_true[0] in tags_pred:
                            self.result['Wrong Range'].append((raw_tag_true, raw_tag_pred, raw_words))
                        else:
                            self.result['Wrong Range and tag'].append((raw_tag_true, raw_tag_pred, raw_words))  # print(raw_tag_true, raw_tag_pred, raw_words, sep='\n')  # print(raw_tag_true_to_end)  # print(raw_tag_pred_to_end)  # print('---------------')
                    else:
                        self.result['Num correct tags'].append((raw_tag_true, raw_tag_pred, raw_words))

        for span in self.spans_tags_pred:
            start, end = span[0], span[-1] + 1
            tags_pred = self._get_tags_pred_in(span)
            tags_true = self._get_tags_true_in(span)
            raw_tag_true = self.tags_true[start: end]
            raw_tag_pred = self.tags_pred[start: end]
            raw_words = self.words_true[start: end]
            if not self._is_overlap(span, with_O_spans_tags_true=True):
                self.result['No Annotation'].append((raw_tag_true, raw_tag_pred, raw_words))

        return self


##

all_words_true, all_tags_true = load_dataset(r'C:\Users\quang\PycharmProjects\DL_NLP_TUH\NLP\NER\NER_Error_analysis\Model results\test_true_bilstm.txt')
unique_tags = get_unique_tags(r'C:\Users\quang\PycharmProjects\DL_NLP_TUH\NLP\NER\NER_Error_analysis\Model results\test_true_bilstm.txt')
all_words_pred, all_tags_pred = load_dataset(r'C:\Users\quang\PycharmProjects\DL_NLP_TUH\NLP\NER\NER_Error_analysis\Model results\test_predictions_bilstm.txt')

errors = ['No Extraction', 'No Annotation', 'Wrong Range', 'Wrong Tag', 'Wrong Range and tag', 'Num correct tags']
df = pd.DataFrame(columns=errors)

for i, (tags_true, tags_pred, words_true, words_pred) in enumerate(zip(all_tags_true, all_tags_pred, all_words_true, all_words_pred)):
    error_types = ErrorTypesGold(tags_true, tags_pred, words_true, words_pred).check()
    df = pd.concat([df, pd.DataFrame([error_types.result])])

df.reset_index(drop=True, inplace=True)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Row'}, inplace=True)
df = pd.concat([df, pd.DataFrame(data={'Sentence': all_words_true})], axis=1)
df.to_csv('NLP/NER/NER_Error_analysis/Output/df_error_types_bilstm_gold.csv', index=False)

# print summary
for column in df.columns:
    if column in ['Num correct tags', 'Row']:
        continue
    print(column)
    print(df[column].apply(lambda x: len(x) if x else False).sum())

##
df_new = df.iloc[:, :-1].copy()


# Chuyen list of tuples ve list of string
def convert_cell_to_tag(cell):
    tags = []
    for my_tuple in cell:
        tag = list(set(word.replace('B-', '').replace('I-', '') for word in my_tuple[0]))
        tags.append(tag[0])
    return tags


for error in errors:
    df_new[error] = df_new[error].map(convert_cell_to_tag)

##
# Tao df moi
unique_tags = ['PATIENT_ID', 'NAME', 'AGE', 'GENDER', 'JOB', 'LOCATION', 'ORGANIZATION', 'SYMPTOM_AND_DISEASE', 'TRANSPORTATION', 'DATE', 'O']

df2 = pd.DataFrame(columns=errors)
for tag in unique_tags:
    counts = {}
    for error in errors:
        count_tag = lambda x: len([tag_inside for tag_inside in x if tag_inside == tag])
        temp = df_new[error]
        counts[error] = temp.map(count_tag).sum()
    df2 = pd.concat([df2, pd.DataFrame(data=counts, index=[0])])
# count tung truong hop roi gan do df moi

df2.reset_index(drop=True, inplace=True)
df2 = pd.concat([pd.DataFrame(data={'Tag': unique_tags}), df2], axis=1)

# Tạo cột Errors
num_errors = []
for tag in unique_tags:
    num_errors.append(df2[df2['Tag'] == tag].iloc[:, 1:-1].sum().sum())
df2 = pd.concat([pd.DataFrame(data={'Errors': num_errors}), df2], axis=1)

# tạo cột total
totals = []
for tag in unique_tags:
    totals.append(df2[df2['Tag'] == tag].iloc[:, 2:].sum().sum())
df2 = pd.concat([pd.DataFrame(data={'Total': totals}), df2], axis=1)

# Sắp xếp lại
df2 = df2[['Tag', 'Total', 'Errors', 'No Extraction', 'No Annotation', 'Wrong Range', 'Wrong Tag', 'Wrong Range and tag']]

# Tạo hàng Total
total_row = df2.sum(axis=0).to_dict()
total_row['Tag'] = 'Total'
df2 = pd.concat([df2, pd.DataFrame(total_row, index=[0])])

df2.to_csv('NLP/NER/NER_Error_analysis/Output/df_error_types_bilstm_gold_summary_for_no_ann.csv', index=False)  ##

##
