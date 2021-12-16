

def load_dataset(filepath):
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
        sentences.append(' '.join(words))
        labels.append(' '.join(label))
        words = []
        label = []
      else:
        words.append(line.split()[0])
        label.append(line.split()[1])

    return sentences, labels


def get_tokens_labels_spans(sentences, labels):
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
