import re 
import sys
import edlib

segid_pattern = re.compile(r'D(\d+).(\d+)')

def sent2doc_segid(idx_path, sent_path, store_path, sep_tag = '</eos> '):
    r"""recover sentences to documents according to an index file at idx_path, with the format D(\d+).(\d+) = D{doc_id}{sentence position in doc} """
    idx_list = open(idx_path, 'r').read().strip().split('\n')
    sent_list = open(sent_path, 'r').read().strip().split('\n')

    try:
        assert(len(idx_list) == len(sent_list))
    except Exception:
        print('---------\nASSERTION ERROR')
        print(f'idx_length: {len(idx_list)}, path =',idx_path)
        print(f'sent_length: {len(sent_list)}, path =',sent_path)
        return False

    tmp = re.match(segid_pattern, idx_list[0])
    current_ord = int(tmp.group(1))
    
    current_doc = []
    doc_txt = []
    
    for i, idx in enumerate(idx_list):
        tmp = re.match(segid_pattern, idx)
        doc_order = int(tmp.group(1))

        if doc_order != current_ord or i == len(idx_list)-1:
            # last sent of the corpus
            if i == len(idx_list)-1:
                current_doc.append( sent_list[i])
            # append
            doc_txt.append(sep_tag.join(current_doc))

            current_doc = []
            current_ord = doc_order 

        current_doc.append( sent_list[i])


    with open(store_path, 'w') as f:
        f.write('\n'.join(doc_txt))
    
    return True

# TODO combine this with the fonction above
def sent2doc_segid_n(idx_path, sent_path, store_path, n, sep_tag = '</eos> '):
    idx_list = open(idx_path, 'r').read().strip().split('\n')
    sent_list = open(sent_path, 'r').read().strip().split('\n')

    try:
        assert(len(idx_list) == len(sent_list))
    except Exception:
        print('---------\nASSERTION ERROR')
        print(f'idx_length: {len(idx_list)}, path =',idx_path)
        print(f'sent_length: {len(sent_list)}, path =',sent_path)
        return False

    tmp = re.match(segid_pattern, idx_list[0])
    current_ord = int(tmp.group(1))
    
    current_doc = []
    doc_txt = []
    
    for i, idx in enumerate(idx_list):
        tmp = re.match(segid_pattern, idx)
        doc_order = int(tmp.group(1))

        if doc_order != current_ord or i == len(idx_list)-1:
            # doc_txt.append(sep_tag.join(current_doc[:n]))
            doc_txt = doc_txt + current_doc[:n]


            current_doc = []
            current_ord = doc_order 

        current_doc.append( sent_list[i])


    with open(store_path, 'w') as f:
        f.write('\n'.join(doc_txt))
    
    return True




######## with eos idx
def get_split_idx(target_aligned, last_char_idx):
    to_split_list = []
    count = 0
    for i, char in enumerate(target_aligned):
        # print(i, char)
        if char != '-':
            count += 1
            if count-1 in last_char_idx:
                to_split_list.append(i)
    return to_split_list

def idx_only_split_at_blank(query_aligned, to_split_list):
    # to debug (realignent for the last sent)
    # if the char to split is not a blank space, split at the first blank at LEFT of it. 
    # to_split_list corresponds to the blank char indicating the sentence boundaries in the reference sequence
    is_content_list = [i for i in range(len(to_split_list)) if query_aligned[to_split_list[i]].strip() ]
    
    res_list = to_split_list.copy()
    for i in is_content_list:
        to_update = True
        new_idx = to_split_list[i]
        while to_update:
            if query_aligned[new_idx].strip() :
                # not empty char, 
                new_idx = new_idx-1
                if new_idx < 0:
                    res_list[i] = new_idx
                    to_update = False
                    # print('idx < 0')
            else:
                # is blank
                res_list[i] = new_idx
                to_update = False
    return res_list


def re_align_doc2sent_idx(sys_doc, ref_doc, eos_tag, tiret_tag = '<hyp>', sentpiece = False ):
    """
    align MT systems into sentences with respect to the alignment of references, 
    ref_fpath contains documents with sentence boundaries, such as sent1.<sep> sent2.<sep> sent3
    the aligned translations will be stored in sys_store_path.
    """
    sys_doc = re.sub('-', tiret_tag, sys_doc)
    ref_doc = re.sub('-', tiret_tag, ref_doc)
    
    # find all separator defined by eos_tag
    eos_info = re.finditer(eos_tag, ref_doc )
    # list of indice of the begining of eos_tag (i.e. end of each sentence), with eos_tag removed
    last_char_idx = [v.span()[0] - len(eos_tag)*i for i, v in enumerate(eos_info)]

    # remove eos_tag
    sys_doc = re.sub(eos_tag, '', sys_doc)
    ref_doc = re.sub(eos_tag, '', ref_doc)
    
    # char level align 
    align_info = edlib.align(sys_doc, ref_doc, task = 'path') # query, target, task
    nice_align_dict = edlib.getNiceAlignment(align_info, sys_doc, ref_doc) # align_res, query, target
    
    # to split index
    ref_to_split_idx = get_split_idx(nice_align_dict['target_aligned'], last_char_idx)
    sys_to_split_idx = idx_only_split_at_blank(nice_align_dict['query_aligned'], ref_to_split_idx )
    ref_to_split_idx = [-1] + ref_to_split_idx+ [len(nice_align_dict['target_aligned'])]
    sys_to_split_idx = [-1] + sys_to_split_idx + [len(nice_align_dict['query_aligned'])]

    ref_sents = []
    sys_sents = []

    for i in range(len(ref_to_split_idx) -1):
        ref_txt = nice_align_dict['target_aligned'][ref_to_split_idx[i]+1 : ref_to_split_idx[i+1]+1  ] 
        sys_txt = nice_align_dict['query_aligned'][sys_to_split_idx[i]+1 : sys_to_split_idx[i+1]+1  ]
        
        # remove the change/insertion symbol
        ref_txt = re.sub('-', '', ref_txt)
        sys_txt = re.sub('-', '', sys_txt)
        # recover the hyphen
        ref_txt = re.sub(tiret_tag, '-', ref_txt)
        sys_txt = re.sub(tiret_tag, '-', sys_txt)
        
        ref_sents.append(ref_txt.strip())
        sys_sents.append(sys_txt.strip())

    assert(len(ref_sents) == len(sys_sents))
    return ref_sents, sys_sents 


def re_align_doc2sent_file_idx(
        sys_fpath, 
        ref_fpath, 
        sys_store_path, 
        ref_store_path = None, 
        eos_tag = '</eos>'
        ):
    sys_doc = re.sub(' +', ' ', open(sys_fpath, 'r').read().strip()).split('\n')
    ref_doc = re.sub(' +', ' ', open(ref_fpath, 'r').read().strip()).split('\n')
    assert(len(sys_doc) == len(ref_doc))
    
    ref_to_write = []
    sys_to_write = []
    for idx in range(len(sys_doc)):
        if sys_doc[idx].strip() and ref_doc[idx].strip(): 

            ref_sents, sys_sents = re_align_doc2sent_idx(sys_doc[idx], ref_doc[idx], eos_tag, tiret_tag = '<hyp>')        
            ref_to_write += ref_sents
            sys_to_write += sys_sents
        
    with open(sys_store_path, 'w') as f:
        f.write('\n'.join(sys_to_write))
    
    if ref_store_path:
        with open(ref_store_path, 'w') as f:
            f.write('\n'.join(ref_to_write))


########## post process after re align

punct_pattern = re.compile(r"[\.\?]")

def _check_begin_sent(sent_txt, thred = 10):
    if len(sent_txt) <= thred:
        print('thred:', thred)
        print('short sent:', sent_txt)
        return  _check_begin_sent(sent_txt, thred//2)
    
    pre_txt = ''
    if sent_txt[0] == ')':
        pre_txt = ')'
        sent_txt = sent_txt[1:]
    find = [l.span() for l in re.finditer( punct_pattern, sent_txt[:thred])]
    if find:
        if len(find)>1:
            print('multiple punct at the beginning:\n',sent_txt)
            print(find)
        # corret segmentation
        idx = find[0][1]
        pre_txt = sent_txt[:idx]
        sent_txt = sent_txt[idx:]
    return pre_txt, sent_txt


def _check_end_sent(sent_txt, thred = 10):
    if len(sent_txt) <= thred:
        print('thred:', thred)
        print('short sent:', sent_txt)
        return _check_end_sent(sent_txt, thred = thred //2)
    find = [l.span() for l in re.finditer( punct_pattern, sent_txt[-thred:])]
    end_txt = ''
    if find:
        if len(find)>1:
            print('multiple punct at the end:\n',sent_txt)
            print(find)
        # corret segmentation
        idx = find[0][1]
        end_txt = sent_txt[-thred + idx:]
        sent_txt = sent_txt[:-thred+ idx]
    # else:
    #     print('No end punctuation find:', sent_txt)
        
    return end_txt, sent_txt



def check_segmentation_after_realign(sys_path, to_store_path, thred = 10):
    sys_sent_txt = open(sys_path, 'r').read().strip().split('\n')
    for i in range(1, len(sys_sent_txt)):
        # begin
        
        if sys_sent_txt[i][0].isupper() or \
        (not sys_sent_txt[i][0].isalpha() and \
        sys_sent_txt[i][0] not in ['-', ')', "'"]):
            # do nothing if begin with capital letter or digit etc.
            if sys_sent_txt[i][:4] == 'NLP.':
                sys_sent_txt[i-1] = sys_sent_txt[i-1] + ' ' + sys_sent_txt[i][:4]
                sys_sent_txt[i] = sys_sent_txt[i][4:]
                
            
            # check the end part
            if sys_sent_txt[i][-1] in ['.', '?']:
                # do nothing if end with normal punctuation
                continue
            end_txt, sent = _check_end_sent(sys_sent_txt[i], thred = thred)
                    
            if end_txt:
                sys_sent_txt[i+1] = end_txt + ' ' + sys_sent_txt[i+1]
                sys_sent_txt[i] = sent
            continue
        # check the begining of the mis segmented sentences     
        pre_txt, sent = _check_begin_sent(sys_sent_txt[i], thred = thred)
        if pre_txt:
            sys_sent_txt[i-1] = sys_sent_txt[i-1] + ' ' + pre_txt
            sys_sent_txt[i] = sent
                    
        # check the end of the mis segmented sentences     
        if sys_sent_txt[i][-1] in ['.', '?']:
                # do nothing if end with normal punctuation
                continue  
        end_txt, sent = _check_end_sent(sys_sent_txt[i], thred = thred)
        if end_txt:
            sys_sent_txt[i+1] = end_txt + ' ' + sys_sent_txt[i+1]
            sys_sent_txt[i] = sent
    with open(to_store_path, 'w') as f:
        f.write('\n'.join([l.strip() for l in sys_sent_txt]))





##########

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(sys.argv[:6])
        print("Usage: post_proess_utils.py realign sys_fpath, ref_fpath, sys_store_path, ref_store_path")
        sys.exit(-1)

    if sys.argv[1] == 'realign':
        re_align_doc2sent_file_idx(
            sys_fpath = sys.argv[2], 
            ref_fpath = sys.argv[3], 
            sys_store_path  = sys.argv[4], 
            ref_store_path = sys.argv[5], 
            eos_tag = '<sep>'
            )
    # </eos>


    