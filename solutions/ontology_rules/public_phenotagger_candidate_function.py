def build_ngram_testset_filted(conll_input,Ngram=8):

    fin_genia=io.StringIO(conll_input)
    fout_context=io.StringIO()
    fout_txt=io.StringIO()
   
    index_dict={}
    allentity=[]
    alltext=fin_genia.read().strip().split('\n\n')
    fin_genia.close()
    num_total=0
    for i in range(0,len(alltext)):
        
        lines=alltext[i].split('\n') 
        ori_txt=[]
        for ele in lines:
            seg=ele.split('\t')
            ori_txt.append(seg[0])
        fout_txt.write(' '.join(ori_txt)+'\n')

        sentence_ngram=min(Ngram,len(lines))
        
        fout_context_list=[]
        temp_entity=[]
        temp_pos=[]
        for ngram in range(2,sentence_ngram+1):
            if ngram==1:
                for j in range(0, len(lines)):                 
                    sid=0
                    eid=0
                    for m in range(0,len(lines)):
                        if m==j:
                            sid=m
                            eid=m
                            fout_context_list.append(lines[m]+'\tO\tB')
                            temp_seg=lines[m].split('\t')
                            temp_entity.append(temp_seg[0])
                            temp_pos.append(temp_seg[3])                            
                        else:
                            pass
    #                        print(sentence[m])
#                            fout_context_list.append(lines[m]+'\tO\tO')
                    if pun_filter(temp_entity)==0 and pos_filter(temp_pos,temp_entity)==0:
                        num_total+=1
                        if ' '.join(temp_entity) not in allentity:
                            allentity.append(' '.join(temp_entity))
                        fout_context.write('HP:None\t'+' '.join(temp_entity)+'\n')
                        fout_context.write('\n'.join(fout_context_list)+'\n\n')
                        index_dict[str(num_total)]=[i,sid,eid]
                    temp_entity=[]
                    temp_pos=[]
                    fout_context_list=[]
            elif ngram==2:
                for j in range(0, len(lines)-1):                  
                    sid=0
                    eid=0
                    for m in range(0,len(lines)):
                        if m==j:
                            fout_context_list.append(lines[m]+'\tO\tB')
                            sid=m
                            temp_seg=lines[m].split('\t')
                            temp_entity.append(temp_seg[0])
                            temp_pos.append(temp_seg[3])
                        elif m==j+1:
                            fout_context_list.append(lines[m]+'\tO\tB')
                            eid=m
                            temp_seg=lines[m].split('\t')
                            temp_entity.append(temp_seg[0])
                            temp_pos.append(temp_seg[3])
                        else:
                            pass
#                            fout_context_list.append(lines[m]+'\tO\tO')
                    
                    if pun_filter(temp_entity)==0 and pos_filter(temp_pos,temp_entity)==0:
                        num_total+=1
                        if ' '.join(temp_entity) not in allentity:
                            allentity.append(' '.join(temp_entity))
                        fout_context.write('HP:None\t'+' '.join(temp_entity)+'\n')
                        fout_context.write('\n'.join(fout_context_list)+'\n\n')
                        index_dict[str(num_total)]=[i,sid,eid]
                    temp_entity=[]
                    temp_pos=[]
                    fout_context_list=[]
            else :
                for j in range(0, len(lines)-ngram+1):               
                    sid=0
                    eid=0
                    for m in range(0,len(lines)):
                        if m==j:
                            fout_context_list.append(lines[m]+'\tO\tB')
                            sid=m
                            temp_seg=lines[m].split('\t')
                            temp_entity.append(temp_seg[0])
                            temp_pos.append(temp_seg[3])
                        elif m>j and m<j+ngram-1:
                            fout_context_list.append(lines[m]+'\tO\tB')
                            temp_seg=lines[m].split('\t')
                            temp_entity.append(temp_seg[0])
                            temp_pos.append(temp_seg[2])
                        elif m==j+ngram-1:
                            fout_context_list.append(lines[m]+'\tO\tB')
                            eid=m
                            temp_seg=lines[m].split('\t')
                            temp_entity.append(temp_seg[0])
                            temp_pos.append(temp_seg[3])
                        else:
                            pass
#                            fout_context_list.append(lines[m]+'\tO\tO')
                    
                    if pun_filter(temp_entity)==0 and pos_filter(temp_pos,temp_entity)==0:
                        num_total+=1
                        if ' '.join(temp_entity) not in allentity:
                            allentity.append(' '.join(temp_entity))
                        fout_context.write('HP:None\t'+' '.join(temp_entity)+'\n')
                        fout_context.write('\n'.join(fout_context_list)+'\n\n')
                        index_dict[str(num_total)]=[i,sid,eid]

                    temp_entity=[]
                    temp_pos=[]
                    fout_context_list=[]

    return fout_context.getvalue(),fout_txt.getvalue(),index_dict
