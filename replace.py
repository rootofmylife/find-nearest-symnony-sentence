with open('./Roy_VnTokenizer/scripts/word_segment.txt') as f:
    mylist = f.read().splitlines()

mylist = [line.split(' ') for line in mylist]
removed_b_list = []

for s in mylist:
    temp_arr = []
    word_more_than_2 = []
    global c1
    c1 = 0
    global c2
    c2 = 0
    for item in s:
        c1 += item.count('[')
        c2 += item.count(']')
        if ((c1 + c2) == 2) and (len(word_more_than_2) == 0):
            temp_arr.append(item.replace('[','').replace(']',''))
            c1 = c2 = 0
        elif ((c1 + c2) == 2) and (len(word_more_than_2) != 0):
            temp_s = ""
            for i in word_more_than_2:
                temp_s += i + "_"
            temp_s += item
            temp_arr.append(temp_s.replace('[','').replace(']',''))
            c1 = c2 = 0
            word_more_than_2 = []
        elif (c1 + c2) < 2:
            word_more_than_2.append(item)
    removed_b_list.append(temp_arr)

removed_b_list = [' '.join(s) for s in removed_b_list]

removed_b_list = [s.split(' ') for s in removed_b_list]

mylist = removed_b_list

output = open('text_not_remove_stop_word.txt', 'w')

for i in mylist:
    for j in i:
        output.write(j + ' ')
    output.write('\n')

output.close()
