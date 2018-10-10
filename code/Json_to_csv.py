import json
import os
import pandas as pd


def json_to_csv(path):
    output = []
    dic = {}
    # path = '../data/r5'
    #files_name = os.listdir(path)

    os.chdir(path)
    for sub_path,d,filelist in os.walk(path):
        for files in filelist:
            if (files.endswith('.json')):
                fpth = os.path.join(sub_path, files)
                f = open(fpth, 'rb')
                data = json.load(f)
                for i in data['result']['docs']:

                    output.append([i['id'], i['title'], ' '.join(i['content'].split()), i['summary']])
                    #
                    # try:
                    #     for j in i['sentiment']['entities']:
                    #         if j['type'] not in dic.keys():
                    #             dic[j['type']] = []
                    #         dic[j['type']].append(j['value'])
                    # except:
                    #     print('document {} has no sentiment'.format(i['id']))
                    #
                    # try:
                    #     for j in i['semantics']['entities']:
                    #         if j['properties'][0]['value'] not in dic.keys():
                    #             dic[j['properties'][0]['value']] = []
                    #         if j['properties'][1]['value'] not in dic.values():
                    #             dic[j['properties'][0]['value']].append(j['properties'][1]['value'])
                    # except:
                    #     print('document {} has no semantics'.format(i['id']))

    # print(dic.keys())
    # for i in dic.keys():
    #     with open('output/{}.txt'.format(i), 'w') as f:
    #         for j in dic[i]:
    #             f.write(j)
    #             f.write(',')
    #     f.close()

    # csv: id,title,content,summary
    if output == []:
        return False
    else:
        if not os.path.exists('output'):
            os.mkdir('output')
        output_data = pd.DataFrame(output, columns=['id', 'title', 'content', 'summary'])
        print(output_data.shape)
        output_data = output_data.drop_duplicates('content')
        output_data.to_csv('output/data.csv', index=False, encoding='utf-8')
        print(output_data.shape)
        return True

if __name__ == '__main__':
    print(json_to_csv('/Users/nick/Files/USYD/Y2S2/Capstone/data/alldata'))